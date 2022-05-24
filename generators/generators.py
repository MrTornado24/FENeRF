"""Implicit generator for 3D volumes"""

import random
from siren.siren import sample_from_3dgrid
import torch.nn as nn
import torch
import time
import curriculums
from torch.cuda.amp import autocast
from .util import gather_points, scatter_points
from .volumetric_rendering import *

class ImplicitGenerator3d(nn.Module):
    def __init__(self, siren, z_dim, output_dim, neural_renderer_img=None, neural_renderer_seg=None, softmax_label=False, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.siren = siren(output_dim=self.output_dim, z_dim=self.z_dim, input_dim=3, device=None)
        self.epoch = 0
        self.step = 0
        self.channel_dim = self.output_dim - 1
        self.softmax_label = softmax_label # generate label probability
        self.neural_renderer_img = neural_renderer_img
        self.neural_renderer_seg = neural_renderer_seg

    def set_device(self, device):
        self.device = device
        self.siren.device = device

        self.generate_avg_frequencies()

    def forward(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """
        if 'img_feat_size' in kwargs:
            img_size = kwargs['img_feat_size']
        batch_size = z.shape[0]
        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Model prediction on course points
        coarse_output = self.siren(transformed_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, self.output_dim)

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, self.output_dim)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        # Create images with NeRF
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        if self.softmax_label:
            seg, rgb = pixels[..., :-3], pixels[..., -3:]
            seg = torch.nn.Softmax(dim=-1)(seg)
            pixels = torch.cat([seg, rgb], dim=-1)
        pixels = pixels.reshape((batch_size, img_size, img_size, -1))
        if not self.neural_renderer_img and not self.neural_renderer_seg:
            pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)

        if self.neural_renderer_seg:
            pixels = pixels.permute(0,3,1,2).contiguous()
            labels, images = pixels[:, :64], pixels[:, 64:]
            images = self.neural_renderer_img(images)
            labels = self.neural_renderer_seg(labels)
            pixels = torch.cat([labels, images], dim=1)
            pixels = pixels * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)
        
        if self.neural_renderer_img:
            pixels = pixels.permute(0,3,1,2).contiguous()
            pixels = self.neural_renderer_img(pixels)
            pixels = pixels * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)

    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""

        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts


    def staged_forward(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """
        if 'img_feat_size' in kwargs:
            img_size = kwargs['img_feat_size']

        batch_size = z.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            raw_frequencies, raw_phase_shifts = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)

            # points-cam: (batch_size, pixels, num_steps, 3)
            # z_vals: (batch_size, pixels, num_steps, 1)
            # rays_d_cam: (batch_size, pixels, 3)
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # Sequentially evaluate siren with max_batch_size to avoid OOM
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], self.output_dim), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)


            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    ### debug ###
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                    #### Start new importance sampling
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                     num_steps, det=False).detach().to(self.device)
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

                # Sequentially evaluate siren with max_batch_size to avoid OOM
                fine_output = torch.zeros((batch_size, fine_points.shape[1], self.output_dim), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals


            pixels, depth, weights_sum = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), fill_mode=kwargs.get('fill_mode', None), fill_color=kwargs.get('fill_color', 'black'), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            weights_sum = weights_sum.reshape((batch_size, img_size, img_size, self.channel_dim))
            weights_sum = weights_sum.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1
            if self.softmax_label:
                seg, rgb = pixels[..., :-3], pixels[..., -3:]
                seg = torch.nn.Softmax(dim=-1)(seg)
                pixels = torch.cat([seg, rgb], dim=-1)
            pixels = pixels.reshape((batch_size, img_size, img_size, -1))
            if not self.neural_renderer_img and not self.neural_renderer_seg:
                pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
                return pixels, depth_map, weights_sum

            if self.neural_renderer_seg:
                pixels = pixels.permute(0,3,1,2).contiguous()
                labels, images = pixels[:, :64], pixels[:, 64:]
                images = self.neural_renderer_img(images)
                labels = self.neural_renderer_seg(labels)
                pixels = torch.cat([labels, images], dim=1)
                pixels = pixels * 2 - 1
                return pixels, depth_map, weights_sum
            
            if self.neural_renderer_img:
                pixels = pixels.permute(0,3,1,2).contiguous()
                pixels = self.neural_renderer_img(pixels)
                pixels = pixels * 2 - 1
                return pixels, depth_map, weights_sum

    # Used for rendering interpolations
    def staged_forward_with_frequencies(self, truncated_frequencies, truncated_phase_shifts, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=0.7, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        if 'img_feat_size' in kwargs:
            img_size = kwargs['img_feat_size']
        batch_size = truncated_frequencies.shape[0]

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)


            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # BATCHED SAMPLE
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], self.output_dim), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)
            # END BATCHED SAMPLE


            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                        num_steps, det=False).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
                # BATCHED SAMPLE
                fine_output = torch.zeros((batch_size, fine_points.shape[1], self.output_dim), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)
                # END BATCHED SAMPLE

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals


            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), fill_mode=kwargs.get('fill_mode', None), fill_color=kwargs.get('fill_color', 'black'), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            if self.softmax_label:
                seg, rgb = pixels[..., :-3], pixels[..., -3:]
                seg = torch.nn.Softmax(dim=-1)(seg)
                pixels = torch.cat([seg, rgb], dim=-1)
            pixels = pixels.reshape((batch_size, img_size, img_size, -1))
            if not self.neural_renderer_img and not self.neural_renderer_seg:
                pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
                return pixels, depth_map

            if self.neural_renderer_seg:
                pixels = pixels.permute(0,3,1,2).contiguous()
                labels, images = pixels[:, :64], pixels[:, 64:]
                images = self.neural_renderer_img(images)
                labels = self.neural_renderer_seg(labels)
                pixels = torch.cat([labels, images], dim=1)
                pixels = pixels * 2 - 1
                return pixels, depth_map
            
            if self.neural_renderer_img:
                pixels = pixels.permute(0,3,1,2).contiguous()
                pixels = self.neural_renderer_img(pixels)
                pixels = pixels * 2 - 1
                return pixels, depth_map


    def forward_with_frequencies(self, frequencies, phase_shifts, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        if 'img_feat_size' in kwargs:
            img_size = kwargs['img_feat_size']
        batch_size = frequencies.shape[0]

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
        transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1
            
        coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, self.output_dim)
        
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                #### Start new importance sampling
                # RuntimeError: Sizes of tensors must match except in dimension 1. Got 3072 and 6144 (The offending index is 0)
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach() # batch_size, num_pixels**2, num_steps
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                #### end new importance sampling
                
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, self.output_dim)

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        if self.softmax_label:
            seg, rgb = pixels[..., :-3], pixels[..., -3:]
            seg = torch.nn.Softmax(dim=-1)(seg)
            pixels = torch.cat([seg, rgb], dim=-1)
        pixels = pixels.reshape((batch_size, img_size, img_size, -1))
        if not self.neural_renderer_img and not self.neural_renderer_seg:
            pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)

        if self.neural_renderer_seg:
            pixels = pixels.permute(0,3,1,2).contiguous()
            labels, images = pixels[:, :64], pixels[:, 64:]
            images = self.neural_renderer_img(images)
            labels = self.neural_renderer_seg(labels)
            pixels = torch.cat([labels, images], dim=1)
            pixels = pixels * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)
        
        if self.neural_renderer_img:
            pixels = pixels.permute(0,3,1,2).contiguous()
            pixels = self.neural_renderer_img(pixels)
            pixels = pixels * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)


class DoubleImplicitGenerator3d(nn.Module):
    def __init__(self, siren, z_geo_dim, z_app_dim, output_dim, softmax_label=False, **kwargs):
        super().__init__()
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim
        self.output_dim = output_dim
        self.siren = siren(output_dim=self.output_dim, z_geo_dim=self.z_geo_dim, z_app_dim=self.z_app_dim, input_dim=3, device=None)
        self.epoch = 0
        self.step = 0
        self.channel_dim = self.output_dim - 1
        self.softmax_label = softmax_label

    def set_device(self, device):
        self.device = device
        self.siren.device = device

        self.generate_avg_frequencies()

    def forward(self, z_geo, z_app, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size = z_app.shape[0]
        grad_points = kwargs.get('grad_points', img_size*img_size)
        if grad_points != img_size*img_size:
            pixels, poses = self.part_forward(z_geo, z_app, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs)
            return pixels, poses

        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Model prediction on course points
        coarse_output = self.siren(transformed_points, z_geo, z_app, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, self.output_dim)

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(fine_points, z_geo, z_app, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, self.output_dim)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        # Create images with NeRF
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        if self.softmax_label:
            seg, rgb = pixels[..., :-3], pixels[..., -3:]
            seg = torch.nn.Softmax(dim=-1)(seg)
            pixels = torch.cat([seg, rgb], dim=-1)
        pixels = pixels.reshape((batch_size, img_size, img_size, -1))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        return pixels, torch.cat([pitch, yaw], -1)


    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""

        z_geo = torch.randn((10000, self.z_geo_dim), device=self.siren.device)
        z_app  =torch.randn((10000, self.z_app_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies_geo, phase_shifts_geo = self.siren.geo_mapping_network(z_geo)
            frequencies_app, phase_shifts_app = self.siren.app_mapping_network(z_app)

        self.avg_frequencies_geo = frequencies_geo.mean(0, keepdim=True)
        self.avg_phase_shifts_geo = phase_shifts_geo.mean(0, keepdim=True)
        self.avg_frequencies_app = frequencies_app.mean(0, keepdim=True)
        self.avg_phase_shifts_app = phase_shifts_app.mean(0, keepdim=True) 
        return self.avg_frequencies_geo, self.avg_phase_shifts_geo, self.avg_frequencies_app, self.avg_phase_shifts_app


    def staged_forward(self, z_geo, z_app, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

        batch_size = z_app.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            raw_frequencies_geo, raw_phase_shifts_geo = self.siren.geo_mapping_network(z_geo)
            raw_frequencies_app, raw_phase_shifts_app = self.siren.app_mapping_network(z_app)

            truncated_frequencies_geo = self.avg_frequencies_geo + psi * (raw_frequencies_geo - self.avg_frequencies_geo)
            truncated_phase_shifts_geo = self.avg_phase_shifts_geo + psi * (raw_phase_shifts_geo - self.avg_phase_shifts_geo)
            truncated_frequencies_app = self.avg_frequencies_app + psi * (raw_frequencies_app - self.avg_frequencies_app)
            truncated_phase_shifts_app = self.avg_phase_shifts_app + psi * (raw_phase_shifts_app - self.avg_phase_shifts_app)

            # points-cam: (batch_size, pixels, num_steps, 3)
            # z_vals: (batch_size, pixels, num_steps, 1)
            # rays_d_cam: (batch_size, pixels, 3)
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)


            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # Sequentially evaluate siren with max_batch_size to avoid OOM
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], self.output_dim), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies_geo[b:b+1], truncated_frequencies_app[b:b+1], truncated_phase_shifts_geo[b:b+1], truncated_phase_shifts_app[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)


            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                    #### Start new importance sampling
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                     num_steps, det=False).detach().to(self.device)
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

                # Sequentially evaluate siren with max_batch_size to avoid OOM
                fine_output = torch.zeros((batch_size, fine_points.shape[1], self.output_dim), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies_geo[b:b+1], truncated_frequencies_app[b:b+1], truncated_phase_shifts_geo[b:b+1], truncated_phase_shifts_app[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals
                
            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), fill_mode=kwargs.get('fill_mode', None), fill_color=kwargs.get('fill_color', 'black'), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            if self.softmax_label:
                seg, rgb = pixels[..., :-3], pixels[..., -3:]
                seg = torch.nn.Softmax(dim=-1)(seg)
                pixels = torch.cat([seg, rgb], dim=-1)
            pixels = pixels.reshape((batch_size, img_size, img_size, -1))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

        return pixels, depth_map

    # Used for rendering interpolations
    def staged_forward_with_frequencies(self, truncated_frequencies_geo, truncated_frequencies_app, truncated_phase_shifts_geo, truncated_phase_shifts_app, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=0.7, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        batch_size = truncated_frequencies_app.shape[0]

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)


            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # BATCHED SAMPLE
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], self.output_dim), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies_geo[b:b+1], truncated_frequencies_app[b:b+1], truncated_phase_shifts_geo[b:b+1], truncated_phase_shifts_app[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)
            # END BATCHED SAMPLE

            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                        num_steps, det=False).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
                # BATCHED SAMPLE
                fine_output = torch.zeros((batch_size, fine_points.shape[1], self.output_dim), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies_geo[b:b+1], truncated_frequencies_app[b:b+1], truncated_phase_shifts_geo[b:b+1], truncated_phase_shifts_app[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)
                # END BATCHED SAMPLE

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals


            pixels, depth, weights_sum = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), fill_mode=kwargs.get('fill_mode', None), fill_color=kwargs.get('fill_color', 'black'), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            weights_sum = weights_sum.reshape((batch_size, img_size, img_size, -1))
            weights_sum = weights_sum.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1
            if self.softmax_label:
                seg, rgb = pixels[..., :-3], pixels[..., -3:]
                seg = torch.nn.Softmax(dim=-1)(seg)
                pixels = torch.cat([seg, rgb], dim=-1)
            pixels = pixels.reshape((batch_size, img_size, img_size, -1))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

        return pixels, depth_map, weights_sum


    def forward_with_frequencies(self, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        batch_size = frequencies_app.shape[0]

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
        transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)


        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1
            
        coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, self.output_dim)
        
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                #### Start new importance sampling
                # RuntimeError: Sizes of tensors must match except in dimension 1. Got 3072 and 6144 (The offending index is 0)
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach() # batch_size, num_pixels**2, num_steps
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                #### end new importance sampling
                
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, self.output_dim)

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        if self.softmax_label:
            seg, rgb = pixels[..., :-3], pixels[..., -3:]
            seg = torch.nn.Softmax(dim=-1)(seg)
            pixels = torch.cat([seg, rgb], dim=-1)
        pixels = pixels.reshape((batch_size, img_size, img_size, self.channel_dim))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        return pixels, torch.cat([pitch, yaw], -1)

    def point_forward(self, transformed_points, transformed_ray_directions_expanded, transformed_ray_origins, transformed_ray_directions, z_vals, z_geo, z_app, num_steps, hierarchical_sample, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size, num_points = transformed_points.shape[:2]
        transformed_points = transformed_points.reshape(batch_size, -1, 3)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, -1, 3)
        # Model prediction on course points
        coarse_output = self.siren(transformed_points, z_geo, z_app, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, num_points, num_steps, self.output_dim)

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, -1, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(-1, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(-1, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, -1, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, -1, num_steps, 1)
                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, -1, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(fine_points, z_geo, z_app, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, num_points, -1, self.output_dim)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        # Create images with NeRF
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        if self.softmax_label:
            seg, rgb = pixels[..., :-3], pixels[..., -3:]
            seg = torch.nn.Softmax(dim=-1)(seg)
            pixels = torch.cat([seg, rgb], dim=-1)

        return pixels

    def part_forward(self, z_geo, z_app, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size = z_app.shape[0]
        grad_points = kwargs.get('grad_points', img_size*img_size)
        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # split points into w/ gradient and w/o gradient
        num_points = transformed_points.shape[1] # transformed_points: [batch_size, img_size*img_size, num_steps, 3]
        assert num_points > grad_points
        rand_idx = torch.randperm(num_points, device=self.device)
        idx_grad = rand_idx[:grad_points]
        idx_no_grad = rand_idx[grad_points:]

        # forward of points w/ gradient
        points_grad = gather_points(points=transformed_points, idx_grad=idx_grad)
        ray_directions_grad= gather_points(points=transformed_ray_directions_expanded, idx_grad=idx_grad)
        transformed_ray_origins_grad = gather_points(points=transformed_ray_origins, idx_grad=idx_grad)
        transformed_ray_directions_grad = gather_points(points=transformed_ray_directions, idx_grad=idx_grad)
        z_vals_grad = gather_points(points=z_vals, idx_grad=idx_grad)
        pixels_grad = self.point_forward(points_grad, ray_directions_grad, transformed_ray_origins_grad, transformed_ray_directions_grad, z_vals_grad, z_geo, z_app, num_steps, hierarchical_sample, lock_view_dependence=False, **kwargs)
        # forward of points w/o gradient
        points_no_grad = gather_points(points=transformed_points, idx_grad=idx_no_grad)
        ray_directions_no_grad= gather_points(points=transformed_ray_directions_expanded, idx_grad=idx_no_grad)
        transformed_ray_origins_no_grad = gather_points(points=transformed_ray_origins, idx_grad=idx_no_grad)
        transformed_ray_directions_no_grad = gather_points(points=transformed_ray_directions, idx_grad=idx_no_grad)
        z_vals_no_grad = gather_points(points=z_vals, idx_grad=idx_no_grad)
        with torch.no_grad():
            pixels_no_grad = self.point_forward(points_no_grad, ray_directions_no_grad, transformed_ray_origins_no_grad, transformed_ray_directions_no_grad, z_vals_no_grad, z_geo, z_app, num_steps, hierarchical_sample, lock_view_dependence=False, **kwargs)
        # concatenate both parts 
        pixels = scatter_points(idx_grad=idx_grad,
                                        points_grad=pixels_grad,
                                        idx_no_grad=idx_no_grad,
                                        points_no_grad=pixels_no_grad,
                                        num_points=num_points)

        pixels = pixels.reshape((batch_size, img_size, img_size, -1))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        return pixels, torch.cat([pitch, yaw], -1)



class StyleGenerator3d(nn.Module):
    def __init__(self, siren, z_dim, output_dim, neural_renderer_img=None, neural_renderer_seg=None, softmax_label=False, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.siren = siren(output_dim=self.output_dim, z_dim=self.z_dim, input_dim=3, device=None)
        self.epoch = 0
        self.step = 0
        self.channel_dim = self.output_dim - 1
        self.softmax_label = softmax_label # generate label probability
        self.neural_renderer_img = neural_renderer_img
        self.neural_renderer_seg = neural_renderer_seg

    def set_device(self, device):
        self.device = device
        self.siren.device = device

    def forward(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """
        if 'img_feat_size' in kwargs:
            img_size = kwargs['img_feat_size']
        batch_size = z.shape[0]
        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Model prediction on course points
        coarse_output = self.siren(transformed_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, self.output_dim)

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, self.output_dim)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        # Create images with NeRF
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        if self.softmax_label:
            seg, rgb = pixels[..., :-3], pixels[..., -3:]
            seg = torch.nn.Softmax(dim=-1)(seg)
            pixels = torch.cat([seg, rgb], dim=-1)
        pixels = pixels.reshape((batch_size, img_size, img_size, -1))
        if not self.neural_renderer_img and not self.neural_renderer_seg:
            pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)

        if self.neural_renderer_seg:
            pixels = pixels.permute(0,3,1,2).contiguous()
            labels, images = pixels[:, :64], pixels[:, 64:]
            images = self.neural_renderer_img(images)
            labels = self.neural_renderer_seg(labels)
            pixels = torch.cat([labels, images], dim=1)
            pixels = pixels * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)
        
        if self.neural_renderer_img:
            pixels = pixels.permute(0,3,1,2).contiguous()
            pixels = self.neural_renderer_img(pixels)
            pixels = pixels * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)

    def staged_forward(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """
        if 'img_feat_size' in kwargs:
            img_size = kwargs['img_feat_size']
        batch_size = z.shape[0]
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end)
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # Sequentially evaluate siren with max_batch_size to avoid OOM
            coarse_output = self.siren(transformed_points, z, ray_directions=transformed_ray_directions_expanded)
            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)


            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    ### debug ###
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                    #### Start new importance sampling
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                     num_steps, det=False).detach().to(self.device)
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

                # Sequentially evaluate siren with max_batch_size to avoid OOM
                fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded)
                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals


            pixels, depth, weights_sum = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            weights_sum = weights_sum.reshape((batch_size, img_size, img_size, self.channel_dim))
            weights_sum = weights_sum.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1
            if self.softmax_label:
                seg, rgb = pixels[..., :-3], pixels[..., -3:]
                seg = torch.nn.Softmax(dim=-1)(seg)
                pixels = torch.cat([seg, rgb], dim=-1)
            pixels = pixels.reshape((batch_size, img_size, img_size, -1))
            if not self.neural_renderer_img and not self.neural_renderer_seg:
                pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
                return pixels, depth_map, weights_sum

            if self.neural_renderer_seg:
                pixels = pixels.permute(0,3,1,2).contiguous()
                labels, images = pixels[:, :64], pixels[:, 64:]
                images = self.neural_renderer_img(images)
                labels = self.neural_renderer_seg(labels)
                pixels = torch.cat([labels, images], dim=1)
                pixels = pixels * 2 - 1
                return pixels, depth_map, weights_sum
            
            if self.neural_renderer_img:
                pixels = pixels.permute(0,3,1,2).contiguous()
                pixels = self.neural_renderer_img(pixels)
                pixels = pixels * 2 - 1
                return pixels, depth_map, weights_sum

    # Used for rendering interpolations
    def staged_forward_with_frequencies(self, truncated_frequencies, truncated_phase_shifts, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=0.7, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        if 'img_feat_size' in kwargs:
            img_size = kwargs['img_feat_size']
        batch_size = truncated_frequencies.shape[0]

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)


            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # BATCHED SAMPLE
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], self.output_dim), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)
            # END BATCHED SAMPLE


            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                        num_steps, det=False).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
                # BATCHED SAMPLE
                fine_output = torch.zeros((batch_size, fine_points.shape[1], self.output_dim), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, self.output_dim)
                # END BATCHED SAMPLE

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals


            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()
            if self.softmax_label:
                seg, rgb = pixels[..., :-3], pixels[..., -3:]
                seg = torch.nn.Softmax(dim=-1)(seg)
                pixels = torch.cat([seg, rgb], dim=-1)
            pixels = pixels.reshape((batch_size, img_size, img_size, -1))
            if not self.neural_renderer_img and not self.neural_renderer_seg:
                pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
                return pixels, depth_map

            if self.neural_renderer_seg:
                pixels = pixels.permute(0,3,1,2).contiguous()
                labels, images = pixels[:, :64], pixels[:, 64:]
                images = self.neural_renderer_img(images)
                labels = self.neural_renderer_seg(labels)
                pixels = torch.cat([labels, images], dim=1)
                pixels = pixels * 2 - 1
                return pixels, depth_map
            
            if self.neural_renderer_img:
                pixels = pixels.permute(0,3,1,2).contiguous()
                pixels = self.neural_renderer_img(pixels)
                pixels = pixels * 2 - 1
                return pixels, depth_map


    def forward_with_frequencies(self, frequencies, phase_shifts, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        if 'img_feat_size' in kwargs:
            img_size = kwargs['img_feat_size']
        batch_size = frequencies.shape[0]

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
        transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)


        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1
            
        coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, self.output_dim)
        
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                #### Start new importance sampling
                # RuntimeError: Sizes of tensors must match except in dimension 1. Got 3072 and 6144 (The offending index is 0)
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach() # batch_size, num_pixels**2, num_steps
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                #### end new importance sampling
                
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, self.output_dim)

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.output_dim))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), black_back=kwargs.get('black_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        if self.softmax_label:
            seg, rgb = pixels[..., :-3], pixels[..., -3:]
            seg = torch.nn.Softmax(dim=-1)(seg)
            pixels = torch.cat([seg, rgb], dim=-1)
        pixels = pixels.reshape((batch_size, img_size, img_size, -1))
        if not self.neural_renderer_img and not self.neural_renderer_seg:
            pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)

        if self.neural_renderer_seg:
            pixels = pixels.permute(0,3,1,2).contiguous()
            labels, images = pixels[:, :64], pixels[:, 64:]
            images = self.neural_renderer_img(images)
            labels = self.neural_renderer_seg(labels)
            pixels = torch.cat([labels, images], dim=1)
            pixels = pixels * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)
        
        if self.neural_renderer_img:
            pixels = pixels.permute(0,3,1,2).contiguous()
            pixels = self.neural_renderer_img(pixels)
            pixels = pixels * 2 - 1
            return pixels, torch.cat([pitch, yaw], -1)