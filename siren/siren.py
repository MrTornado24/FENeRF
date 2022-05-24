import sys
from numpy.lib.type_check import imag
from torch._C import device

from torch.functional import align_tensors
sys.path.append('/apdcephfs/share_1330077/starksun/projects/pi-GAN')
from fid_evaluation import output_images
import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from .latent_grid import StyleGenerator2D
from .layers import *

class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(30. * x)

def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

# class CustomMappingNetwork(nn.Module):
#     def __init__(self, z_dim, map_hidden_dim, map_output_dim):
#         super().__init__()
#         self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
#                                      nn.LeakyReLU(0.2, inplace=True),

#                                     nn.Linear(map_hidden_dim, map_hidden_dim),
#                                     nn.LeakyReLU(0.2, inplace=True),

#                                     nn.Linear(map_hidden_dim, map_hidden_dim),
#                                     nn.LeakyReLU(0.2, inplace=True),

#                                     nn.Linear(map_hidden_dim, map_output_dim))

#         self.network.apply(kaiming_leaky_init)
#         with torch.no_grad():
#             self.network[-1].weight *= 0.25

#     def forward(self, z):
#         frequencies_offsets = self.network(z)
#         frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
#         phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

#         return frequencies, phase_shifts

class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, n_blocks=3):
        super().__init__()
        self.network = [nn.Linear(z_dim, map_hidden_dim),
                        nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_blocks):
            self.network.append(nn.Linear(map_hidden_dim, map_hidden_dim))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.network.append(nn.Linear(map_hidden_dim, map_output_dim))
        self.network = nn.Sequential(*self.network)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z) # z: (n_batch * n_point, n_channel)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        if x.shape[1] != freq.shape[1]:
            freq = freq.unsqueeze(1).expand_as(x) #TODO: all x conditioned on a single freq and phase_shift --> every x conditioned on a specific freq and phase_shift
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)


class TALLSIREN(nn.Module):
    """Primary SIREN  architecture used in pi-GAN generators."""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30

        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = self.color_layer_linear(rbg)

        return torch.cat([rbg, sigma], dim=-1)


class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor

class SPATIALSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        
        return torch.cat([rbg, sigma], dim=-1)
    

class SPATIALSIRENBASELINEHD(nn.Module):
    """Same architecture as SPATIALSIRENBASELINE but use neural renderer"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 64))
        
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        # rbg = torch.sigmoid(self.color_layer_linear(rbg))
        rbg = self.color_layer_linear(rbg)
        return torch.cat([rbg, sigma], dim=-1)

    
class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


def sample_from_3dgrid(coordinates, grid):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    coordinates = coordinates.float()
    grid = grid.float()
    
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=True)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features


def modified_first_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class EmbeddingPiGAN128(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""
    
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(32 + 3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        print(self.network)

        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        x = torch.cat([shared_features, input], -1)

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)

    
    
class EmbeddingPiGAN256(EmbeddingPiGAN128):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 64, 64, 64)*0.1)


class SPATIALSIRENGRID(nn.Module):
      """Same architecture as SPATIALSIRENBASELINE but use local latent sampled from grid"""

      def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
            super().__init__()
            self.device = device
            self.input_dim = input_dim
            self.z_dim = z_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.local_coordinates = True
            
            self.network = nn.ModuleList([
                  FiLMLayer(3, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
            ])
            self.final_layer = nn.Linear(hidden_dim, 1)
            
            self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
            self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
            
            self.mapping_network = CustomMappingNetwork(32, 256, (len(self.network) + 1)*hidden_dim*2, n_blocks=1)

            self.grid_latent_network = StyleGenerator2D(out_res=32, out_ch=32, z_dim=z_dim, ch_mul=1, ch_max=256, skip_conn=False)
            
            self.network.apply(frequency_init(25))
            self.final_layer.apply(frequency_init(25))
            self.color_layer_sine.apply(frequency_init(25))
            self.color_layer_linear.apply(frequency_init(25))
            self.network[0].apply(first_layer_film_sine_init)
            self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

      def forward(self, input, z, ray_directions, **kwargs):
            latent_grid = self.grid_latent_network(z)
            input_grid = self.gridwarper(input) # range: (-1.4, 1.4)
            sampled_latent = self.sample_local_latents(latent_grid, input_grid)
            frequencies, phase_shifts = self.mapping_network(sampled_latent)
            if self.local_coordinates:
                # map global coordinate space into local coordinate space (i.e. each grid cell has a [-1, 1] range)
                preserve_y = sampled_latent.ndim == 4  # if latents are 2D, then keep the y coordinate global
                input = self.get_local_coordinates(
                    global_coords=input, local_grid_length=32, preserve_y=False
                )
            return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, box_warp=False, **kwargs)
      
      def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
            frequencies = frequencies*15 + 30
            x = self.gridwarper(input)
                  
            for index, layer in enumerate(self.network):
                  start = index * self.hidden_dim
                  end = (index+1) * self.hidden_dim
                  x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
            
            sigma = self.final_layer(x)
            rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
            rbg = torch.sigmoid(self.color_layer_linear(rbg))
            
            return torch.cat([rbg, sigma], dim=-1)

      def sample_local_latents(self, local_latents, xyz):
        B, local_z_dim, H, W = local_latents.shape
        # take only x and z coordinates, since our latent codes are in a 2D grid (no y dimension)
        # for the purposes of grid_sample we treat H*W as the H dimension and samples_per_ray as the W dimension
        xyz = xyz[:, :, [0, 2]].unsqueeze(1)  # [B, H * W, samples_per_ray, 2]
        # all samples get the most detailed latent codes
        sampled_local_latents = nn.functional.grid_sample(
            input=local_latents, # (b, c, h, w)
            grid=xyz, # (b, 1, n_pixel, 2)
            mode='bilinear',  # bilinear mode will use trilinear interpolation if input is 5D
            align_corners=False,
            padding_mode="zeros",
        )
        # output is shape [B, local_z_dim, H * W, samples_per_ray]
        # put channel dimension at end: [B, H * W, samples_per_ray, local_z_dim]
        sampled_local_latents = sampled_local_latents.permute(0, 2, 3, 1)

        # merge everything else into batch dim: [B * H * W * samples_per_ray, local_z_dim]
        sampled_local_latents = sampled_local_latents.reshape(B, -1, local_z_dim)

        return sampled_local_latents

      def get_local_coordinates(self, global_coords, local_grid_length, preserve_y=True):
        local_coords = global_coords.clone()
        # it is assumed that the global coordinates are scaled to [-1, 1]
        # convert to [0, 1] scale
        local_coords = (local_coords + 1) / 2
        # scale so that each grid cell in the local_latent grid is 1x1 in size
        local_coords = local_coords * local_grid_length
        # subtract integer from each coordinate so that they are all in range [0, 1]
        local_coords = local_coords - (local_coords - 0.5).round()
        # return to [-1, 1] scale
        local_coords = (local_coords * 2) - 1

        if preserve_y:
            # preserve the y dimension in the global coordinate frame, since it doesn't have a local latent code
            coords = torch.cat([local_coords[..., 0:1], global_coords[..., 1:2], local_coords[..., 2:3]], dim=-1)
        else:
            coords = torch.cat([local_coords[..., 0:1], local_coords[..., 1:2], local_coords[..., 2:3]], dim=-1)
        return coords


class SPATIALSIRENVOLUME(nn.Module):
      """Same architecture as SPATIALSIRENBASELINE but use local latent sampled from volume"""

      def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
            super().__init__()
            self.device = device
            self.input_dim = input_dim
            self.z_dim = z_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            
            self.network = nn.ModuleList([
                  FiLMLayer(3, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
                  FiLMLayer(hidden_dim, hidden_dim),
            ])
            self.final_layer = nn.Linear(hidden_dim, 1)
            
            self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
            self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
            
            self.mapping_network = CustomMappingNetwork(32, 256, (len(self.network) + 1)*hidden_dim*2)
            # self.volume_latent_network = VolumeStyleGenerator(
            #       mapping_fmaps=z_dim,
            #       style_mixing_prob=0.9,       # Probability of mixing styles during training. None = disable.
            #       truncation_psi=0.7,          # Style strength multiplier for the truncation trick. None = disable.
            #       truncation_cutoff=8,          # Number of layers for which to apply the truncation trick. None = disable.
            #       resolution=32,
            #       fmap_base=512,
            #       fmap_max=256)

            self.volume_latent_network = VolumeStyleGenerator(input_nc=z_dim, output_nc=32, n_samples=3, norm='batch', activation='ReLU', padding_type='zero')
            
            self.network.apply(frequency_init(25))
            self.final_layer.apply(frequency_init(25))
            self.color_layer_sine.apply(frequency_init(25))
            self.color_layer_linear.apply(frequency_init(25))
            self.network[0].apply(first_layer_film_sine_init)
            
            self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

      def forward(self, input, z, ray_directions, **kwargs):
            latent_grid = self.volume_latent_network(z)
            input_grid = self.gridwarper(input)
            # interpolate latent
            # samples = F.grid_sample(latent_grid,
            #                     input[..., [0, 2]].unsqueeze(2),
            #                     align_corners=True,
            #                     mode='bilinear',
            #                     padding_mode='zeros')
            samples = sample_from_3dgrid(input_grid, latent_grid)
            frequencies, phase_shifts = self.mapping_network(samples)

            return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, box_warp=False, **kwargs)
      
      def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
            frequencies = frequencies*15 + 30
            input = self.gridwarper(input)
            x = input
            for index, layer in enumerate(self.network):
                  start = index * self.hidden_dim
                  end = (index+1) * self.hidden_dim
                  x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
            
            sigma = self.final_layer(x)
            rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
            rbg = torch.sigmoid(self.color_layer_linear(rbg))
            
            return torch.cat([rbg, sigma], dim=-1)
      

class SPATIALSIRENSEMANTIC(nn.Module):
    """Same architecture as TALLSIREN but synthesis semantic map"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_batch_size = 2500
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.label_layer_sine = FiLMLayer(hidden_dim, hidden_dim)
        self.label_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 19)) # 19 semantic labels 
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 2)*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.label_layer_sine.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.activation = nn.Softmax(dim=-1)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        n_batch, n_pixel = input.shape[:2]
        # output = torch.zeros((n_batch, n_pixel, self.output_dim)).to(input)
        # for b in range(n_batch):
        #     head = 0
        #     while head < n_pixel:
        #         tail = head + self.max_batch_size
        #         output[b:b+1, head:tail] = self.forward_with_frequencies_phase_shifts(input[b:b+1, head:tail], frequencies[b:b+1], phase_shifts[b:b+1], ray_directions[b:b+1, head:tail], **kwargs)
        #         head += self.max_batch_size
        # return output
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        start += self.hidden_dim
        end += self.hidden_dim
        sigma = self.final_layer(x)
        labels = self.label_layer_sine(x, frequencies[..., start:end], phase_shifts[..., start:end])
        # TODO: w. / w.o softmax activation on label
        labels = self.label_layer_linear(labels)
        start += self.hidden_dim
        end += self.hidden_dim
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., start:end], phase_shifts[..., start:end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([labels, rbg, sigma], dim=-1)


class SPATIALSIRENBASELINESEMANTIC(nn.Module):
    """Same architecture as SPATIALSIRENSEMANTIC but doesn't condition on geometry code when regressing labels"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_batch_size = 2500
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)        
        self.label_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 19)) # 19 semantic labels 
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.activation = nn.Softmax(dim=-1)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        n_batch, n_pixel = input.shape[:2]
        # output = torch.zeros((n_batch, n_pixel, self.output_dim)).to(input)
        # for b in range(n_batch):
        #     head = 0
        #     while head < n_pixel:
        #         tail = head + self.max_batch_size
        #         output[b:b+1, head:tail] = self.forward_with_frequencies_phase_shifts(input[b:b+1, head:tail], frequencies[b:b+1], phase_shifts[b:b+1], ray_directions[b:b+1, head:tail], **kwargs)
        #         head += self.max_batch_size
        # return output
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        sigma = self.final_layer(x)
        # labels = torch.sigmoid(self.label_layer_linear(x))
        labels = self.label_layer_linear(x)
        start += self.hidden_dim
        end += self.hidden_dim
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., start:end], phase_shifts[..., start:end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([labels, rbg, sigma], dim=-1)


class SPATIALSIRENDISENTANGLE(nn.Module):
    """Same architecture as TALLSIREN but use double latent codes"""

    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_batch_size = 2500
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.activation = nn.Softmax(dim=-1)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30 # TODO: 为什么做变换

        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        rbg = torch.cat([ray_directions, x], dim=-1)
        sigma = self.final_layer(x)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])

        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)

class SPATIALSIRENDISENTANGLE_debug(nn.Module):
    """Same architecture as TALLSIREN but use double latent codes"""

    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_batch_size = 2500
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_pre = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        # self.color_layer_sine = FiLMLayer(hidden_dim + 32, hidden_dim) # ray_drection dim: 3 --> 32
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.dir_mapping_network = nn.Sequential(
            nn.Linear(3, 256),
            nn.Linear(256, 32)
        )
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.activation = nn.Softmax(dim=-1)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)

        # n_batch, n_pixel = input.shape[:2]
        # output = torch.zeros((n_batch, n_pixel, self.output_dim)).to(input)
        # for b in range(n_batch):
        #     head = 0
        #     while head < n_pixel:
        #         tail = head + self.max_batch_size
        #         output[b:b+1, head:tail] = self.forward_with_frequencies_phase_shifts(input[b:b+1, head:tail], frequencies[b:b+1], phase_shifts[b:b+1], ray_directions[b:b+1, head:tail], **kwargs)
        #         head += self.max_batch_size
        # return output
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30 # TODO: 为什么做变换

        
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        sigma = self.final_layer(x)
        # ray_directions = self.dir_mapping_network(ray_directions)
        x = self.color_layer_pre(x)
        rbg = torch.cat([ray_directions, x], dim=-1)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])

        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)


class SPATIALSIRENAUGDISENTANGLE(nn.Module):
    """Same architecture as SPATIALSIRENDISENTANGLE but has augmented color branch and narrower density feature branch"""

    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_batch_size = 2500
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_pre = nn.Sequential(
            nn.Linear(hidden_dim, 3),
        )
        # self.color_layer_sine = FiLMLayer(hidden_dim + 32, hidden_dim) # ray_drection dim: 3 --> 32
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(3 + 3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)

        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30 # TODO: 为什么做变换
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        sigma = self.final_layer(x)
        x = self.color_layer_pre(x)
        rbg = torch.cat([ray_directions, x], dim=-1)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])

        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)


class RESSIRENDISENTANGLE(nn.Module):
    """
    Same architecture as SIRENDISENTANGLE but use residual architecure
    code accroding to http://gvv.mpi-inf.mpg.de/projects/i3DMM/
    """

    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_batch_size = 2500
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.res_coord_layer = nn.Linear(hidden_dim, 3)
        self.density_layer_linear = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.color_layer_pre = nn.Sequential(nn.Linear(3, hidden_dim))
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        # self.dir_mapping_network = nn.Sequential(
        #     nn.Linear(3, 256),
        #     nn.Linear(256, 32)
        # )
        
        self.network.apply(frequency_init(25))
        self.density_layer_linear.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.activation = nn.Softmax(dim=-1)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)

        # n_batch, n_pixel = input.shape[:2]
        # output = torch.zeros((n_batch, n_pixel, self.output_dim)).to(input)
        # for b in range(n_batch):
        #     head = 0
        #     while head < n_pixel:
        #         tail = head + self.max_batch_size
        #         output[b:b+1, head:tail] = self.forward_with_frequencies_phase_shifts(input[b:b+1, head:tail], frequencies[b:b+1], phase_shifts[b:b+1], ray_directions[b:b+1, head:tail], **kwargs)
        #         head += self.max_batch_size
        # return output
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30 # TODO: 为什么做变换

        
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        coords_res = self.res_coord_layer(x)
        input = input + coords_res
        sigma = self.density_layer_linear(input)
        # ray_directions = self.dir_mapping_network(ray_directions)
        rbg = self.color_layer_pre(input)
        rbg = torch.cat([ray_directions, rbg], dim=-1)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])

        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)


class SPATIALSIRENSEMANTICDISENTANGLE(nn.Module):
    """Same architecture as TALLSIREN but use double latent codes and render semantic maps"""

    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.label_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim - 4)) # output_dim = seg_channel + rgb_channel + density_channel
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.color_layer_sine[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30 # TODO: 为什么做变换
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        sigma = self.final_layer(x)
        start += self.hidden_dim
        end += self.hidden_dim
        labels = self.label_layer_linear(x)
        # rbg = torch.cat([ray_directions, input, labels], dim=-1)
        rbg = torch.cat([ray_directions, x], dim=-1)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        return torch.cat([labels, rbg, sigma], dim=-1)


class SIRENBASELINESEMANTICDISENTANGLE(nn.Module):
    """Same architecture as TALLSIREN baseline but use double latent codes and render semantic maps"""

    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.label_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim - 4))
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        rbg = torch.cat([ray_directions, x], dim=-1)
        sigma = self.final_layer(x)
        labels = self.label_layer_linear(x)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        return torch.cat([labels, rbg, sigma], dim=-1)


class SIRENBASELINESEMANTICDISENTANGLE_debug(nn.Module):
    """Same architecture as SIRENBASELINESEMANTICDISENTANGLE_debug except adding sigmoid to label"""

    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.label_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 19)) # 19 semantic labels 
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        rbg = torch.cat([ray_directions, x], dim=-1)
        sigma = self.final_layer(x)
        labels = torch.sigmoid(self.label_layer_linear(x))
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        return torch.cat([labels, rbg, sigma], dim=-1)


class SPATIALSIRENSEMANTICHD(nn.Module):
    """Same architecture as SPATIALSIRENSEMANTIC but on a high resolution"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_batch_size = 2500
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.label_layer_sine = FiLMLayer(hidden_dim, hidden_dim)
        self.label_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 64)) # 19 semantic labels 
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 64))
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 2)*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.label_layer_sine.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        self.activation = nn.Softmax(dim=-1)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        n_batch, n_pixel = input.shape[:2]
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        start += self.hidden_dim
        end += self.hidden_dim
        sigma = self.final_layer(x)
        labels = self.label_layer_sine(x, frequencies[..., start:end], phase_shifts[..., start:end])
        # TODO: w. / w.o softmax activation on label
        labels = self.label_layer_linear(labels)
        start += self.hidden_dim
        end += self.hidden_dim
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., start:end], phase_shifts[..., start:end])
        rbg = self.color_layer_linear(rbg)
        # rbg = torch.sigmoid(self.color_layer_linear(rbg))
        return torch.cat([labels, rbg, sigma], dim=-1)


class EmbeddingPiGAN128SEMANTICDISENTANGLE(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""
    
    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(32 + 3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])

        self.final_layer = nn.Linear(hidden_dim, 1)

        # self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.label_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim-4)
        )
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        x = torch.cat([shared_features, input], -1)

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        rbg = torch.cat([ray_directions, x], dim=-1)
        sigma = self.final_layer(x)
        labels = self.label_layer_linear(x)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([labels, rbg, sigma], dim=-1)


class TextureEmbeddingPiGAN128SEMANTICDISENTANGLE(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details.
        Embeddings are in color prediction branch instead of density network"""
    
    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])

        self.final_layer = nn.Linear(hidden_dim, 1)

        # self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+32+3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.label_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim-4)
        )
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        # x = torch.cat([shared_features, input], -1)
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        rbg = torch.cat([ray_directions, shared_features, x], dim=-1)
        sigma = self.final_layer(x)
        labels = self.label_layer_linear(x)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([labels, rbg, sigma], dim=-1)


class TextureEmbeddingPiGAN256SEMANTICDISENTANGLE(TextureEmbeddingPiGAN128SEMANTICDISENTANGLE):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details.
        Embeddings are in color prediction branch instead of density network"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1,32,64,64,64)*0.1)


class TextureEmbeddingPiGAN256SEMANTICDISENTANGLE_DIM_96(TextureEmbeddingPiGAN128SEMANTICDISENTANGLE):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details.
        Embeddings are in color prediction branch instead of density network"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1,32,96,96,96)*0.1)


class TextureEmbeddingPiGAN128SEMANTICDISENTANGLE_WO_DIR(nn.Module):
    """
    1. Smaller architecture that has an additional cube of embeddings. Often gives better fine details.
        Embeddings are in color prediction branch instead of density network;
    2. remove view direction
    3. add more color layers 
        """
    
    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])

        self.final_layer = nn.Linear(hidden_dim, 1)

        # self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+32, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),


        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.label_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim-4)
        )
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)
        self.color_layer_sine[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        # x = torch.cat([shared_features, input], -1)
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        rbg = torch.cat([shared_features, x], dim=-1)
        sigma = self.final_layer(x)
        labels = self.label_layer_linear(x)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([labels, rbg, sigma], dim=-1)


class TextureEmbeddingPiGAN128SEMANTICDISENTANGLE_WO_DIR_debug(nn.Module):
    """
    1. Smaller architecture that has an additional cube of embeddings. Often gives better fine details.
        Embeddings are in color prediction branch instead of density network;
    2. remove view direction
    3. add more color layers 
        """
    
    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])

        self.final_layer = nn.Linear(hidden_dim, 1)

        # self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+32, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.label_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim-4)
        )
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)
        self.color_layer_sine[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        # x = torch.cat([shared_features, input], -1)
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        rbg = torch.cat([shared_features, x], dim=-1)
        sigma = self.final_layer(x)
        labels = self.label_layer_linear(x)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([labels, rbg, sigma], dim=-1)


class TextureEmbeddingPiGAN128SEMANTICDISENTANGLE_WO_DIR_debug2(nn.Module):
    """
    1. Smaller architecture that has an additional cube of embeddings. Often gives better fine details.
        Embeddings are in color prediction branch instead of density network;
    2. remove view direction
    3. add more color layers 
        """
    
    def __init__(self, input_dim=2, z_geo_dim=100, z_app_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_geo_dim = z_geo_dim
        self.z_app_dim = z_app_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])

        self.final_layer = nn.Linear(hidden_dim, 1)

        # self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_sine = nn.ModuleList([
            FiLMLayer(hidden_dim+32, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.geo_mapping_network = CustomMappingNetwork(z_geo_dim, 256, len(self.network)*hidden_dim*2)
        self.app_mapping_network = CustomMappingNetwork(z_app_dim, 256, len(self.color_layer_sine)*hidden_dim*2)
        self.label_layer_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim-4)
        )
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.label_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)
        # self.color_layer_sine[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z_geo, z_app, ray_directions, **kwargs):
        frequencies_geo, phase_shifts_geo = self.geo_mapping_network(z_geo)
        frequencies_app, phase_shifts_app = self.app_mapping_network(z_app)
        return self.forward_with_frequencies_phase_shifts(input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app, ray_directions, **kwargs):
        frequencies_geo = frequencies_geo*15 + 30
        frequencies_app = frequencies_app*15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        # x = torch.cat([shared_features, input], -1)
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_geo[..., start:end], phase_shifts_geo[..., start:end])
        
        rbg = torch.cat([shared_features, x], dim=-1)
        sigma = self.final_layer(x)
        labels = self.label_layer_linear(x)
        for index, layer in enumerate(self.color_layer_sine):
            start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
            rbg = layer(rbg, frequencies_app[..., start:end], phase_shifts_app[..., start: end])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([labels, rbg, sigma], dim=-1)


class TextureEmbeddingPiGAN256SEMANTICDISENTANGLE_WO_DIR_DIM_96(TextureEmbeddingPiGAN128SEMANTICDISENTANGLE_WO_DIR):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details.
        Embeddings are in color prediction branch instead of density network"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1,32,96,96,96)*0.1)

        
def main():
    # model = SPATIALSIRENVOLUME(input_dim=3, z_dim=256, hidden_dim=256, output_dim=4, device=None)
    model = SPATIALSIRENSEMANTIC(input_dim=3, z_dim=256, hidden_dim=256, output_dim=4, device=None)
    input, z, ray_directions = torch.randn(2, 4000, 3), torch.rand(2, 256), torch.rand(2, 4000, 3)
    output = model(input, z, ray_directions)
    print(output.shape)

if __name__ == "__main__":
    main()



