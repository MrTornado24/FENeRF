import plyfile
import argparse
import torch
import numpy as np
import skimage.measure
import scipy
import mrcfile
import os

N_CHANNELS = 22 


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
                   
    return samples.unsqueeze(0), voxel_origin, voxel_size


def sample_generator(generator, z, max_batch=24000, voxel_resolution=256, voxel_origin=[0,0,0], cube_length=2.0, psi=0.5):
    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
    
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    transformed_ray_directions_expanded[..., -1] = -1
    
    
    # generator.generate_avg_frequencies()
    avg_frequencies_geo, avg_phase_shifts_geo, avg_frequencies_app, avg_phase_shifts_app = generator.generate_avg_frequencies()
    with torch.no_grad():
        raw_frequencies_geo, raw_phase_shifts_geo = generator.siren.geo_mapping_network(z)
        raw_frequencies_app, raw_phase_shifts_app = generator.siren.app_mapping_network(z)
        truncated_frequencies_geo = avg_frequencies_geo + psi * (raw_frequencies_geo - avg_frequencies_geo)
        truncated_phase_shifts_geo = avg_phase_shifts_geo + psi * (raw_phase_shifts_geo - avg_phase_shifts_geo)
        truncated_frequencies_app = avg_frequencies_app + psi * (raw_frequencies_app - avg_frequencies_app)
        truncated_phase_shifts_app = avg_phase_shifts_app + psi * (raw_phase_shifts_app - avg_phase_shifts_app)
    with torch.no_grad():
        while head < samples.shape[1]:
            coarse_output = generator.siren.forward_with_frequencies_phase_shifts(samples[:, head:head+max_batch], truncated_frequencies_geo, truncated_frequencies_app, truncated_phase_shifts_geo, truncated_phase_shifts_app, ray_directions=transformed_ray_directions_expanded[:, :samples.shape[1]-head]).reshape(samples.shape[0], -1, N_CHANNELS)            
            sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
            head += max_batch
    
    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    
    return sigmas


def sample_generator_wth_frequencies_phase_shifts(generator, meta, max_batch=100000, voxel_resolution=256, voxel_origin=[0,0,0], cube_length=2.0, psi=0.5):
    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = samples.to(generator.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=generator.device)
    
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=generator.device)
    transformed_ray_directions_expanded[..., -1] = -1
    truncated_frequencies_geo, truncated_frequencies_app, truncated_phase_shifts_geo, truncated_phase_shifts_app = meta['truncated_frequencies_geo'], meta['truncated_frequencies_app'], meta['truncated_phase_shifts_geo'], meta['truncated_phase_shifts_app']
    # generator.generate_avg_frequencies()    
    with torch.no_grad():
        while head < samples.shape[1]:
            coarse_output = generator.siren.forward_with_frequencies_phase_shifts(samples[:, head:head+max_batch], truncated_frequencies_geo, truncated_frequencies_app, truncated_phase_shifts_geo, truncated_phase_shifts_app, ray_directions=transformed_ray_directions_expanded[:, :samples.shape[1]-head]).reshape(samples.shape[0], -1, N_CHANNELS)            
            sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
            head += max_batch
    
    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    
    return sigmas



    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[3,4,5])
    parser.add_argument('--cube_size', type=float, default=0.3)
    parser.add_argument('--voxel_resolution', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='shapes')
    parser.add_argument('--latent_path', type=str, default=None)
    opt = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = torch.load(opt.path, map_location=torch.device(device))
    ema = torch.load(opt.path.split('generator')[0] + 'ema.pth')
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    
    if opt.latent_path is None:
        for seed in opt.seeds:
            torch.manual_seed(seed)
            
            z = torch.randn(1, 256, device=device)

            voxel_grid = sample_generator(generator, z, cube_length=opt.cube_size, voxel_resolution=opt.voxel_resolution)
            os.makedirs(opt.output_dir, exist_ok=True)
            with mrcfile.new_mmap(os.path.join(opt.output_dir, f'{seed}.mrc'), overwrite=True, shape=voxel_grid.shape, mrc_mode=2) as mrc:
                mrc.data[:] = voxel_grid

    else:
        meta = torch.load(opt.latent_path)
        # ### debug: transform for inversed latent codes 
        w_geo_frequency_offsets, w_geo_phase_shift_offsets, w_app_frequency_offsets, w_app_phase_shift_offsets = meta['w_geo_frequency_offsets'].to(device), meta['w_geo_phase_shift_offsets'].to(device), meta['w_app_frequency_offsets'].to(device), meta['w_app_phase_shift_offsets'].to(device)
        w_geo_frequencies, w_geo_phase_shifts, w_app_frequencies, w_app_phase_shifts = meta['w_geo_frequencies'].to(device), meta['w_geo_phase_shifts'].to(device), meta['w_app_frequencies'].to(device), meta['w_app_phase_shifts'].to(device)
        meta['truncated_frequencies_geo'] = w_geo_frequencies + w_geo_frequency_offsets
        meta['truncated_frequencies_app'] =  w_app_frequencies + w_app_frequency_offsets
        meta['truncated_phase_shifts_geo'] = w_geo_phase_shifts + w_geo_phase_shift_offsets
        meta['truncated_phase_shifts_app'] = w_app_phase_shifts + w_app_phase_shift_offsets
        
        voxel_grid = sample_generator_wth_frequencies_phase_shifts(generator, meta, cube_length=opt.cube_size, voxel_resolution=opt.voxel_resolution)
        os.makedirs(opt.output_dir, exist_ok=True)
        with mrcfile.new_mmap(os.path.join(opt.output_dir, f'{opt.seeds[0]}.mrc'), overwrite=True, shape=voxel_grid.shape, mrc_mode=2) as mrc:
            mrc.data[:] = voxel_grid







