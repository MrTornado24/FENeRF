"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""

import os
import shutil
import torch
import copy
import argparse
from torch.utils.data import dataset

from torchvision.utils import save_image
from pytorch_fid import fid_score
# from kid_score import calculate_kid_given_paths
from tqdm import tqdm

import datasets
import curriculums


COLOR_MAP = {
            0: [0, 0, 0], 
            1: [204, 0, 0],
            2: [76, 153, 0], 
            3: [204, 204, 0], 
            4: [51, 51, 255], 
            5: [204, 0, 204], 
            6: [0, 255, 255], 
            7: [255, 204, 204], 
            8: [102, 51, 0], 
            9: [255, 0, 0], 
            10: [102, 204, 0], 
            11: [255, 255, 0], 
            12: [0, 0, 153], 
            13: [0, 0, 204], 
            14: [255, 51, 153], 
            15: [0, 204, 204], 
            16: [0, 51, 0], 
            17: [255, 153, 51], 
            18: [0, 204, 0]}


def mask2color(masks):
      masks = torch.argmax(masks, dim=1).float()
      sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float)
      for key in COLOR_MAP:
            sample_mask[masks==key] = torch.tensor(COLOR_MAP[key], dtype=torch.float)
      sample_mask = sample_mask.permute(0,3,1,2)
      return sample_mask


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1

def output_real_labels_semantic(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        _, real_labels, _ = next(dataloader)

        for img in real_labels:
            img = mask2color(img[None])[0]
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1


def setup_evaluation(dataset_name, generated_dir, dataset_path, target_size=128, num_imgs=8000, **kwargs):

    # Only make real images if they haven't been made yet
    if kwargs.get('background_mask', False):
        real_dir = os.path.join('/apdcephfs/share_1330077/starksun/projects/pi-GAN/data', 'EvalImages', dataset_name + '_black_back' + '_real_images_' + str(target_size))
    else:
        real_dir = os.path.join('/apdcephfs/share_1330077/starksun/projects/pi-GAN/data', 'EvalImages', dataset_name + '_real_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, dataset_path=dataset_path, img_size=target_size,  background_mask=kwargs.get('background_mask', False), return_label=False)

        print('outputting real images...')
        output_real_images(dataloader, num_imgs, real_dir)
        print('...done')

    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir


def output_images(generator, input_metadata, rank, world_size, output_dir, num_imgs=2048):
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = 128
    metadata['batch_size'] = 4

    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 1

    img_counter = rank
    generator.eval()
    img_counter = rank

    if rank == 0: pbar = tqdm("generating images", total = num_imgs)
    with torch.no_grad():
        while img_counter < num_imgs:
            z = torch.randn((metadata['batch_size'], generator.module.z_dim), device=generator.module.device)
            generated_imgs, _, _ = generator.module.staged_forward(z, **metadata)
            for img in generated_imgs:
                if img.shape[0] != 3:
                    img = img[-3:]
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
                img_counter += world_size
                if rank == 0: pbar.update(world_size)
    if rank == 0: pbar.close()


def output_images_double(generator, input_metadata, rank, world_size, output_dir, num_imgs=2048):
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = 128
    metadata['batch_size'] = 4

    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 1

    img_counter = rank
    generator.eval()
    img_counter = rank

    if rank == 0: pbar = tqdm("generating images", total = num_imgs)
    with torch.no_grad():
        while img_counter < num_imgs:
            z_geo = torch.randn((metadata['batch_size'], generator.module.z_geo_dim), device=generator.module.device)
            z_app = torch.randn((metadata['batch_size'], generator.module.z_app_dim), device=generator.module.device)
            generated_imgs, _ = generator.module.staged_forward(z_geo, z_app, **metadata)
            for img in generated_imgs:
                if img.shape[0] != 3:
                    img = img[-3:]
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
                img_counter += world_size
                if rank == 0: pbar.update(world_size)
    if rank == 0: pbar.close()


def calculate_fid(dataset_name, generated_dir, background_mask=False, target_size=256, **kwargs):
    if background_mask:
        real_dir = os.path.join('/apdcephfs/share_1330077/starksun/projects/pi-GAN/data', 'EvalImages', dataset_name + '_black_back' + '_real_images_' + str(target_size))
    else:
        real_dir = os.path.join('/apdcephfs/share_1330077/starksun/projects/pi-GAN/data', 'EvalImages', dataset_name + '_real_images_' + str(target_size))
    fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 128, 'cuda', 2048)
    torch.cuda.empty_cache()
    return fid

# def calculate_kid(dataset_name, generated_dir, background_mask=False, target_size=256, **kwargs):
#     if background_mask:
#         real_dir = os.path.join('/apdcephfs/share_1330077/starksun/projects/pi-GAN/data', 'EvalImages', dataset_name + '_black_back' + '_real_images_' + str(target_size))
#     else:
#         real_dir = os.path.join('/apdcephfs/share_1330077/starksun/projects/pi-GAN/data', 'EvalImages', dataset_name + '_real_images_' + str(target_size))
#     kid = calculate_kid_given_paths([real_dir, generated_dir], 128, 'cuda', 2048)
#     torch.cuda.empty_cache()

#     return kid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_imgs', type=int, default=8000)

    opt = parser.parse_args()

    real_images_dir = setup_evaluation(opt.dataset, None, target_size=opt.img_size, num_imgs=opt.num_imgs)