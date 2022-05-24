import argparse
import math
import glob
import numpy as np
import sys
import os
from train_double_latent_semantic import mask2color
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import curriculums


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()
    
def generate_img(gen, z_geo, z_app, **kwargs):
    
    with torch.no_grad():
        img, depth_map = generator.staged_forward(z_geo, z_app, **kwargs)
        img, segmap = img[:, -3:], img[:, :-3]
    return img, mask2color(segmap) / 255.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    # curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['last_back'] = False
    curriculum['nerf_noise'] = 0
    # curriculum['fill_mode'] = 'weight'
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    
    os.makedirs(opt.output_dir, exist_ok=True)

    generator = torch.load(opt.path, map_location=torch.device(device))
    generator.softmax_label = False
    generator.neural_renderer_img = None
    generator.neural_renderer_seg = None
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters()) # TODO: what is ema? why use ema parameter?
    generator.set_device(device)
    generator.eval()
    
    face_angles = [-0.5, -0.25, 0., 0.25, 0.5]
    # face_angles = [-0.5, 0., 0.5]

    face_angles = [a + curriculum['h_mean'] for a in face_angles]

    for seed in tqdm(opt.seeds):
        images = []
        segmaps = []
        for i, yaw in enumerate(face_angles):
            curriculum['h_mean'] = yaw
            torch.manual_seed(seed)
            z_geo = torch.randn((1, 256), device=device)
            z_app = torch.randn((1, 256), device=device)
            img, segmap = generate_img(generator, z_geo, z_app, **curriculum)
            images.append(img)
            segmaps.append(segmap)
        save_image(torch.cat(images), os.path.join(opt.output_dir, f'grid_{seed}_RGB.png'), normalize=True, range=(-1,1))
        save_image(torch.cat(segmaps), os.path.join(opt.output_dir, f'grid_{seed}_SEG.png'), noralize=True, range=(0,1))

