import argparse
import math
import os
from re import I
from unicodedata import normalize
from numpy.lib.utils import deprecate

from torchvision.utils import save_image

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import skvideo.io
import imageio
import curriculums
import cv2
from torchvision.utils import save_image, make_grid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--interpolation_type', type=str, default='video_double_latent_interpolation')
parser.add_argument('--latent_type', default='geo') # for double latent
parser.add_argument('--seeds', nargs='+', default=[0])
parser.add_argument('--output_dir', type=str, default='vids')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_batch_size', type=int, default=2400000)
parser.add_argument('--depth_map', action='store_true')
parser.add_argument('--lock_view_dependence', action='store_true')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--ray_step_multiplier', type=int, default=2)
parser.add_argument('--num_frames', type=int, default=36)
parser.add_argument('--curriculum', type=str, default='CelebA')
parser.add_argument('--trajectory', type=str, default='front')
parser.add_argument('--psi', type=float, default=0.5)
parser.add_argument("--fill_color", type=str, default='black')
parser.add_argument("--fov", type=int, default=12)
parser.add_argument("--save_with_video", action='store_true')
parser.add_argument("--save_with_latent", action='store_true')
parser.add_argument("--seed_mode", default='single', type=str, help='if the seeds are speficifed a range or a number')
opt = parser.parse_args()


os.makedirs(opt.output_dir, exist_ok=True)

curriculum = getattr(curriculums, opt.curriculum)
curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
curriculum['img_size'] = opt.image_size
curriculum['psi'] = opt.psi
curriculum['v_stddev'] = 0
curriculum['h_stddev'] = 0
curriculum['lock_view_dependence'] = opt.lock_view_dependence
curriculum['last_back'] = curriculum.get('eval_last_back', False)
curriculum['num_frames'] = opt.num_frames
curriculum['nerf_noise'] = 0
curriculum['fov'] = opt.fov
curriculum['fill_mode'] = curriculum.get('fill_mode', 'weight')
if curriculum['fill_mode'] == 'seg_padding_background':
      curriculum['fill_mode'] = 'eval_seg_padding_background'
curriculum['fill_color'] = opt.fill_color
curriculum = {key: value for key, value in curriculum.items() if type(key) is str}


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


def tensor_to_PIL(img):    
    # if normalize:
    #     img = img.squeeze() * 0.5 + 0.5
    # else:
    #     img = img.squeeze()
    img = make_grid(img, normalize=True)
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


class FrequencyInterpolator:
    def __init__(self, generator, z1, z2, psi=0.5):
        avg_frequencies, avg_phase_shifts = generator.generate_avg_frequencies()
        raw_frequencies1, raw_phase_shifts1 = generator.siren.mapping_network(z1)
        self.truncated_frequencies1 = avg_frequencies + psi * (raw_frequencies1 - avg_frequencies)
        self.truncated_phase_shifts1 = avg_phase_shifts + psi * (raw_phase_shifts1 - avg_phase_shifts)
        raw_frequencies2, raw_phase_shifts2 = generator.siren.mapping_network(z2)
        self.truncated_frequencies2 = avg_frequencies + psi * (raw_frequencies2 - avg_frequencies)
        self.truncated_phase_shifts2 = avg_phase_shifts + psi * (raw_phase_shifts2 - avg_phase_shifts)

    def forward(self, t):
            if opt.latent_type == 'non':
                frequencies = self.truncated_frequencies1
                phase_shifts = self.truncated_phase_shifts1
            else:
                  frequencies = self.truncated_frequencies1 * (1-t) + self.truncated_frequencies2 * t
                  phase_shifts = self.truncated_phase_shifts1 * (1-t) + self.truncated_phase_shifts2 * t
            

            return frequencies, phase_shifts


class DoubleFrequencyInterpolator:
    def __init__(self, generator, z1_geo, z2_geo, z1_app, z2_app, psi=1., latent_type='geo'):
        self.latent_type = latent_type
        avg_frequencies_geo, avg_phase_shifts_geo, avg_frequencies_app, avg_phase_shifts_app = generator.generate_avg_frequencies()

        raw_frequencies1_geo, raw_phase_shifts1_geo = generator.siren.geo_mapping_network(z1_geo)
        raw_frequencies1_app, raw_phase_shifts1_app = generator.siren.app_mapping_network(z1_app)
        self.truncated_frequencies1_geo = avg_frequencies_geo + psi * (raw_frequencies1_geo - avg_frequencies_geo)
        self.truncated_phase_shifts1_geo = avg_phase_shifts_geo + psi * (raw_phase_shifts1_geo - avg_phase_shifts_geo)
        self.truncated_frequencies1_app = avg_frequencies_app + psi * (raw_frequencies1_app - avg_frequencies_app)
        self.truncated_phase_shifts1_app = avg_phase_shifts_app + psi * (raw_phase_shifts1_app - avg_phase_shifts_app)

      #   raw_frequencies2, raw_phase_shifts2 = generator.siren.mapping_network(z2)
      #   self.truncated_frequencies2 = avg_frequencies + psi * (raw_frequencies2 - avg_frequencies)
      #   self.truncated_phase_shifts2 = avg_phase_shifts + psi * (raw_phase_shifts2 - avg_phase_shifts)
        raw_frequencies2_geo, raw_phase_shifts2_geo = generator.siren.geo_mapping_network(z2_geo)
        raw_frequencies2_app, raw_phase_shifts2_app = generator.siren.app_mapping_network(z2_app)
        self.truncated_frequencies2_geo = avg_frequencies_geo + psi * (raw_frequencies2_geo - avg_frequencies_geo)
        self.truncated_phase_shifts2_geo = avg_phase_shifts_geo + psi * (raw_phase_shifts2_geo - avg_phase_shifts_geo)
        self.truncated_frequencies2_app = avg_frequencies_app + psi * (raw_frequencies2_app - avg_frequencies_app)
        self.truncated_phase_shifts2_app = avg_phase_shifts_app + psi * (raw_phase_shifts2_app - avg_phase_shifts_app)


    def forward(self, t):
        ### debug: increase t's range from (0,1) to (-1,1)
        if self.latent_type == 'app':
            t = (t - 0.5) * 2
        if self.latent_type == 'geo':
            frequencies_geo = self.truncated_frequencies1_geo * (1-t) + self.truncated_frequencies2_geo * t
            phase_shifts_geo = self.truncated_phase_shifts1_geo * (1-t) + self.truncated_phase_shifts2_geo * t
            frequencies_app = self.truncated_frequencies1_app
            phase_shifts_app = self.truncated_phase_shifts1_app
        elif self.latent_type == 'app':
            frequencies_app = self.truncated_frequencies1_app * (1-t) + self.truncated_frequencies2_app * t
            phase_shifts_app = self.truncated_phase_shifts1_app * (1-t) + self.truncated_phase_shifts2_app * t
            frequencies_geo = self.truncated_frequencies1_geo
            phase_shifts_geo = self.truncated_phase_shifts1_geo
        elif self.latent_type == 'both':
            frequencies_geo = self.truncated_frequencies1_geo * (1-t) + self.truncated_frequencies2_geo * t
            phase_shifts_geo = self.truncated_phase_shifts1_geo * (1-t) + self.truncated_phase_shifts2_geo * t
            frequencies_app = self.truncated_frequencies1_app * (1-t) + self.truncated_frequencies2_app * t
            phase_shifts_app = self.truncated_phase_shifts1_app * (1-t) + self.truncated_phase_shifts2_app * t
        elif self.latent_type == 'non':
            frequencies_geo = self.truncated_frequencies1_geo 
            phase_shifts_geo = self.truncated_phase_shifts1_geo
            frequencies_app = self.truncated_frequencies1_app 
            phase_shifts_app = self.truncated_phase_shifts1_app

        return frequencies_geo, frequencies_app, phase_shifts_geo, phase_shifts_app


def tensor_to_PIL(img):
    img = make_grid(img, normalize=True)
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


def run_video_latent_interpolation(opt):
      generator = torch.load(opt.path, map_location=torch.device(device))
      generator.output_dim = 4
      generator.channel_dim = 3
      ema_file = opt.path.split('generator')[0] + 'ema.pth'
      ema = torch.load(ema_file)
      ema.copy_to(generator.parameters())
      generator.set_device(device)
      generator.eval()

      if opt.trajectory == 'front':
            trajectory = []
            for t in np.linspace(0, 1, curriculum['num_frames']):
                  pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
                  yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                  fov = curriculum['fov'] + 5 + np.sin(t * 2 * math.pi) * 5
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'orbit':
            trajectory = []
            for t in np.linspace(0, 1, curriculum['num_frames']):
                  pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/4
                  yaw = t * 2 * math.pi
                  fov = curriculum['fov']

                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'rotation_horizontal':
            trajectory = []
            for t in np.linspace(-1, 1, curriculum['num_frames']//2):
                  pitch = math.pi/2
                  yaw = math.pi/2 + t * 0.5
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))
            for t in np.linspace(1, -1, curriculum['num_frames']//2):
                  pitch = math.pi/2
                  yaw = math.pi/2 + t * 0.5
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'rotation_angles':
            trajectory = []
            angles = [-0.5, -0.25, 0., 0.25, 0.5]
            for t, angle in enumerate(angles):
                  pitch = math.pi / 2
                  yaw = math.pi / 2 + angle
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'rotation_pi':
            trajectory = []
            face_angles = [-0.5, -0.25, 0., 0.25, 0.5]
            for t in np.linspace(-1, 1, curriculum['num_frames']):
                  pitch = math.pi/2
                  yaw = math.pi/2 + t * 0.5 * math.pi
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'non_rotation':
            trajectory = []
            for t in np.linspace(0, 1, curriculum['num_frames']):
                  pitch = math.pi/2
                  yaw = math.pi/2
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'sphere':
            trajectory = []
            for t in np.linspace(0, 1, curriculum['num_frames']):
                  pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
                  yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))            

      print(opt.seeds)

      for i, seed in enumerate(opt.seeds):
            frames = []
            depths = []
            images = []
            torch.manual_seed(seed)

            output_dir = os.path.join(opt.output_dir, f'interpolation_{opt.latent_type}_{seed}')
            output_name = f'interp_{opt.latent_type}_{seed}.mp4'
            if not os.path.exists(output_dir):
                  os.makedirs(output_dir)
            if not os.path.exists(os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}")):
                  os.makedirs(os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}"))
            if opt.save_with_video:
                  # writer = skvideo.io.FFmpegWriter(os.path.join(opt.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
                  ### debug: replace skvideo with cv2 video writer
                  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                  nrows, ncols = 1, 1
                  writer = cv2.VideoWriter(os.path.join(output_dir, output_name), fourcc,
                                25, (256 * ncols, 256 * nrows))

            z_current = torch.randn(1, 256, device=device)
            z_next = torch.randn(1, 256, device=device)

            frequencyInterpolator = FrequencyInterpolator(generator, z_current, z_next, psi=opt.psi)
            j = 0
            with torch.no_grad():
                  for t, pitch, yaw, fov in tqdm(trajectory):
                        curriculum['h_mean'] = yaw# + 3.14/2
                        curriculum['v_mean'] = pitch# + 3.14/2
                        curriculum['fov'] = fov
                        curriculum['h_stddev'] = 0
                        curriculum['v_stddev'] = 0
                        img, depth_map = generator.staged_forward_with_frequencies(*frequencyInterpolator.forward(t), max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                        frames.append(tensor_to_PIL(img))
                        save_image(img, os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"img_{j}.png"), nrow=1, normalize=True)
                        images.append(img)
                        j += 1

            save_image(torch.cat(images), os.path.join(output_dir, f"{opt.interpolation_type}_img_{i}.png"), nrow=opt.num_frames, normalize=True)
            if opt.save_with_video:
                  for frame in frames:
                        # writer.writeFrame(np.array(frame))
                        frame = np.array(frame)
                        frame = frame[..., ::-1]
                        writer.write(frame.astype('uint8'))


                  # writer.close()
                  writer.release()


def run_video_double_latent_interpolation(opt):
      generator = torch.load(opt.path, map_location=torch.device(device))
      generator.output_dim = curriculum['output_dim']
      generator.channel_dim = curriculum['output_dim'] - 1
      ema_file = opt.path.split('generator')[0] + 'ema.pth'
      ema = torch.load(ema_file)
      ema.copy_to(generator.parameters())
      generator.set_device(device)
      generator.eval()

      if opt.trajectory == 'front':
            trajectory = []
            for t in np.linspace(0, 1, curriculum['num_frames'], endpoint=True):
                  pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
                  yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                  fov = curriculum['fov'] + 5 + np.sin(t * 2 * math.pi) * 5
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'orbit':
            trajectory = []
            for t in np.linspace(0, 0.5, curriculum['num_frames'], endpoint=True):
                  # pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/4
                  pitch = math.pi/2
                  yaw = t * 2 * math.pi
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'rotation_horizontal':
            trajectory = []
            for t in np.linspace(-1, 1, curriculum['num_frames']):
                  pitch = math.pi/2
                  yaw = math.pi/2 + t * 0.5
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'non_rotation':
            trajectory = []
            for t in np.linspace(0, 1, curriculum['num_frames'], endpoint=True):
                  pitch = math.pi/2
                  yaw = math.pi/2
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'sphere':
            trajectory = []
            for t in np.linspace(0, 1, curriculum['num_frames'], endpoint=True):
                  pitch = 0.2 * np.cos(t * 2 * math.pi) + 1/2 * math.pi
                  yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif opt.trajectory == 'zoom':
            trajectory = []
            for t in np.linspace(0, 1, curriculum['num_frames']):
                  pitch = 1/2 * math.pi
                  yaw = math.pi/2
                  fov = curriculum['fov'] + np.sin(t * 2 * math.pi) * 5
                  trajectory.append((t, pitch, yaw, fov))

      seeds = opt.seeds

      for i, seed in enumerate(seeds):
            output_dir = os.path.join(opt.output_dir, f'interpolation_{opt.latent_type}_{seed}')
            output_name = f'interp_{opt.latent_type}_{seed}.mp4'
            if not os.path.exists(output_dir):
                  os.makedirs(output_dir)
            if not os.path.exists(os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}")):
                        os.makedirs(os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}"))
            if opt.save_with_video:
                  # writer = skvideo.io.FFmpegWriter(os.path.join(output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
                  ### debug: replace skvideo with cv2 video writer
                  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                  nrows, ncols = 1, 4
                  writer = cv2.VideoWriter(os.path.join(output_dir, output_name), fourcc,
                                25, (256 * ncols, 256 * nrows))
            semantic_maps = []
            images = []
            depths = []
            weights_sum = []

            torch.manual_seed(seed)
            z_geo_current = torch.randn(1, 256, device=device)
            z_app_current = torch.randn(1, 256, device=device)
            torch.manual_seed(int(seed)+1)
            z_geo_next = torch.randn(1, 256, device=device)
            z_app_next = torch.randn(1, 256, device=device)

            frequencyInterpolator = DoubleFrequencyInterpolator(generator, z_geo_current, z_geo_next, z_app_current, z_app_next, psi=opt.psi, latent_type=opt.latent_type)
            j = 0
            if not opt.save_with_video:
                  with torch.no_grad():
                        for t, pitch, yaw, fov in tqdm(trajectory):
                              curriculum['h_mean'] = yaw# + 3.14/2
                              curriculum['v_mean'] = pitch# + 3.14/2
                              curriculum['fov'] = fov
                              curriculum['h_stddev'] = 0
                              curriculum['v_stddev'] = 0
                              frame, depth_map, weight_sum = generator.staged_forward_with_frequencies(*frequencyInterpolator.forward(t), max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                              label, img = frame[:, :-3], frame[:, -3:]
                              depths.append(depth_map[None])
                              label = mask2color(label)
                              images.append(img)
                              semantic_maps.append(label)
                              weights_sum.append(weight_sum[:, -3:])
                              save_image(label, os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"label_{j}.png"), nrow=1, normalize=True)
                              save_image(img, os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"img_{j}.png"), nrow=1, normalize=True)
                              save_image(weight_sum[:, -3:], os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"acc_{j}.png"), nrow=1, normalize=True)
                              save_image(depth_map, os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"depth_{j}.png"), nrow=1, normalize=True)
                              j += 1
                              depth_map = depth_map[0].detach().cpu().numpy()
                              print("max:", np.nanmax(depth_map))
                              depth_map = depth_map /2. * 255.
                              depth_color = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
                              depth_color[np.isnan(depth_color)] = 0
                              imageio.imwrite(os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"depth_color_{j}.png"), depth_color)

                  save_image(torch.cat(images), os.path.join(output_dir, f"interp.png"), nrow=opt.num_frames, normalize=True)
                  save_image(torch.cat(semantic_maps), os.path.join(output_dir, f"interp_seg.png"), nrow=opt.num_frames, normalize=True)
                  save_image(torch.cat(weights_sum), os.path.join(output_dir, f"interp_acc_map.png"), nrow=opt.num_frames, normalize=True)
                  save_image(torch.cat(depths), os.path.join(output_dir, f'interp_depth_map.png'), nrow=opt.num_frames, normalize=False)
            
            else:
                  frame_images, frame_segmaps, frame_weights, frame_depths_color = [], [], [], []
                  with torch.no_grad():
                        for i, (t, pitch, yaw, fov) in tqdm(enumerate(trajectory)):
                              curriculum['h_mean'] = yaw# + 3.14/2
                              curriculum['v_mean'] = pitch# + 3.14/2
                              curriculum['fov'] = fov
                              curriculum['h_stddev'] = 0
                              curriculum['v_stddev'] = 0
                              frame, depth_map, weight_sum = generator.staged_forward_with_frequencies(*frequencyInterpolator.forward(t), max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
                              label, img = frame[:, :-3], frame[:, -3:]
                              label = mask2color(label)
                              save_image(label, os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"label_{i}.png"), nrow=1, normalize=True)
                              save_image(img, os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"img_{i}.png"), nrow=1, normalize=True)
                              save_image(weight_sum[:, -3:], os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"acc_map_{i}.png"), nrow=1, normalize=True)
                              images.append(img)
                              semantic_maps.append(label)
                              weights_sum.append(weight_sum[:, -3:])
                              frame_images.append(tensor_to_PIL(img))
                              frame_segmaps.append(tensor_to_PIL(label))
                              frame_weights.append(tensor_to_PIL(weight_sum[:, -3:]))
                              j += 1
                              depth_map = depth_map[0].detach().cpu().numpy()
                              print("max:", np.nanmax(depth_map))
                              depth_map = depth_map /2. * 255.
                              depth_color = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
                              depth_color[np.isnan(depth_color)] = 0
                              imageio.imwrite(os.path.join(output_dir, "images", f"{opt.latent_type}_{opt.trajectory}", f"depth_color_{j}.png"), depth_color)
                              frame_depths_color.append(depth_color)

                        for img, label, depth_color in zip(frame_images, frame_segmaps, frame_depths_color):
                              blend = np.array(img) * 0.5 + np.array(label) * 0.5
                              res = np.concatenate([np.array(img), np.array(label), blend, depth_color], axis=1)
                              # writer.writeFrame(res)
                              res = res[..., ::-1]
                              writer.write(res.astype('uint8'))

                  # writer.close()
                  writer.release()

            if opt.save_with_latent:
                  meta = {
                  'truncated_frequencies_geo': frequencyInterpolator.truncated_frequencies1_geo,
                  'truncated_phase_shifts_geo': frequencyInterpolator.truncated_phase_shifts1_geo,
                  'truncated_frequencies_app': frequencyInterpolator.truncated_frequencies1_app,
                  'truncated_phase_shifts_app': frequencyInterpolator.truncated_phase_shifts1_app,
                  }
                  torch.save(meta, os.path.join(output_dir, f'freq_phase_offset_{seed}.pth'))

    
def set_trajectory(trajectory_type, num_frames=None):
      if num_frames != None:
            num_frames = num_frames
      else:
            num_frames = curriculum['num_frames']
      if trajectory_type == 'front':
            trajectory = []
            for t in np.linspace(0, 1, num_frames, endpoint=True):
                  pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
                  yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                  fov = curriculum['fov'] + 5 + np.sin(t * 2 * math.pi) * 5
                  trajectory.append((t, pitch, yaw, fov))

      elif trajectory_type == 'orbit':
            trajectory = []
            for t in np.linspace(0, 0.5, num_frames, endpoint=True):
                  # pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/4
                  pitch = math.pi/2
                  yaw = t * 2 * math.pi
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif trajectory_type == 'rotation_horizontal':
            trajectory = []
            for t in np.linspace(-1, 1, num_frames//2, endpoint=True):
                  pitch = math.pi/2
                  yaw = math.pi/2 + t * 0.5
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))
            for t in np.linspace(1, -1, num_frames//2, endpoint=True):
                  pitch = math.pi/2
                  yaw = math.pi/2 + t * 0.5
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif trajectory_type == 'rotation_pi':
            trajectory = []
            for t in np.linspace(-1, 1, num_frames):
                  pitch = math.pi/2
                  yaw = math.pi/2 + t * 0.2 * math.pi
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif trajectory_type == 'non_rotation':
            trajectory = []
            for t in np.linspace(-1, 1, num_frames, endpoint=True):
                  pitch = math.pi/2
                  yaw = math.pi/2
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif trajectory_type == 'sphere':
            trajectory = []
            for t in np.linspace(0, 1, num_frames, endpoint=True):
                  pitch = 0.2 * np.cos(t * 2 * math.pi) + 1/2 * math.pi
                  yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                  fov = curriculum['fov']
                  trajectory.append((t, pitch, yaw, fov))

      elif trajectory_type == 'zoom':
            trajectory = []
            for t in np.linspace(0, 1, num_frames):
                  pitch = 1/2 * math.pi
                  yaw = math.pi/2
                  fov = curriculum['fov'] + np.sin(t * 2 * math.pi) * 5
                  trajectory.append((t, pitch, yaw, fov))
      return trajectory


eval(f'run_{opt.interpolation_type}')(opt)
