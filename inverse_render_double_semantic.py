import argparse
from genericpath import isdir
import math
import os
from pickle import NONE
from unicodedata import normalize
from numpy.core.numeric import identity, normalize_axis_tuple, zeros_like
from torch._C import TensorType
from torchvision.utils import save_image
import glob
import torch
import numpy as np
from PIL import Image
import PIL
from tqdm import tqdm
import numpy as np
import skvideo.io
import curriculums
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import curriculums
from torch.utils.tensorboard import SummaryWriter
import lpips
import cv2
# import seaborn as sns


COLOR_MAP = {
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


COLOR_MAP_COMPLETE = {
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


def mask2labels(mask_np, color_map=COLOR_MAP):
        label_size = len(color_map.keys())
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        if label_size == 19:
            for i in range(label_size):
                labels[i][mask_np==i] = 1.0
        elif label_size == 18:
            for i in range(label_size):
                labels[i][mask_np==i+1] = 1.0
        
        return labels


# def plot_miou(data_root):
#     # plt.style.use("seaborn")
#     sns.set_theme()
#     with open(os.path.join(data_root, 'mious.npy'), 'rb') as f:
#         mious = np.load(f)
#     steps = np.arange(len(mious)) 
#     ci = 2 * np.std(mious)/np.sqrt(len(steps))
#     fig, ax = plt.subplots()
#     ax.scatter(steps, mious, s=2, alpha=0.7)
#     ax.fill_between(steps, (mious-ci), (mious+ci), color='b', alpha=0.2)
#     ax.set_xlabel("Iterations")
#     ax.set_ylabel("MIoU")
#     ax.autoscale(tight=True)
#     fp2 = np.polyfit(steps,mious,3)
#     f2 = np.poly1d(fp2)
#     fx = np.linspace(0,steps[-1],1000)
#     ax.plot(fx, f2(fx), color='b')## f2.order: 函数的阶数plt.legend(["d=%i" % f2.order],loc="upper right")
#     fig.savefig(os.path.join(data_root, 'miou.png'))


def tensor_to_PIL(img):    
    # if normalize:
    #     img = img.squeeze() * 0.5 + 0.5
    # else:
    #     img = img.squeeze()
    img = make_grid(img, normalize=True)
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


def mIOU(source, target):
    mIOU = torch.mean(torch.div(
        torch.sum(source * target, dim=[2, 3]).float(),
        torch.sum((source + target)>0, dim=[2, 3]).float() + 1e-6), dim=1)
    return mIOU


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, default='debug')
parser.add_argument('generator_path', type=str)
parser.add_argument('--image_path', type=str)
parser.add_argument('--seg_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--load_checkpoint', type=bool, default=False)
parser.add_argument('--seeds', nargs='+', default=[0])
parser.add_argument("--init_seed", default=0, type=int)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--fov', default=12, type=int)
parser.add_argument('--num_frames', type=int, default=100)
parser.add_argument('--max_batch_size', type=int, default=2400000)
parser.add_argument("--lock_view_dependence", default=False)
parser.add_argument("--iteration", type=int, default=1000)
parser.add_argument("--background_mask", action='store_true')
parser.add_argument("--white_background_mask", action='store_true')
parser.add_argument("--inverse_type", default='semantic', help='inverse rendering signal, i.e. semantic map or image or both') 
parser.add_argument("--img_loss", default='mse')
parser.add_argument("--seg_loss", type=str, default='mse')
parser.add_argument("--lambda_img", type=float, default=0.)
parser.add_argument("--lambda_seg", type=float, default=0.)
parser.add_argument("--lambda_percept", type=float, default=0.)
parser.add_argument("--lambda_norm", type=float, default=1.)
parser.add_argument("--latent_normalize", action="store_true")
parser.add_argument("--latent_type", default='app')
parser.add_argument("--psi", type=float, default=0)
parser.add_argument("--init_psi", type=float, default=0)
parser.add_argument("--trajectory", default='front')
parser.add_argument('--depth_map', action='store_true')
parser.add_argument("--save_with_video", action='store_true')
parser.add_argument("--recon", action="store_true")
parser.add_argument("--fill_color", type=str, default='black', help='the rendering background color, only for segmantic 18 type models')
parser.add_argument("--no_center_crop", action='store_true')
parser.add_argument("--checkpoint_path", default='', type=str)
opt = parser.parse_args()
generator = torch.load(opt.generator_path, map_location=torch.device(device))
ema_file = opt.generator_path.split('generator')[0] + 'ema.pth'
ema = torch.load(ema_file, map_location=device)
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()
generator.softmax_label = False
percept = lpips.LPIPS(net='vgg',  version='0.0').to(device)

transform_img = transforms.Compose(
                    [transforms.Resize(320), 
                    transforms.CenterCrop(256), 
                    transforms.Resize((opt.image_size, opt.image_size), 
                    interpolation=PIL.Image.NEAREST), 
                    transforms.ToTensor(), 
                    transforms.Normalize([0.5], [0.5])])

transform_seg = transforms.Compose(
                    [transforms.Resize(320), 
                    transforms.CenterCrop(256), 
                    transforms.Resize((opt.image_size, opt.image_size), 
                    interpolation=PIL.Image.NEAREST), 
                    transforms.ToTensor()])    

transform_seg_19 = transforms.Compose(
                    [transforms.Resize(320), 
                    transforms.CenterCrop(256), 
                    transforms.Resize((256, 256), 
                    interpolation=PIL.Image.NEAREST), 
                    transforms.ToTensor()]) 

### transform without crop ###
if opt.no_center_crop:
    transform_img = transforms.Compose(
                        [
                        transforms.Resize((opt.image_size, opt.image_size), 
                        interpolation=PIL.Image.NEAREST), 
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])])

    transform_seg = transforms.Compose(
                        [ 
                        transforms.Resize((opt.image_size, opt.image_size), 
                        interpolation=PIL.Image.NEAREST), 
                        transforms.ToTensor()])    

    transform_seg_19 = transforms.Compose(
                        [
                        transforms.Resize((256, 256), 
                        interpolation=PIL.Image.NEAREST), 
                        transforms.ToTensor()]) 


options = {
    'img_size': opt.image_size,
    'fov': opt.fov,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'num_steps': 24,
    'h_stddev': 0,
    'v_stddev': 0,
    'h_mean': torch.tensor(math.pi/2).to(device),
    'v_mean': torch.tensor(math.pi/2).to(device),
    'hierarchical_sample': False,
    'sample_dist': None,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
    'fade_steps': 10000,
    'z_app_lambda': 0,
    'z_geo_lambda': 0,
    'pos_lambda': 0,
    'tok_interval': 2000,
    'tok_v': 0.6,
    'betas': (0, 0.9),
    'fill_mode': 'eval_seg_padding_background'
}

render_options = {
    'img_size': 256,
    'fov': opt.fov,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'num_steps': 48,
    'h_stddev': 0,
    'v_stddev': 0,
    'v_mean': math.pi/2,
    'hierarchical_sample': True,
    'sample_dist': None,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
    'last_back': False,
    'fill_mode': 'eval_seg_padding_background',
    'fill_color': opt.fill_color
}


def run_inverse_render(opt, img_path, seg_path):
        torch.manual_seed(opt.init_seed)
        img_ind = os.path.basename(img_path).split('.')[0]
        seg_ind = os.path.basename(seg_path).split('.')[0]
        save_dir = opt.save_dir
        os.makedirs(save_dir, exist_ok=True)
        mious = []
        ### check if any checkpoint path is spicified
        checkpoint_path = opt.checkpoint_path

        print(checkpoint_path)
        if not os.path.exists(checkpoint_path) or opt.load_checkpoint:
            gt_image = Image.open(img_path).convert('RGB')
            gt_seg = Image.open(seg_path).convert('L')
            width, height = gt_image.size
            if opt.background_mask:
                trans = transforms.Compose([transforms.ToTensor()])
                trans_inv = transforms.Compose([transforms.ToPILImage()])
                # debug: 
                i, l = trans(gt_image), trans(gt_seg.resize((width, height), resample=PIL.Image.NEAREST)) * 255.
                l = l.expand_as(i)
                i[l == 0] = 0
                gt_image = trans_inv(i) 
            elif opt.white_background_mask:
                trans = transforms.Compose([transforms.ToTensor()])
                trans_inv = transforms.Compose([transforms.ToPILImage()])
                # debug: 
                i, l = trans(gt_image), trans(gt_seg.resize((width, height), resample=PIL.Image.NEAREST)) * 255.
                l = l.expand_as(i)
                i[l == 0] = 1
                gt_image = trans_inv(i) 
            gt_image.save("debug.png")
            gt_image = transform_img(gt_image)[None].to(device)
            gt_seg_18 = transform_seg(gt_seg)
            # debug:
            gt_seg_18 = mask2labels((gt_seg_18 * 255.)[0])
            gt_seg_18 = (gt_seg_18 - 0.5) / 0.5
            gt_seg_18 = torch.tensor(gt_seg_18, dtype=torch.float)[None].to(device)
            gt_seg_19 = transform_seg_19(gt_seg)
            gt_seg_19 = mask2labels((gt_seg_19*255.)[0], COLOR_MAP_COMPLETE)
            gt_seg_19 = torch.tensor(gt_seg_19, dtype=torch.float)[None]

            ### init latent code and optimized offset ###
            z_geo = torch.randn((10000, 256), device=device)
            z_geo_mean = z_geo.mean(0, keepdim=True)
            rand_z_geo = torch.randn((1, 256), device=device)
            with torch.no_grad():
                geo_frequencies, geo_phase_shifts = generator.siren.geo_mapping_network(z_geo)
                rand_geo_frequencies, rand_geo_phase_shifts = generator.siren.geo_mapping_network(rand_z_geo)
            # mean
            w_geo_frequencies = geo_frequencies.mean(0, keepdim=True) 
            w_geo_phase_shifts = geo_phase_shifts.mean(0, keepdim=True)
            w_geo_frequencies = w_geo_frequencies + opt.init_psi * (rand_geo_frequencies - w_geo_frequencies)
            w_geo_phase_shifts = w_geo_phase_shifts + opt.init_psi * (rand_geo_phase_shifts - w_geo_phase_shifts)
            
            # std 
            geo_frequencies_std = torch.std(geo_frequencies, dim=0, keepdim=True)
            geo_phase_shifts_std = torch.std(geo_frequencies, dim=0, keepdim=True)
            
            # offsets
            w_geo_frequency_offsets = torch.zeros_like(w_geo_frequencies)
            w_geo_phase_shift_offsets = torch.zeros_like(w_geo_phase_shifts)
            w_geo_frequency_offsets.requires_grad_()
            w_geo_phase_shift_offsets.requires_grad_()
            
            z_app = torch.randn((10000, 256), device=device)
            z_app_mean = z_app.mean(0, keepdim=True)
            rand_z_app = torch.randn((1, 256), device=device)
            with torch.no_grad():
                  app_frequencies, app_pahse_shifts = generator.siren.app_mapping_network(z_app)
                  rand_app_frequencies, rand_app_phase_shifts = generator.siren.app_mapping_network(rand_z_app)
      
            # mean
            w_app_frequencies = app_frequencies.mean(0, keepdim=True)
            w_app_phase_shifts = app_pahse_shifts.mean(0, keepdim=True)
            w_app_frequencies = w_app_frequencies + opt.init_psi * (rand_app_frequencies - w_app_frequencies)
            w_app_phase_shifts = w_app_phase_shifts + opt.init_psi * (rand_app_phase_shifts - w_app_phase_shifts)

            # std
            app_frequencies_std = torch.std(app_frequencies, dim=0, keepdim=True)
            app_phase_shifts_std = torch.std(app_frequencies, dim=0, keepdim=True)

            # offsets
            w_app_frequency_offsets = torch.zeros_like(w_app_frequencies)
            w_app_phase_shift_offsets = torch.zeros_like(w_app_phase_shifts)
            w_app_frequency_offsets.requires_grad_()
            w_app_phase_shift_offsets.requires_grad_()

            if opt.load_checkpoint:
                meta = torch.load(opt.checkpoint_path)
                app_frequency_offsets, app_phase_shift_offsets = meta['w_app_frequency_offsets'].to(device), meta['w_app_phase_shift_offsets'].to(device)
                w_app_frequencies, w_app_phase_shifts = meta['w_app_frequencies'].detach().to(device), meta['w_app_phase_shifts'].detach().to(device)
                geo_frequency_offsets, geo_phase_shift_offsets = meta['w_geo_frequency_offsets'].to(device), meta['w_geo_phase_shift_offsets'].to(device)
                w_geo_frequencies, w_geo_phase_shifts = meta['w_geo_frequencies'].detach().to(device), meta['w_geo_phase_shifts'].detach().to(device)
                w_app_frequencies += app_frequency_offsets
                w_app_phase_shifts += app_phase_shift_offsets
                w_geo_frequencies += geo_frequency_offsets
                w_geo_phase_shifts += geo_phase_shift_offsets


            # initialize logger
            logdir = os.path.join(save_dir, "logs")
            logger = SummaryWriter(logdir)

            n_iterations = opt.iteration

            ### initialize optimizer and scheduler ###  
            if opt.lambda_img == 0:
                  optimizer = torch.optim.Adam([w_geo_frequency_offsets, w_geo_phase_shift_offsets], lr=1e-2, weight_decay = 1e-4)
            elif opt.lambda_seg == 0:
                  optimizer = torch.optim.Adam([w_app_frequency_offsets, w_app_phase_shift_offsets], lr=1e-2, weight_decay = 1e-4)
            elif opt.lambda_img > 0 and opt.lambda_seg > 0:
                  optimizer = torch.optim.Adam([w_geo_frequency_offsets, w_geo_phase_shift_offsets, w_app_frequency_offsets, w_app_phase_shift_offsets], lr=1e-2, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.75)

            ### Start training ###
            for i in range(n_iterations):
                  noise_w_geo_frequencies = 0.03 * torch.randn_like(w_geo_frequencies) * (n_iterations - i)/n_iterations
                  noise_w_geo_phase_shifts = 0.03 * torch.randn_like(w_geo_phase_shifts) * (n_iterations - i)/n_iterations
                  noise_w_app_frequencies = 0.03 * torch.randn_like(w_app_frequencies) * (n_iterations - i)/n_iterations
                  noise_w_app_phase_shifts = 0.03 * torch.randn_like(w_app_phase_shifts) * (n_iterations - i)/n_iterations
                  frame, position = generator.forward_with_frequencies(w_geo_frequencies + noise_w_geo_frequencies + w_geo_frequency_offsets, w_app_frequencies + noise_w_app_frequencies + w_app_frequency_offsets, w_geo_phase_shifts + noise_w_geo_phase_shifts + w_geo_phase_shift_offsets, w_app_phase_shifts + noise_w_app_phase_shifts + w_app_phase_shift_offsets, **options)
                  
                  seg_loss = torch.nn.MSELoss(reduction="mean")(frame[:, :-3], gt_seg_18) # can also use cross-entropy loss
                  img_loss = torch.nn.MSELoss(reduction="mean")(frame[:, -3:], gt_image)

                  p_loss = percept(frame[:, -3:], gt_image).sum()
                  
                  loss = opt.lambda_seg * seg_loss + opt.lambda_img * img_loss + opt.lambda_percept * p_loss
                  
                  if opt.latent_normalize:
                        # norm_loss = ((w_geo_frequency_offsets / geo_frequencies_std) ** 2).mean()
                        # norm_loss += ((w_geo_phase_shift_offsets / geo_phase_shifts_std) ** 2).mean()
                        # norm_loss += ((w_app_frequency_offsets / app_frequencies_std) ** 2).mean()
                        # norm_loss += ((w_app_phase_shift_offsets / app_phase_shifts_std) ** 2).mean()
                        norm_loss = (w_geo_frequency_offsets ** 2).mean()
                        norm_loss += (w_geo_phase_shift_offsets ** 2).mean()
                        norm_loss += (w_app_frequency_offsets ** 2).mean()
                        norm_loss += (w_app_phase_shift_offsets ** 2).mean()

                  loss += opt.lambda_norm * norm_loss

                  
                  loss.backward()
                  optimizer.step()
                  optimizer.zero_grad()
                  scheduler.step()

                  if i % 200 == 0:
                        # gen_labels = mask2color(frame[:, :-3]) # save labels
                        # save_image(frame[:, -3:], os.path.join(save_dir, f"{i}_img.jpg"), normalize=True)
                        # save_image(gen_labels, os.path.join(save_dir, f"{i}_seg.jpg"), normalize=True)

                        with torch.no_grad():
                            # vis_trajectory = set_trajectory(opt)
                            # y = 0
                            for angle in [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                            # for t, pitch, yaw, fov in tqdm(vis_trajectory):
                                # render_options['h_mean'] = yaw
                                # render_options['v_mean'] = pitch
                                # render_options['fov'] = fov
                                # render_options['h_stddev'] = 0
                                # render_options['v_stddev'] = 0
                                img, _, _ = generator.staged_forward_with_frequencies(w_geo_frequencies + w_geo_frequency_offsets, w_app_frequencies + w_app_frequency_offsets, w_geo_phase_shifts + w_geo_phase_shift_offsets, w_app_phase_shifts + w_app_phase_shift_offsets, h_mean=math.pi/2+angle, max_batch_size=opt.max_batch_size, lock_view_dependence=opt.lock_view_dependence, **render_options)
                                gen_labels = mask2color(img[:, :-3]) # save labels
                                save_image(img[:, -3:], os.path.join(save_dir, f"{i}_{angle}_img.jpg"), normalize=True)
                                save_image(gen_labels, os.path.join(save_dir, f"{i}_{angle}_seg.jpg"), normalize=True)
                                gen_masks = torch.argmax(img[:, :-3], dim=1).float()
                                gen_masks = mask2labels(gen_masks[0].detach().cpu().numpy(), COLOR_MAP_COMPLETE)
                                # y += 1
                                # if angle == 0:
                                #       miou = mIOU(torch.Tensor(gen_masks[None]), gt_seg_19)
                                #       logger.add_scalar(f'mIoU', miou.item(), i)
                  if i % 20 == 0:
                      with torch.no_grad():  
                              for angle in [0]:
                                    # render_options['h_mean'] = yaw
                                    img, _, _ = generator.staged_forward_with_frequencies(w_geo_frequencies + w_geo_frequency_offsets, w_app_frequencies + w_app_frequency_offsets, w_geo_phase_shifts + w_geo_phase_shift_offsets, w_app_phase_shifts + w_app_phase_shift_offsets, max_batch_size=opt.max_batch_size, h_mean=math.pi/2+angle, lock_view_dependence=opt.lock_view_dependence, **render_options)
                                    gen_labels = mask2color(img[:, :-3]) # save labels
                                    gen_masks = torch.argmax(img[:, :-3], dim=1).float()
                                    gen_masks = mask2labels(gen_masks[0].detach().cpu().numpy(), COLOR_MAP_COMPLETE)
                                    if angle == 0:
                                          miou = mIOU(torch.Tensor(gen_masks[None]), gt_seg_19)
                                          logger.add_scalar(f'mIoU', miou.item(), i)
                                          mious.append(miou.item())
                
                                          

                              
            meta = {
                  'w_geo_frequencies': w_geo_frequencies,
                  'w_geo_phase_shifts': w_geo_phase_shifts,
                  'w_geo_frequency_offsets': w_geo_frequency_offsets,
                  'w_geo_phase_shift_offsets': w_geo_phase_shift_offsets,
                  'w_app_frequencies': w_app_frequencies,
                  'w_app_phase_shifts': w_app_phase_shifts,
                  'w_app_frequency_offsets': w_app_frequency_offsets,
                  'w_app_phase_shift_offsets': w_app_phase_shift_offsets
            }
            checkpoint_path =  os.path.join(save_dir, f'freq_phase_offset_{opt.name}.pth')
            torch.save(meta, checkpoint_path)
            miou_file = os.path.join(save_dir, f'mious.npy')
            np.save(miou_file, mious)
            # plot_miou(save_dir)
        return checkpoint_path
    

def run_render_recon_video(opt, checkpoint_path):
    # render video
    meta = torch.load(checkpoint_path)
    trajectory = set_trajectory(opt)
    w_geo_frequency_offsets, w_geo_phase_shift_offsets, w_app_frequency_offsets, w_app_phase_shift_offsets = meta['w_geo_frequency_offsets'].to(device), meta['w_geo_phase_shift_offsets'].to(device), meta['w_app_frequency_offsets'].to(device), meta['w_app_phase_shift_offsets'].to(device)
    w_geo_frequencies, w_geo_phase_shifts, w_app_frequencies, w_app_phase_shifts = meta['w_geo_frequencies'].to(device), meta['w_geo_phase_shifts'].to(device), meta['w_app_frequencies'].to(device), meta['w_app_phase_shifts'].to(device)
    output_name = f'reconstructed_debug_{opt.trajectory}_{opt.fill_color}.mp4'

    # writer = skvideo.io.FFmpegWriter(os.path.join(opt.save_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
    ### debug: replace ffmpeg with cv2
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    nrows, ncols = 1, 3
    writer = cv2.VideoWriter(os.path.join(opt.save_dir, output_name), fourcc, 25, (256 * ncols, 256 * nrows))

    semantic_maps = []
    images = []

    with torch.no_grad():
        for _, pitch, yaw, _ in tqdm(trajectory):
                render_options['h_mean'] = yaw 
                render_options['v_mean'] = pitch 

                frame, depth_map, _ = generator.staged_forward_with_frequencies(w_geo_frequencies + w_geo_frequency_offsets, w_app_frequencies + w_app_frequency_offsets, w_geo_phase_shifts + w_geo_phase_shift_offsets, w_app_phase_shifts + w_app_phase_shift_offsets, max_batch_size=opt.max_batch_size, lock_view_dependence=opt.lock_view_dependence, **render_options)
                semantic_map, image = frame[:, :-3], frame[:, -3:]
                semantic_map = mask2color(semantic_map)
                images.append(tensor_to_PIL(image))
                semantic_maps.append(tensor_to_PIL(semantic_map))
                # depths.append(depth_map.unsqueeze(0).expand(-1, 3, -1, -1).squeeze().permute(1, 2, 0).cpu().numpy())

    for image, semantic_map in zip(images, semantic_maps):
        blend = np.array(image) * 0.5 + np.array(semantic_map) * 0.5
        res = np.concatenate([np.array(image), np.array(semantic_map), blend], axis=1)
        res = res[..., ::-1]
        writer.write(res.astype('uint8'))
    writer.release()

    return meta

   
def set_trajectory(opt):
    if opt.trajectory == 'front':
        trajectory = []
        for t in np.linspace(0, 1, opt.num_frames):
                pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
                yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                fov = render_options['fov'] + 5 + np.sin(t * 2 * math.pi) * 5
                trajectory.append((t, pitch, yaw, fov))

    elif opt.trajectory == 'orbit':
        trajectory = []
        for t in np.linspace(0, 0.5, opt.num_frames):
                # pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/4
                pitch = math.pi/2
                yaw = t * 2 * math.pi
                fov = render_options['fov']
                trajectory.append((t, pitch, yaw, fov))
    elif opt.trajectory == 'non_rotation':
        trajectory = []
        for t in np.linspace(0, 1, opt.num_frames):
                pitch = math.pi/2
                yaw = math.pi/2
                fov = render_options['fov']
                trajectory.append((t, pitch, yaw, fov))
    elif opt.trajectory == 'sphere':
            trajectory = []
            for t in np.linspace(0, 1, opt.num_frames):
                  pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi/2
                  yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                  fov = render_options['fov']
                  trajectory.append((t, pitch, yaw, fov))
    elif opt.trajectory == 'inverse_sphere':
            trajectory = []
            for t in np.linspace(0, 1, opt.num_frames):
                  pitch = 0.2 * (1 - np.cos(t * 2 * math.pi)) + math.pi/2
                  yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
                  fov = render_options['fov']
                  trajectory.append((t, pitch, yaw, fov))
    elif opt.trajectory == 'rotation_horizontal':
        trajectory = []
        for t in np.linspace(-1, 1, opt.num_frames):
                pitch = math.pi/2
                yaw = math.pi/2 + t * 0.5
                fov = render_options['fov']
                trajectory.append((t, pitch, yaw, fov))
    elif opt.trajectory == 'zoom':
        trajectory = []
        for t in np.linspace(-1, 1):
            pitch = math.pi/2
            yaw = math.pi/2
            fov = render_options['fov'] + 5 + np.sin(t * 2 * math.pi) * 5
            trajectory.append((t, pitch, yaw, fov))
    elif opt.trajectory == 'rotation_linear':
        trajectory = []
        for t in np.linspace(-0.4, 0.4, opt.num_frames):
            pitch = math.pi/2
            yaw = math.pi/2 + t
            fov = render_options['fov']
            trajectory.append((t, pitch, yaw, fov))

    return trajectory


if __name__ == "__main__":
    if os.path.isdir(opt.image_path) and os.path.isdir(opt.seg_path):
        img_paths, seg_paths = sorted(glob.glob(opt.image_path + '/*.jpg')), sorted(glob.glob(opt.seg_path + '/*.png'))
        for img_path, seg_path in zip(img_paths, seg_paths):
                checkpont_path = run_inverse_render(opt, img_path, seg_path)
                if opt.recon:
                    run_render_recon_video(opt, checkpont_path)

    elif os.path.isfile(opt.image_path) and os.path.isfile(opt.seg_path):
        checkpoint_path = run_inverse_render(opt, opt.image_path, opt.seg_path)
        if opt.recon:
            run_render_recon_video(opt, checkpoint_path)



    

