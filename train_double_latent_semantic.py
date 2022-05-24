"""Train double latent & semantic pi-GAN. Supports distributed training."""

import argparse
import os
import numpy as np
import math

from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard.summary import make_image
from torchvision.utils import save_image, make_grid

from generators import generators
from discriminators import discriminators
from siren import siren
import fid_evaluation

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy

from torch_ema import ExponentialMovingAverage

from torch.utils.tensorboard import SummaryWriter


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


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def mask2color(masks):
      masks = torch.argmax(masks, dim=1).float()
      sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float)
      for key in COLOR_MAP:
            sample_mask[masks==key] = torch.tensor(COLOR_MAP[key], dtype=torch.float)
      sample_mask = sample_mask.permute(0,3,1,2)
      return sample_mask


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def cleanup():
    dist.destroy_process_group()

def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def train(rank, world_size, opt):
    torch.manual_seed(0)

    setup(rank, world_size, opt.port)
    device = torch.device(rank)
    # with open('/apdcephfs/share_1330077/starksun/projects/pi-GAN/device_debug.txt', 'a') as f:
    #     print(rank, file=f)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z_geo = z_sampler((25, metadata['latent_geo_dim']), device='cpu', dist=metadata['z_dist'])
    fixed_z_app = z_sampler((25, metadata['latent_app_dim']), device='cpu', dist=metadata['z_dist'])

    SIREN = getattr(siren, metadata['model'])

    CHANNELS = 3

    CHANNELS_SEG = curriculum.get('channel_seg', 18)

    scaler = torch.cuda.amp.GradScaler()

    # initialize logger if rank is 0
    if rank == 0:
        logger = SummaryWriter(os.path.join(opt.output_dir, 'logs'))
    
    if opt.load_dir != '':
        if opt.load_step == 0:
            generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
            discriminator_img = torch.load(os.path.join(opt.load_dir, 'discriminator_img.pth'), map_location=device)
            discriminator_seg = torch.load(os.path.join(opt.load_dir, 'discriminator_seg.pth'), map_location=device)
            ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
            ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
        else:
            generator = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_generator.pth'), map_location=device)
            discriminator_img = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_discriminator_img.pth'), map_location=device)
            discriminator_seg = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_discriminator_seg.pth'), map_location=device)
            ema = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_ema.pth'), map_location=device)
            ema2 = torch.load(os.path.join(opt.load_dir, f'{opt.load_step}_ema2.pth'), map_location=device)
    else:
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_geo_dim'], metadata['latent_app_dim'], metadata['output_dim']).to(device)
        discriminator_img = getattr(discriminators, metadata['discriminator_img'])(metadata['latent_geo_dim'], metadata['latent_app_dim'], 3).to(device)
        discriminator_seg = getattr(discriminators, metadata['discriminator_seg'])(metadata['latent_geo_dim'], metadata['latent_app_dim'], CHANNELS_SEG + 3).to(device)        
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_img_ddp = DDP(discriminator_img, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    discriminator_seg_ddp = DDP(discriminator_seg, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)    
    generator = generator_ddp.module
    discriminator_img = discriminator_img_ddp.module
    discriminator_seg = discriminator_seg_ddp.module


    if metadata.get('unique_lr', False):
        geo_mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.geo_mapping_network.named_parameters()]
        geo_mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in geo_mapping_network_param_names]
        app_mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.app_mapping_network.named_parameters()]
        app_mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in app_mapping_network_param_names]
        generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in geo_mapping_network_param_names and n not in app_mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': geo_mapping_network_parameters, 'name': 'geo_mapping_network', 'lr':metadata['gen_lr']*5e-2},
                                        {'params': app_mapping_network_parameters, 'name': 'app_mapping_network', 'lr': metadata['gen_lr']*5e-2}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator_ddp.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_img_D = torch.optim.Adam(discriminator_img_ddp.parameters(), lr=metadata['disc_img_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    optimizer_seg_D = torch.optim.Adam(discriminator_seg_ddp.parameters(), lr=metadata['disc_seg_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator_img.step = opt.set_step
        discriminator_seg.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator_img))
        f.write(str(discriminator_seg))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator_img.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    for _ in range (opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator_img.step)

        # debug
        if metadata.get('start_density_mask', False):
            if discriminator_img.step > 10e3:
                metadata['fill_mode'] = 'debug'
                
        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'geo_mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            elif param_group.get('name', None) == 'app_mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_img_D.param_groups:
            param_group['lr'] = metadata['disc_img_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_seg_D.param_groups:
            param_group['lr'] = metadata['disc_seg_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader, CHANNELS = datasets.get_dataset_distributed(metadata['dataset'],
                                        world_size,
                                        rank,
                                        **metadata)

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator_img.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator_img.step)


            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator_img.step - step_last_upsample))

        for i, (imgs, label, _) in enumerate(dataloader):
            if discriminator_img.step % opt.model_save_interval == 0 and rank == 0:
                # now = datetime.now()
                # now = now.strftime("%d--%H:%M--")
                torch.save(ema, os.path.join(opt.output_dir, str(discriminator_img.step) + '_ema.pth'))
                torch.save(ema2, os.path.join(opt.output_dir, str(discriminator_img.step) + '_ema2.pth'))
                torch.save(generator_ddp.module, os.path.join(opt.output_dir, str(discriminator_img.step) + '_generator.pth'))
                torch.save(discriminator_img_ddp.module, os.path.join(opt.output_dir, str(discriminator_img.step) + '_discriminator_img.pth'))
                torch.save(discriminator_seg_ddp.module, os.path.join(opt.output_dir, str(discriminator_img.step) + '_discriminator_seg.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, str(discriminator_img.step) + '_optimizer_G.pth'))
                torch.save(optimizer_img_D.state_dict(), os.path.join(opt.output_dir, str(discriminator_img.step) + '_optimizer_img_D.pth'))
                torch.save(optimizer_seg_D.state_dict(), os.path.join(opt.output_dir, str(discriminator_img.step) + '_optimizer_seg_D.pth'))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, str(discriminator_img.step) + '_scaler.pth'))
            metadata = curriculums.extract_metadata(curriculum, discriminator_img.step)

            if dataloader.batch_size != metadata['batch_size']: break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator_ddp.train()
            discriminator_img_ddp.train()
            discriminator_seg_ddp.train()

            alpha = min(1, (discriminator_img.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True).float()
            real_labels = label.to(device, non_blocking=True).float()

            metadata['nerf_noise'] = max(0, 1. - discriminator_img.step/5000.)

            # TRAIN IMAGE DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z_geo = z_sampler((real_imgs.shape[0], metadata['latent_geo_dim']), device=device, dist=metadata['z_dist'])
                    z_app = z_sampler((real_imgs.shape[0], metadata['latent_app_dim']), device=device, dist=metadata['z_dist'])
                    split_batch_size = z_app.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata['batch_split']):
                        subset_z_geo = z_geo[split * split_batch_size:(split+1) * split_batch_size]
                        subset_z_app = z_app[split * split_batch_size:(split+1) * split_batch_size]
                        g_imgs, g_pos = generator_ddp(subset_z_geo, subset_z_app, **metadata)
                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                r_img_preds, _, _, _ = discriminator_img_ddp(real_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_img_real = torch.autograd.grad(outputs=scaler.scale(r_img_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_img_real = [p * inv_scale for p in grad_img_real][0]

            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_img_penalty = (grad_img_real.view(grad_img_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_img_penalty = 0.5 * metadata['r1_lambda'] * grad_img_penalty
                else:
                    grad_img_penalty = 0
                fake_imgs = gen_imgs[:, -3:]
                g_img_preds, g_img_pred_latent_geo, g_img_pred_latent_app, g_img_pred_position = discriminator_img_ddp(fake_imgs, alpha, **metadata)
                if metadata['z_geo_lambda'] > 0 or metadata['z_app_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    latent_img_penalty = metadata['z_geo_lambda'] * torch.nn.MSELoss()(g_img_pred_latent_geo, z_geo) + metadata['z_app_lambda'] * torch.nn.MSELoss()(g_img_pred_latent_app, z_app)
                    position_img_penalty = torch.nn.MSELoss()(g_img_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_img_penalty = latent_img_penalty + position_img_penalty
                else:
                    identity_img_penalty=0

                d_img_loss = torch.nn.functional.softplus(g_img_preds).mean() + torch.nn.functional.softplus(-r_img_preds).mean() + grad_img_penalty + identity_img_penalty
                discriminator_losses.append(d_img_loss.item())

            if rank == 0:
                logger.add_scalar('d_img_loss', d_img_loss.item(), discriminator_img.step)

            optimizer_img_D.zero_grad()
            scaler.scale(d_img_loss).backward()
            scaler.unscale_(optimizer_img_D)
            torch.nn.utils.clip_grad_norm_(discriminator_img_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_img_D)

            # TRAIN SEMANTIC DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z_geo = z_sampler((real_imgs.shape[0], metadata['latent_geo_dim']), device=device, dist=metadata['z_dist'])
                    z_app = z_sampler((real_imgs.shape[0], metadata['latent_app_dim']), device=device, dist=metadata['z_dist'])
                    split_batch_size = z_app.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata['batch_split']):
                        subset_z_geo = z_geo[split * split_batch_size:(split+1) * split_batch_size]
                        subset_z_app = z_app[split * split_batch_size:(split+1) * split_batch_size]
                        g_imgs, g_pos = generator_ddp(subset_z_geo, subset_z_app, **metadata)
                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_labels.requires_grad_() 
                real_imgs.requires_grad_() 
                real_labels_imgs = torch.cat([real_labels, real_imgs], dim=1)
                r_seg_preds, _, _, _ = discriminator_seg_ddp(real_labels_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Semantic Segmentation Gradient penalty
                grad_seg_real = torch.autograd.grad(outputs=scaler.scale(r_seg_preds.sum()), inputs=real_labels_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_seg_real = [p * inv_scale for p in grad_seg_real][0]


            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_seg_penalty = (grad_seg_real.view(grad_seg_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_seg_penalty = 0.5 * metadata['r1_lambda'] * grad_seg_penalty
                else:
                    grad_seg_penalty = 0

                ### fake semantic discriminator
                g_seg_preds, g_seg_pred_latent_geo, g_seg_pred_latent_app, g_seg_pred_position = discriminator_seg_ddp(gen_imgs, alpha, **metadata)
                if metadata['z_geo_lambda'] > 0 or metadata['z_app_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    latent_seg_penalty = metadata['z_app_lambda'] * torch.nn.MSELoss()(g_seg_pred_latent_app, z_app) + metadata['z_geo_lambda'] * torch.nn.MSELoss()(g_seg_pred_latent_geo, z_geo)
                    position_seg_penalty = torch.nn.MSELoss()(g_seg_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_seg_penalty = latent_seg_penalty + position_seg_penalty
                else:
                    identity_seg_penalty=0
                
                ### option 1: non-saturating loss ###
                d_seg_loss = torch.nn.functional.softplus(g_seg_preds).mean() + torch.nn.functional.softplus(-r_seg_preds).mean() + grad_seg_penalty + identity_seg_penalty
                
                ### option 2: hinge loss ###
                # d_seg_loss = (seg_gan_loss(g_seg_preds, False, for_discriminator=True).mean() + seg_gan_loss(r_seg_preds, True, for_discriminator=True).mean()) / 2.0
                
            if rank == 0:
                logger.add_scalar('d_seg_loss', d_seg_loss.item(), discriminator_img.step)

            optimizer_seg_D.zero_grad()
            scaler.scale(d_seg_loss).backward()
            scaler.unscale_(optimizer_seg_D)
            torch.nn.utils.clip_grad_norm_(discriminator_seg_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_seg_D)

            d_loss = d_img_loss.detach().item() + d_seg_loss.detach().item()
            discriminator_losses.append(d_loss)
            if rank == 0:
                logger.add_scalar('d_loss', d_loss, discriminator_img.step)


            # TRAIN GENERATOR
            z_geo = z_sampler((imgs.shape[0], metadata['latent_geo_dim']), device=device, dist=metadata['z_dist'])
            z_app = z_sampler((imgs.shape[0], metadata['latent_app_dim']), device=device, dist=metadata['z_dist'])

            split_batch_size = z_app.shape[0] // metadata['batch_split']

            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    subset_z_geo = z_geo[split * split_batch_size:(split+1) * split_batch_size]
                    subset_z_app = z_app[split * split_batch_size:(split+1) * split_batch_size]
                    gen_imgs, gen_positions = generator_ddp(subset_z_geo, subset_z_app, **metadata)
                    fake_labels, fake_imgs = gen_imgs[:, :-3], gen_imgs[:, -3:]
                    g_img_preds, g_img_pred_latent_geo, g_img_pred_latent_app, g_img_pred_position = discriminator_img_ddp(fake_imgs, alpha, **metadata)

                    # stop gradient from d_seg to g_img
                    fake_imgs = fake_imgs.detach()
                    fake_labels_imgs = torch.cat([fake_labels, fake_imgs], dim=1)
                    g_seg_preds, g_seg_pred_latent_geo, g_seg_pred_latent_app, g_seg_pred_position = discriminator_seg_ddp(fake_labels_imgs, alpha, **metadata)

                    topk_percentage = max(0.99 ** (discriminator_img.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_img_preds.shape[0])

                    g_img_preds = torch.topk(g_img_preds, topk_num, dim=0).values
                    g_seg_preds = torch.topk(g_seg_preds, topk_num, dim=0).values

                    if metadata['z_app_lambda'] > 0 or metadata['z_geo_lambda'] > 0 or metadata['pos_lambda'] > 0:
                        latent_img_penalty = metadata['z_geo_lambda'] * torch.nn.MSELoss()(g_img_pred_latent_geo, subset_z_geo) + metadata['z_app_lambda'] * torch.nn.MSELoss()(g_img_pred_latent_app, subset_z_app)
                        position_img_penalty = torch.nn.MSELoss()(g_img_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_img_penalty = latent_img_penalty + position_img_penalty
                    else:
                        identity_img_penalty = 0

                    if metadata['z_app_lambda'] > 0 or metadata['z_geo_lambda'] > 0 or metadata['pos_lambda'] > 0:
                        latent_seg_penalty = metadata['z_geo_lambda'] * torch.nn.MSELoss()(g_seg_pred_latent_geo, subset_z_geo) + metadata['z_app_lambda'] * torch.nn.MSELoss()(g_seg_pred_latent_app, subset_z_app)
                        position_seg_penalty = torch.nn.MSELoss()(g_seg_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_seg_penalty = latent_seg_penalty + position_seg_penalty
                    else:
                        identity_seg_penalty = 0
                    

                    g_img_loss = torch.nn.functional.softplus(-g_img_preds).mean() + identity_img_penalty
                    g_seg_loss = (torch.nn.functional.softplus(-g_seg_preds).mean() + identity_seg_penalty) * metadata['g_seg_loss_lambda']
                    g_loss = g_img_loss + g_seg_loss
                    generator_losses.append(g_loss.item())

                scaler.scale(g_loss).backward()
            if rank == 0:
                    logger.add_scalar('g_loss', g_loss.item(), generator.step)
                    logger.add_scalar('g_seg_loss', g_seg_loss.item(), generator.step)
                    logger.add_scalar('g_img_loss', g_img_loss.item(), generator.step)
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())

            if rank == 0:
                interior_step_bar.update(1)
                if i%10 == 0:
                    tqdm.write(f"[Experiment: {opt.output_dir}] [Epoch: {discriminator_img.epoch}/{opt.n_epochs}] [D img loss: {d_img_loss.item()}] [D seg loss: {d_seg_loss.item()}] [G loss: {g_loss.item()}] [Step: {discriminator_img.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]")

                if discriminator_img.step % opt.sample_interval == 0:
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z_geo.to(device), fixed_z_app.to(device), **copied_metadata)[0]
                            gen_labels = mask2color(gen_imgs[:, :-3])
                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_img.step}_seg_fixed.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_img.step}_img_fixed.png"), nrow=5, normalize=True)
                
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z_geo.to(device), fixed_z_app.to(device), **copied_metadata)[0]
                            gen_labels = mask2color(gen_imgs[:, :-3])
                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_img.step}_seg_tilted.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_img.step}_img_tilted.png"), nrow=5, normalize=True)
            
                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z_geo.to(device), fixed_z_app.to(device), **copied_metadata)[0]
                            gen_labels = mask2color(gen_imgs[:, :-3])
                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_img.step}_seg_fixed_ema.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_img.step}_img_fixed_ema.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z_geo.to(device),  fixed_z_app.to(device), **copied_metadata)[0]
                            gen_labels = mask2color(gen_imgs[:, :-3])
                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_img.step}_seg_tilted_ema.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_img.step}_img_tilted_ema.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['img_size'] = 128
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['psi'] = 0.7
                            gen_imgs = generator_ddp.module.staged_forward(torch.randn_like(fixed_z_geo).to(device), torch.randn_like(fixed_z_app).to(device), **copied_metadata)[0]
                            gen_labels = mask2color(gen_imgs[:, :-3])
                    save_image(gen_labels[:25], os.path.join(opt.output_dir, f"{discriminator_img.step}_seg_random.png"), nrow=5, normalize=True)
                    save_image(gen_imgs[:25, -3:], os.path.join(opt.output_dir, f"{discriminator_img.step}_img_random.png"), nrow=5, normalize=True)

                    ema.restore(generator_ddp.parameters())

                if discriminator_img.step % opt.sample_interval == 0:
                    torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                    torch.save(generator_ddp.module, os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_img_ddp.module, os.path.join(opt.output_dir, 'discriminator_img.pth'))
                    torch.save(discriminator_seg_ddp.module, os.path.join(opt.output_dir, 'discriminator_seg.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_img_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_img_D.pth'))
                    torch.save(optimizer_seg_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_seg_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))

            if opt.eval_freq > 0 and (discriminator_img.step + 1) % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')

                if rank == 0:
                    fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir, **metadata)
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images_double(generator_ddp, metadata, rank, world_size, generated_dir)
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, **metadata)
                    with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator_img.step}:{fid}')
                    logger.add_scalar('fid', fid, discriminator_img.step)

                torch.cuda.empty_cache()

            discriminator_img.step += 1
            discriminator_seg.step += 1
            generator.step += 1
        discriminator_img.epoch += 1
        discriminator_seg.epoch += 1
        generator.epoch += 1

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=2000, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)
    parser.add_argument('--num_gpus', type=int, default=1)
    
    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = opt.num_gpus
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)