from generators.neural_rendering import NeuralRenderer
import math

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step and curriculum[curriculum_step].get('img_size', 512) > current_size:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['img_size'] == current_size:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


CelebA = {
    0: {'batch_size': 24 * 2, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    # 'dataset_path': '/home/ericryanchan/data/celeba/img_align_celeba/*.jpg',
    'dataset_path': '/media/data2/sunjx/FENeRF/data/celebahq/data512x512/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 512,
    'output_dim': 4,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    # 'model': 'EmbeddingPiGAN128',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'fill_mode': 'eval_white_back',
    'target_size': 128
}


CelebA_double_semantic = {
    0: {'batch_size': 24, 'num_steps': 12, 'img_size': 32, 'batch_split': 6, 'gen_lr': 5e-5, 'disc_img_lr': 2e-4, 'disc_seg_lr': 1e-4},
    int(10e3): {'batch_size': 12, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr':2e-5, 'disc_img_lr': 1e-4, 'disc_seg_lr': 5e-5},
    int(50e3):{'batch_size': 4, 'num_steps': 24, 'img_size': 128, 'batch_split': 4, 'gen_lr': 5e-6, 'disc_img_lr': 5e-5, 'disc_seg_lr': 2e-5},
    int(500e3): {},
    # 'dataset_path': '/home/ericryanchan/data/celeba/img_align_celeba/*.jpg',
    'dataset_path': 'data/celebahq_mask',
    'background_mask': True,
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_geo_dim': 256,
    'latent_app_dim': 256,
    'output_dim': 22,
    'grad_clip': 10,
    # 'model': 'SPATIALSIRENSEMANTICDISENTANGLE',
    'model': 'SIRENBASELINESEMANTICDISENTANGLE',
    'generator': 'DoubleImplicitGenerator3d',
    'discriminator_img': 'CCSDoubleEncoderDiscriminator',
    'discriminator_seg': 'CCSDoubleEncoderDiscriminator',
    'dataset': 'CelebAMaskHQ_wo_background_seg_18',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_geo_lambda': 0,
    'z_app_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': False,
    'd_seg_loss_lambda': 0.1,
    'g_seg_loss_lambda': 0.1,
    'softmax_label': False,
    'target_size': 128,
    'fill_mode': 'seg_padding_background'
}


CelebA_double_semantic_texture_embedding_256_dim_96 = {
    0: {'batch_size': 24, 'num_steps': 24, 'img_size': 32, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_img_lr': 2e-4, 'disc_seg_lr': 2e-4},
    int(20e3): {'batch_size': 48, 'num_steps': 24, 'img_size': 64, 'batch_split': 4, 'gen_lr':6e-5, 'disc_img_lr': 2e-4, 'disc_seg_lr': 2e-4},
    int(50e3):{'batch_size': 24, 'num_steps': 24, 'img_size': 128, 'batch_split': 4, 'gen_lr': 2e-5, 'disc_img_lr': 5e-5, 'disc_seg_lr': 2e-5},
    int(500e3): {},
    'dataset_path': 'data/celebahq_mask',
    'background_mask': True,
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_geo_dim': 256,
    'latent_app_dim': 256,
    'output_dim': 22,
    'grad_clip': 10,
    # 'model': 'SIRENBASELINESEMANTICDISENTANGLE',
    'model': 'TextureEmbeddingPiGAN256SEMANTICDISENTANGLE_DIM_96',
    'generator': 'DoubleImplicitGenerator3d',
    'discriminator_img': 'CCSDoubleEncoderDiscriminator',
    'discriminator_seg': 'CCSDoubleEncoderDiscriminator',
    'dataset': 'CelebAMaskHQ_wo_background_seg_18',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_geo_lambda': 0,
    'z_app_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': False,
    'd_seg_loss_lambda': 0.1,
    'g_seg_loss_lambda': 0.1,
    'softmax_label': False,
    'target_size': 128,
    'fill_mode': 'seg_padding_background'
}

