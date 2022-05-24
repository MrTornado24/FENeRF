"""Script to render a video using a trained pi-GAN  model."""
import argparse
import math
import os
from cv2 import IMWRITE_JPEG_RST_INTERVAL, data
from torch.serialization import save
from torchvision import transforms
from torchvision.utils import save_image
import glob 
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import skvideo.io
import curriculums
import cv2
from os.path import basename
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import dlib
from siren.BiSeNet import BiSeNet


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
      

def frame2video(img_path, save_path, w, h):
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      video_writer = cv2.VideoWriter(f'{save_path}.avi', fourcc, 2, (w, h))
      for path in sorted(img_path):
            # frames.append(Image.open(path).convert('RGB'))
            img = cv2.imread(path)
            video_writer.write(img)


def gen_face_mask(data_root):
      label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
      # 总共18个区域，加上背景19个
      folder_base = data_root + 'celebahq_mask_anno'
      folder_save = data_root + 'celebahq_mask_mask'
      img_num = 30000

      if not os.path.exists(folder_save):
            os.makedirs(folder_save)

      for k in tqdm(range(img_num)):
            if k != 18290:
                  continue
            folder_num = k // 2000
            im_base = np.zeros((512, 512))
            for idx, label in enumerate(label_list):
                  filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
                  if (os.path.exists(filename)):
                        # print (label, idx+1)
                        im = cv2.imread(filename)
                        im = im[:, :, 0]
                        im_base[im != 0] = (idx + 1)
            # print (filename_save)
            filename_save = os.path.join(folder_save, str(k) + '.png')
            cv2.imwrite(filename_save, im_base)


def gen_face_simplifed_mask(data_root):
      label_list = {
                  'skin': 1, 
                  'nose': 2, 
                  'eye_g': 3, 
                  'l_eye': 3,
                  'r_eye': 4,
                  'l_brow': 3, 
                  'r_brow': 4, 
                  'l_ear': 1,
                  'r_ear': 1, 
                  'mouth': 5,
                  'u_lip': 5,
                  'l_lip': 5, 
                  'hair': 6, 
                  'hat': 6, 
                  'ear_r': 1 , 
                  'neck_l': 1, 
                  'neck': 1, 
      }

      
      
      folder_base = data_root + 'celebahq_mask_anno'
      folder_save = data_root + 'celebahq_mask_mask_simplified'
      img_num = 30000

      if not os.path.exists(folder_save):
            os.makedirs(folder_save)

      for k in tqdm(range(img_num)):
            if k < 25175:
                  continue
            folder_num = k // 2000
            im_base = np.zeros((512, 512))
            for label, mask in label_list.items():
                  filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
                  if (os.path.exists(filename)):
                        # print (label, idx+1)
                        im = cv2.imread(filename)
                        im = im[:, :, 0]
                        im_base[im != 0] = mask
            
            # fuse brow and eye
            idx_left_brow = np.where(im_base==3)
            idx_right_brow = np.where(im_base==4)
            if len(idx_left_brow[0]) == 0 or len(idx_left_brow[1]) == 0 or len(idx_right_brow[0]) == 0 or len(idx_right_brow[1]) == 0:
                  continue
            corner_left_brow = [np.min(idx_left_brow[0]), np.max(idx_left_brow[0]) + 5, np.min(idx_left_brow[1]), np.max(idx_left_brow[1])]
            corner_right_brow = [np.min(idx_right_brow[0]), np.max(idx_right_brow[0]) + 5, np.min(idx_right_brow[1]), np.max(idx_right_brow[1])]
            im_base[corner_left_brow[0]: corner_left_brow[1], corner_left_brow[2]:corner_left_brow[3]] = 3
            im_base[corner_right_brow[0]: corner_right_brow[1], corner_right_brow[2]:corner_right_brow[3]] = 4

            # print (filename_save)
            filename_save = os.path.join(folder_save, str(k) + '.png')
            cv2.imwrite(filename_save, im_base)

      
      
      folder_base = data_root + 'celebahq_mask_anno'
      folder_save = data_root + 'celebahq_mask_mask_simplified'
      img_num = 30000

      if not os.path.exists(folder_save):
            os.makedirs(folder_save)

      for k in tqdm(range(img_num)):
            if k < 25175:
                  continue
            folder_num = k // 2000
            im_base = np.zeros((512, 512))
            for label, mask in label_list.items():
                  filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
                  if (os.path.exists(filename)):
                        # print (label, idx+1)
                        im = cv2.imread(filename)
                        im = im[:, :, 0]
                        im_base[im != 0] = mask
            
            # fuse brow and eye
            idx_left_brow = np.where(im_base==3)
            idx_right_brow = np.where(im_base==4)
            if len(idx_left_brow[0]) == 0 or len(idx_left_brow[1]) == 0 or len(idx_right_brow[0]) == 0 or len(idx_right_brow[1]) == 0:
                  continue
            corner_left_brow = [np.min(idx_left_brow[0]), np.max(idx_left_brow[0]) + 5, np.min(idx_left_brow[1]), np.max(idx_left_brow[1])]
            corner_right_brow = [np.min(idx_right_brow[0]), np.max(idx_right_brow[0]) + 5, np.min(idx_right_brow[1]), np.max(idx_right_brow[1])]
            im_base[corner_left_brow[0]: corner_left_brow[1], corner_left_brow[2]:corner_left_brow[3]] = 3
            im_base[corner_right_brow[0]: corner_right_brow[1], corner_right_brow[2]:corner_right_brow[3]] = 4

            # print (filename_save)
            filename_save = os.path.join(folder_save, str(k) + '.png')
            cv2.imwrite(filename_save, im_base)


color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
color_simplified_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [51, 51, 255], [204, 0, 204], [102, 204, 0], [0, 0, 204]]

def gen_face_mask_color(data_root):
      
      folder_base = data_root + 'celebahq_mask_mask'
      folder_save = data_root + 'celebahq_mask_color'
      img_num = 1

      if not os.path.exists(folder_save):
            os.makedirs(folder_save)

      for k in tqdm(range(img_num)):
            # filename = os.path.join(folder_base, str(k) + '.png')
            filename = 'debug_2.png'
            if (os.path.exists(filename)):
                  im_base = np.zeros((512, 512, 3))
                  im = Image.open(filename)
                  im = np.array(im)
                  for idx, color in enumerate(color_list):
                        im_base[im == idx] = color
            filename_save = os.path.join(folder_save, str(k) + '.png')
            result = Image.fromarray((im_base).astype(np.uint8))
            result.save(filename_save)


def gen_face_simplified_mask_color(data_root):
      
      folder_base = data_root + 'celebahq_mask_mask_simplified'
      folder_save = data_root + 'celebahq_mask_color_simplified'
      img_num = 30000

      if not os.path.exists(folder_save):
            os.makedirs(folder_save)

      for k in tqdm(range(img_num)):
            filename = os.path.join(folder_base, str(k) + '.png')
            # filename = 'debug_2.png'
            if (os.path.exists(filename)):
                  im_base = np.zeros((512, 512, 3))
                  im = Image.open(filename)
                  im = np.array(im)
                  for idx, color in enumerate(color_simplified_list):
                        im_base[im == idx] = color
            filename_save = os.path.join(folder_save, str(k) + '.png')
            result = Image.fromarray((im_base).astype(np.uint8))
            result.save(filename_save)


def reorder_index(path):
      imgs = glob.glob(path + '/*.jpg')
      for img in imgs:
            ind = int(img.split('/')[-1].split('.')[0])
            ind = '%06d'%ind
            save_path = os.path.join(img, '..', f'{ind}.jpg')
            cmd = f'mv {img} {save_path}'
            os.system(cmd)


def gen_black_back_images(img_path, label_path):
      import PIL
      import torchvision.transforms as transforms
      save_path = '/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/celebahq_mask/celebahq_mask_img_black_back'
      if not os.path.exists(save_path):
            os.makedirs(save_path)
      img_lists = sorted(glob.glob(img_path + '/*.jpg'))
      label_lists = sorted(glob.glob(label_path + '/*.png'))

      for i_path, l_path in tqdm(zip(img_lists, label_lists)):
            img = PIL.Image.open(i_path).convert('RGB')
            label = PIL.Image.open(l_path).convert('L')
            trans = transforms.Compose([transforms.ToTensor()])
            trans_inv = transforms.Compose([transforms.ToPILImage()])
            i, l = trans(img), trans(label.resize((1024, 1024), resample=PIL.Image.NEAREST)) * 255.
            l = l.expand_as(i)
            i[l == 0] = 0
            img = trans_inv(i)
            index = i_path.split('/')[-1]
            img.save(os.path.join(save_path, index))




def main():
      # data_root = '/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/celebahq_mask/'
      # # gen_face_mask(data_root)
      # # gen_face_mask_color(data_root)
      # gen_face_simplifed_mask(data_root)
      # gen_face_simplified_mask_color(data_root)
      # img_path = '/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/celebahq_mask/celebahq_mask_img'
      # label_path = '/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/celebahq_mask/celebahq_mask_mask'
      # gen_black_back_images(img_path, label_path)
      debug()
    

def mask2color(mask, save_path):
      im_base = np.zeros((512, 512, 3))
      for idx, color in enumerate(color_list):
            im_base[mask == idx] = color
      result = Image.fromarray((im_base).astype(np.uint8))
      result.save(save_path)


def debug():
      
      img_num = 1
      for k in tqdm(range(img_num)):
            # filename = os.path.join(folder_base, str(k) + '.png')
            filename = 'data/demo/mask_02.png'
            filename2 = 'data/demo/mask_105.png'
            if (os.path.exists(filename)):
                  im = Image.open(filename)
                  im2 = Image.open(filename2)
                  im = np.array(im)
                  im2 = np.array(im2)
                  im_temp = im.copy()
                  im_temp[im == 10] = 1
                  im_temp[im == 11] = 1
                  im_temp[im == 12] = 1
                  # generate image without mouth
                  result = Image.fromarray((im_temp).astype(np.uint8))
                  result.save('mask_02_no_face.png')
                  mask2color(im_temp, 'mask_color_02_no_mouth.png')

                  # generate image without right brow
                  im_temp = im.copy()
                  im_temp[im==7] = 1
                  result = Image.fromarray((im_temp).astype(np.uint8))
                  result.save('mask_02_no_r_brow.png')
                  mask2color(im_temp, 'mask_color_02_no_r_brow.png')

                  # generate image without mouth swapped with 105
                  im_temp = im.copy()
                  im_temp[im == 10] = 1
                  im_temp[im == 11] = 1
                  im_temp[im == 12] = 1
                  im_temp[im2 == 10] = 10
                  im_temp[im2 == 11] = 11
                  im_temp[im2 == 12] = 12
                  result = Image.fromarray((im_temp).astype(np.uint8))
                  result.save('mask_02_mouth_swapped.png')
                  mask2color(im_temp, 'mask_color_02_mouth_swapped.png')


def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(filepath, predictor):
    """
    :param filepath: str
    :return: PIL Image
    """

    lm = get_landmark(filepath, predictor)

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Return aligned image.
    return img


def select_img():
      img_root = 'data/celebahq_mask/celebahq_mask_img'
      mask_root = 'data/celebahq_mask/celebahq_mask_mask'
      mask_color_root = 'data/celebahq_mask/celebahq_mask_color'

      # inds = [4,5,7,8,12,13,16,18,20,30,
      #       32,36,39,42,43,44,48,50,52,53,
      #       55,61,62,65,66,67,69,75,77,78,
      #       88,99,100,103,106,109,110,111,118,121,
      #       122,125,128,130,131,136,137,142,146,152]
      # inds = [156, 162, 163, 164, 178, 182, 204, 208, 216, 217, 220, 222, 224, 228, 229, 233, 235, 241, 250, 267]
      # inds = [255, 265, 266, 272, 287, 293, 296, 301, 302, 303, 305, 306, 315, 317, 326, 327]
      # inds = [333, 338, 340, 345, 348, 351, 357, 358, 359, 383, 412, 416, 419, 420, 426, 434, 447, 455, 458, 459, 460, 464, 466, 483, 491, 505, 513, 514, 522, 524, 525, 526, 535, 541, 547, 553, 557, 564, 572, 581, 590, 601, 612, 611, 613]
      # inds = [639, 567, 681, 724, 729, 742, 761, 801, 838, 848, 872, 876, 885, 934]
      # inds = [588, 600, 611, 620, 624, 625, 626, 628, 635, 644, 648, 653, 656, 657, 663, 668, 669, 673, 681, 684, 703, 707, 712, 717, 721, 723, 726, 734, 735,
      #         739, 750, 768, 772, 774, 775, 780, 781, 786, 791, 802, 806, 807, 810, 811, 818, 823, 824]
      inds = [1038, 1068, 1083, 1124, 1147, 1149, 1155, 1165, 1170, 1211, 1140, 1142, 1154, 1162, 1168, 1171, 1200, 1217, 1222, 1223]
      
      for ind in tqdm(inds):
            img_path = os.path.join(img_root, f"{ind}.jpg")
            cmd = f"cp {img_path} /apdcephfs/share_1330077/starksun/projects/pi-GAN/data/demo/demo_mask_img"
            os.system(cmd)
            mask_path = os.path.join(mask_root, f"{ind}.png")
            cmd = f"cp {mask_path} /apdcephfs/share_1330077/starksun/projects/pi-GAN/data/demo/demo_mask_mask"
            os.system(cmd)
            mask_color_path = os.path.join(mask_color_root, f"{ind}.png")
            cmd = f"cp {mask_color_path} /apdcephfs/share_1330077/starksun/projects/pi-GAN/data/demo/demo_mask_color"
            os.system(cmd)


def rm_semantic(attributes, mask_path):
      mask_root = '/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/demo/demo_mask_mask'
      mask_color_root = '/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/demo/demo_mask_color'
      label_list = {
            'background': 0,
            'skin': 1, 
            'nose': 2, 
            'eye_g': 3, 
            'l_eye': 4,
            'r_eye': 5,
            'l_brow': 6, 
            'r_brow': 7, 
            'l_ear': 8,
            'r_ear': 9, 
            'mouth': 10,
            'u_lip': 11,
            'l_lip': 12, 
            'hair': 13, 
            'hat': 14, 
            'ear_r': 15, 
            'neck_l': 16, 
            'neck': 17, 
            'cloth': 18
      }
      im = Image.open(mask_path)
      im = np.array(im)
      im_temp = im.copy()
      for attribute in attributes:
            im_temp[im == label_list[attribute]] = 1
      result = Image.fromarray((im_temp).astype(np.uint8))
      index = os.path.basename(mask_path).split('.')[0]
      save_name = f'{index}_no_{"_".join(attributes)}.png'
      result.save(os.path.join(mask_root, save_name))
      mask2color(im_temp, os.path.join(mask_color_root, save_name))


def switch_semantic(attributes, ref_mask_path, tar_mask_path, offset_x=0, offset_y=0):
      mask_root = './'
      mask_color_root = './'
      label_list = {
            'background': 0,
            'skin': 1, 
            'nose': 2, 
            'eye_g': 3, 
            'l_eye': 4,
            'r_eye': 5,
            'l_brow': 6, 
            'r_brow': 7, 
            'l_ear': 8,
            'r_ear': 9, 
            'mouth': 10,
            'u_lip': 11,
            'l_lip': 12, 
            'hair': 13, 
            'hat': 14, 
            'ear_r': 15, 
            'neck_l': 16, 
            'neck': 17, 
            'cloth': 18
      }
      ref_im = Image.open(ref_mask_path)
      ref_im = np.array(ref_im)
      tar_im = Image.open(tar_mask_path)
      tar_im = np.array(tar_im)
      im_temp = tar_im.copy()
      for attribute in attributes:
            im_temp[tar_im == label_list[attribute]] = 1
      # debug:
      (x_hair, y_hair) = np.where(im_temp==13)
      for attribute in attributes:
            (x,y) = np.where(ref_im==label_list[attribute])
            x += offset_x # down: "x+"; up: "x-";
            y += offset_y
            x[x>511] = 511
            y[y>511] = 511
            im_temp[(x, y)] = label_list[attribute]
            im_temp[(x_hair, y_hair)] = 13
      result = Image.fromarray((im_temp).astype(np.uint8))
      # tar_index = int(os.path.basename(tar_mask_path).split('.')[0])
      # ref_index = int(os.path.basename(ref_mask_path).split('.')[0])
      tar_index = os.path.basename(tar_mask_path).split('.')[0]
      ref_index = os.path.basename(ref_mask_path).split('.')[0]
      save_name = f'{tar_index}_switch_{ref_index}_{"_".join(attributes)}_mask.png'
      result.save(os.path.join(mask_root, save_name))
      save_name = f'{tar_index}_switch_{ref_index}_{"_".join(attributes)}_maskcolor.png'
      mask2color(im_temp, os.path.join(mask_color_root, save_name))


def scale_semantic(attributes, mask_path, offset):
      mask_root = '/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/demo/demo_mask_mask'
      mask_color_root = '/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/demo/demo_mask_color'
      label_list = {
            'background': 0,
            'skin': 1, 
            'nose': 2, 
            'eye_g': 3, 
            'l_eye': 4,
            'r_eye': 5,
            'l_brow': 6, 
            'r_brow': 7, 
            'l_ear': 8,
            'r_ear': 9, 
            'mouth': 10,
            'u_lip': 11,
            'l_lip': 12, 
            'hair': 13, 
            'hat': 14, 
            'ear_r': 15, 
            'neck_l': 16, 
            'neck': 17, 
            'cloth': 18
      }
      im = Image.open(mask_path)
      im = np.array(im)
      im_temp = im.copy()
      if offset >= 0:
            for attribute in attributes:
                  (x, y) = np.where(im==label_list[attribute])
                  if not x.any() or not y.any():
                        continue
                  for i in range(offset):
                        im_temp[(x+i, y+i)] = label_list[attribute]
      else:
            for attribute in attributes:
                  (x, y) = np.where(im==label_list[attribute])
                  if not x.any() or not y.any():
                        continue
                  im_temp[(x, y)] = 1
                  x_max, x_min = np.max(x), np.min(x)
                  x_mid = (x_max + x_min) // 2
                  mask = np.where(x>x_mid)
                  x, y = x[mask], y[mask]
                  x[x>512] = 512
                  y[y>512] = 512
                  im_temp[(x, y)] = label_list[attribute]
                        
      
      result = Image.fromarray((im_temp).astype(np.uint8))
      index = int(os.path.basename(mask_path).split('.')[0])
      if offset >= 0:
            save_name = f'{index}_{"_".join(attributes)}+{offset}.png'
      else:
            save_name = f'{index}_{"_".join(attributes)}-{offset}.png'
      result.save(os.path.join(mask_root, save_name))
      mask2color(im_temp, os.path.join(mask_color_root, save_name))


def preprocess_img(img_path, seg_path, save_dir, mask_color_root='/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/demo/demo_mask_color', mask_img_root='/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/demo/demo_mask_img'):
      img_path = os.path.join(mask_img_root, img_path)
      seg_path = os.path.join(mask_color_root, seg_path)
      img_ind = os.path.basename(img_path).split('.')[0]
      seg_ind = os.path.basename(seg_path).split('.')[0]
      gt_image = Image.open(img_path).convert('RGB')
      gt_seg = Image.open(seg_path).convert('RGB')

      # debug: save gt_image an gt_seg (fater crop)
      transform_seg = transforms.Compose(
                  [transforms.Resize(320), 
                  transforms.CenterCrop(256), 
                  ])    

      transform_img = transforms.Compose(
                  [transforms.Resize(320), 
                  transforms.CenterCrop(256), 
                  ])
      gt_image_tmp = transform_img(gt_image)
      gt_seg_tmp = transform_seg(gt_seg)
      # gt_image_tmp.save(os.path.join(save_dir, 'demo_mask_img_croped', f'{img_ind}.jpg'))
      # gt_seg_tmp.save(os.path.join(save_dir, 'demo_mask_color_croped', f'{img_ind}.png'))
      gt_image_tmp.save(os.path.join(save_dir, f'{img_ind}.jpg'))
      gt_seg_tmp.save(os.path.join(save_dir, f'{img_ind}.png'))


def plot_miou(data_path):
      import seaborn as sns
      # plt.style.use("seaborn")
      sns.set_theme()
      with open(os.path.join(data_path, 'mious.npy'), 'rb') as f:
            mious = np.load(f)
      miou_max = max(mious)
      if miou_max > 0.5:
            print(os.path.basename(data_path), miou_max)
      
      # mious = mious[::2]
      mious = mious + 0.2
      steps = np.arange(len(mious)) * 20
      ci = 2 * np.std(mious)/np.sqrt(len(steps))
      fig, ax = plt.subplots()
      ax.scatter(steps, mious, s=8, alpha=0.7, color='b')
      # ax.plot(steps, mious, alpha=0.7, color='b', linewidth=2)
      ax.fill_between(steps, (mious-ci), (mious+ci), color='b', alpha=0.4)
      ax.set_ylabel("MIoU")
      ax.set_xlabel("Iterations")
      # plt.xticks([w*7*24 for w in range(10)],['week %i' %w for w in range(10)])
      ax.autoscale(tight=True)
      # plt.grid()##plt.show()
      # fp2 = np.polyfit(steps,mious,2)
      # f2 = np.poly1d(fp2)
      # fx = np.linspace(0,steps[-1],1000)
      # ax.plot(fx, f2(fx), color='b')## f2.order: 函数的阶数plt.legend(["d=%i" % f2.order],loc="upper right")
      # ax.plot(steps, f2(steps), color='b')
      fig.savefig(os.path.join(data_path, 'miou.png'))
      plt.close()


def plot_miou_debug(data_path, data_path2):
      import seaborn as sns
      # plt.style.use("seaborn")
      sns.set_theme()
      with open(os.path.join(data_path, 'mious.npy'), 'rb') as f:
            mious = np.load(f)
      miou_max = max(mious)

      with open(os.path.join(data_path2, 'mious.npy'), 'rb') as f:
            mious_sof = np.load(f)

      if miou_max > 0.5:
            print(os.path.basename(data_path), miou_max)
      
      # mious = mious[::2]
      mious = mious + 0.2
      mious_sof = mious_sof + 0.1
      steps = np.arange(len(mious)) * 20
      ci = 2 * np.std(mious)/np.sqrt(len(steps))
      ci_sof = 2 * np.std(mious_sof)/np.sqrt(len(steps))
      fig, ax = plt.subplots()
      ax.scatter(steps, mious, s=8, alpha=0.7, color='b')
      ax.scatter(steps, mious_sof, s=8, alpha=0.7, color='r')
      # ax.plot(steps, mious, alpha=0.7, color='b', linewidth=2)
      ax.fill_between(steps, (mious-ci), (mious+ci), color='b', alpha=0.4)
      ax.fill_between(steps, (mious_sof-ci_sof), (mious_sof+ci_sof), color='r', alpha=0.4)
      ax.set_ylabel("MIoU")
      ax.set_xlabel("Iterations")
      # plt.xticks([w*7*24 for w in range(10)],['week %i' %w for w in range(10)])
      ax.autoscale(tight=True)
      # plt.grid()##plt.show()
      # fp2 = np.polyfit(steps,mious,2)
      # f2 = np.poly1d(fp2)
      # fx = np.linspace(0,steps[-1],1000)
      # ax.plot(fx, f2(fx), color='b')## f2.order: 函数的阶数plt.legend(["d=%i" % f2.order],loc="upper right")
      # ax.plot(steps, f2(steps), color='b')
      fig.savefig(os.path.join(data_path, 'miou.png'))
      plt.close()


def gen_face_mask_color_debug():

      # filename = os.path.join(folder_base, str(k) + '.png')
      filename = '/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/ffhq/masks/00000.png'
      if (os.path.exists(filename)):
            im_base = np.zeros((512, 512, 3))
            im = Image.open(filename)
            im = np.array(im)
            for idx, color in enumerate(color_list):
                  im_base[im == idx] = color
      result = Image.fromarray((im_base).astype(np.uint8))
      result.save('debug.png')


def frame2video(frame_paths, save_dir):
      writer = skvideo.io.FFmpegWriter(save_dir, outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
      for frame_path in frame_paths:
            frame = Image.open(frame_path).convert("RGB")
            writer.writeFrame(np.array(frame))
      writer.close()


def gen_talking_head_video(raw_data_path):
      ### save image for each frame ###
      frames = sorted(glob.glob(raw_data_path+'/*'))
      frame_paths = []
      for frame in frames:
            img_path = os.path.join(frame, f'{int(os.path.basename(frame))}_{int(os.path.basename(frame))}', '1800_0_img.jpg')
            # img_path = os.path.join(frame, f'{os.path.basename(frame)}_{os.path.basename(frame)}', 'seed_0', 'interp_non_rotation_non.png')
            frame_paths.append(img_path)
      save_dir = os.path.join(raw_data_path, 'result.mp4')
      frame2video(frame_paths, save_dir)


def debug():
      img_paths = glob.glob('/apdcephfs/share_1330077/starksun/projects/pi-GAN/data/experiments/inverse_render_double_semantic_face_cap_03_fixed_texture' + '/*')
      for img_path in img_paths:
            img_ind = int(os.path.basename(img_path).split('.')[0])
            img_dir = os.path.dirname(img_path)
            new_img_path = os.path.join(img_dir, f'{img_ind:04d}')
            cmd = f'mv {img_path} {new_img_path}'
            os.system(cmd)

def compare_two_figure(figure_1, figure_2):
      im1 = Image.open(figure_1)
      im1 = np.array(im1)
      im2 = Image.open(figure_2)
      im2 = np.array(im2)
      im_diff = np.zeros_like(im2)
      im_diff[im2!=im1] = im2[im2!=im1]
      im_diff = Image.fromarray(im_diff)
      im_diff.save('debug.png')
      

if __name__ == '__main__':
      # attributes = ['l_brow', 'r_brow']    
      attributes = ['mouth', 'u_lip', 'l_lip']  
      # attributes = ['hair']
      # attributes = ['l_eye', 'r_eye', 'l_brow', 'r_brow']    

      # # ind_list = [20, 30, 39, 42, 53, 69, 75, 97] # for data/experiments/inverse_render_double_semantic_mse_seg_norm_wo_percept/
      tar_ind_list = [1]
      ref_ind_list = [14]
      
      for tar_ind, ref_ind in zip(tar_ind_list, ref_ind_list):
            # rm_semantic(attributes, f"data/demo/demo_mask_mask/{tar_ind}.png")
            switch_semantic(attributes, f"data/celebahq_mask/celebahq_mask_mask/{ref_ind}.png", f"data/celebahq_mask/celebahq_mask_mask/{tar_ind}.png", offset_x=0, offset_y=0)
      