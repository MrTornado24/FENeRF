# prepare semantic maps for FFHQ
import os
import glob
import json
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, utils
from generators.BiSeNet import BiSeNet


remap_list_celebahq = torch.tensor([0, 1, 6, 7, 4, 5, 2, 2, 10, 11, 12, 8, 9, 15, 3, 17, 16, 18, 13, 14]).float()


remap_list = torch.tensor([0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16]).float()


def id_remap(seg, type='sof'):
    if type == 'sof':
        return remap_list[seg.long()].to(seg.device)
    elif type == 'celebahq':
        return remap_list_celebahq[seg.long()].to(seg.device)
    
    
def parsing_img(bisNet, image, to_tensor, argmax=True):
    with torch.no_grad():
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0).cuda()
        segmap = bisNet(img)[0]
        if argmax:
            segmap = segmap.argmax(1, keepdim=True)
        segmap = id_remap(segmap, 'celebahq')
    return img, segmap

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


def initFaceParsing(n_classes=20, path=None):
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(path+'/segNet-20Class.pth'))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
    return net, to_tensor


def vis_condition_img_celebahq(img):
    N,C,H,W = img.size()
    condition_img_color = torch.zeros((N,3,H,W))
    num_of_class = int(torch.max(img))
    for pi in range(1, num_of_class + 1):
        index = torch.nonzero(img == pi)
        condition_img_color[index[:,0],:,index[:,2], index[:,3]] = torch.tensor(COLOR_MAP[pi], dtype=torch.float)
    condition_img_color = condition_img_color/255*2.0-1.0
    return condition_img_color


def face_parsing(img_path, save_dir, bisNet):
    # img = Image.open(os.path.join(save_dir, 'images512x512', img_path)).convert('RGB')
    img = Image.open(img_path).convert('RGB')
    _, seg_label = parsing_img(bisNet, img.resize((512, 512)), to_tensor)
    seg_mask = seg_label.detach().cpu().numpy()[0][0]
    seg_mask = Image.fromarray(seg_mask.astype(np.uint8), mode="L")
    img_ind = os.path.basename(img_path)
    save_path = os.path.join(save_dir, 'masks1024x1024', img_ind)
    seg_mask.save(save_path)

    seg_label_rgb = vis_condition_img_celebahq(seg_label)
    save_path = os.path.join(save_dir, 'maskcolors1024x1024', img_ind)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    utils.save_image(seg_label_rgb, save_path, normalize=True,range=(-1, 1))

if __name__ == "__main__":
    ckpt_dir = 'checkpoints'
    img_paths = glob.glob('D:/datasets/FFHQ/images1024x1024/*.png')
    save_dir = './tmp'
    bisNet, to_tensor = initFaceParsing(path=ckpt_dir)
    
    for img_path in sorted(img_paths):
        face_parsing(img_path, save_dir, bisNet)
        
        

