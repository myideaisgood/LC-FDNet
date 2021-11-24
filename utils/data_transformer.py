import torch
import torch.nn.functional as F

import numpy as np
import random

def RGB2YUV(img):

    img = img.astype(np.float32)

    r,g,b = np.split(img, 3, axis=2)

    u = b - np.round((87*r + 169*g) / 256.0)
    v = r - g
    y = g + np.round((86*v + 29*u) / 256.0)

    yuv_img = np.concatenate([y,u,v], axis=2)

    return yuv_img

def YUV2RGB(img):
    
    img = img.astype(np.float32)

    y,u,v = np.split(img, 3, axis=2)

    g = y - np.round((86*v + 29*u)/ 256.0)
    r = v + g
    b = u + np.round((87*r + 169*g)/256.0)

    rgb_img = np.concatenate([r,g,b], axis=2)

    return (rgb_img).astype(np.uint8)

def pad_img(img):

    C, H, W = img.shape

    pad_h = (H % 2)
    pad_w = (W % 2)
    padding = (0,pad_w,0,pad_h)

    pad_rep = torch.nn.ReplicationPad2d(padding)
    img = pad_rep(img.unsqueeze(0))
    img = img[0]
    
    return img, (pad_h, pad_w)


def space_to_depth_tensor(img, BLOCK_SIZE=2):

    C, H, W = img.shape

    H_ = H // BLOCK_SIZE
    W_ = W // BLOCK_SIZE

    img = torch.reshape(img, (C, H_, BLOCK_SIZE, W_, BLOCK_SIZE))
    img = img.permute(0,2,4,1,3)
    img = torch.reshape(img, (C, BLOCK_SIZE*BLOCK_SIZE, H_, W_))

    return img

def depth_to_space_tensor(img, BLOCK_SIZE=2):

    B, C, D, H, W = img.shape
    
    H_ = H * BLOCK_SIZE
    W_ = W * BLOCK_SIZE

    img = torch.reshape(img, (B, C, BLOCK_SIZE, BLOCK_SIZE, H, W))
    img = img.permute(0,1,4,2,5,3)
    img = torch.reshape(img, (B, C, H_, W_))

    return img

def space_to_depth(img, BLOCK_SIZE=2):

    H, W, C = img.shape

    H_ = H // BLOCK_SIZE
    W_ = W // BLOCK_SIZE

    img_depth = img.reshape(H_, BLOCK_SIZE, W_, BLOCK_SIZE, C)
    img_depth = np.transpose(img_depth, (1,3,0,2,4))
    img_depth = img_depth.reshape(BLOCK_SIZE*BLOCK_SIZE, H_, W_, C)

    return img_depth

def depth_to_space(img, BLOCK_SIZE=2):

    D, H, W, C = img.shape
    
    H_ = H * BLOCK_SIZE
    W_ = W * BLOCK_SIZE

    img = img.reshape(BLOCK_SIZE, BLOCK_SIZE, H, W, C)
    img = np.transpose(img, (2,0,3,1,4))
    img_space = img.reshape(H_, W_, C)

    return img_space

def tensor2image(img):

    img = img.permute(0,2,3,1)
    img = img.cpu().numpy()

    return img

def RandomCrop(img, CROP_SIZE):

    H, W, _ = img.shape

    h_idx = random.randint(0, H-CROP_SIZE-1)
    w_idx = random.randint(0, W-CROP_SIZE-1)

    img = img[h_idx:h_idx+CROP_SIZE, w_idx:w_idx+CROP_SIZE]

    return img