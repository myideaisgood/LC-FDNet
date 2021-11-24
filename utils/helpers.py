import torch

import numpy as np
import cv2
import os
import logging
from time import time

from model import FDNet
from utils.average_meter import AverageMeter
from utils.data_transformer import *   

def var_or_cuda(x, device=None):
    x = x.contiguous()
    if torch.cuda.is_available() and device != torch.device('cpu'):
        if device is None:
            x = x.cuda(non_blocking=True)
        else:
            x = x.cuda(device=device, non_blocking=True)

    return x

def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

def count_parameters(network):
    return sum(p.numel() for p in network.parameters())

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def setup_networks(color_names, loc_names, logging, HIDDEN_UNIT):

    networks = {}

    for color in color_names:
        networks[color] = {}

    idx = 2

    for loc in loc_names:
        for color in color_names:
            networks[color][loc] = FDNet(color=color, indim=idx+1, hu=HIDDEN_UNIT)
            logging.info('Parameters in Network %s %s : %d.' % (color, loc, count_parameters(networks[color][loc])))
            idx += 1
    return networks

def setup_optimizers(networks, color_names, loc_names, LR):

    optimizers = {}

    for color in color_names:
        optimizer_color = {}
        for loc in loc_names:
            optimizer_color[loc] = torch.optim.Adam(filter(lambda p: p.requires_grad, networks[color][loc].parameters()), lr=LR)
        optimizers[color] = optimizer_color    

    return optimizers

def setup_schedulers(optimizers, color_names, loc_names, DECAY_STEP, DECAY_RATE):

    schedulers = {}
    for color in color_names:
        scheduler_color = {}
        for loc in loc_names:
            scheduler_color[loc] = torch.optim.lr_scheduler.StepLR(optimizers[color][loc], step_size=DECAY_STEP, gamma=DECAY_RATE)
        schedulers[color] = scheduler_color

    return schedulers

def schedulers_step(schedulers, color_names, loc_names):

    for color in color_names:
        for loc in loc_names:
            schedulers[color][loc].step()

def network_set(networks, color_names, loc_names, set='train'):

    for color in color_names:
        for loc in loc_names:
            if set=='train':
                networks[color][loc].train()
            else:
                networks[color][loc].eval()

def network2cuda(networks, device, color_names, loc_names):

    for color in color_names:
        for loc in loc_names:
            networks[color][loc].to(device)


def img2cuda(imgs, device):

    for idx, img in enumerate(imgs):
        imgs[idx] = var_or_cuda(img, device=device)
    
    return imgs

def get_AverageMeter(color_names, loc_names):
    
    input = {}

    for color in color_names:
        input_color = {}
        for loc in loc_names:
            input_color[loc] = AverageMeter()
        input[color] = input_color
    
    return input  

def abcd_unite(img_a, img_b, img_c, img_d, color_names):

    imgs = {}

    for color in color_names:
        if color=='Y':
            idx = 0
        elif color=='U':
            idx = 1
        elif color=='V':
            idx = 2

        img_color = {}

        img_color['a'] = img_a[:,idx:idx+1]
        img_color['b'] = img_b[:,idx:idx+1]
        img_color['c'] = img_c[:,idx:idx+1]
        img_color['d'] = img_d[:,idx:idx+1]

        imgs[color] = img_color

    return imgs

# Get input to the network
# Network for 'd' : input is 'a'
# Network for 'b' : input is 'a' & 'd'
# Network for 'c' : input is 'a' & 'd' & 'b'
def get_inputs(imgs, color_names, loc_names):

    inputs = {}

    for color in color_names:
        inputs[color] = {}

    gt_a = torch.cat([imgs['Y']['a'], imgs['U']['a'], imgs['V']['a']], dim=1)
    gt_d = torch.cat([imgs['Y']['d'], imgs['U']['d'], imgs['V']['d']], dim=1) 
    gt_b = torch.cat([imgs['Y']['b'], imgs['U']['b'], imgs['V']['b']], dim=1) 

    for loc in loc_names:
        
        if loc == 'd':
            prev_loc = gt_a
        elif loc == 'b':
            prev_loc = torch.cat([gt_a, gt_d], dim=1)
        elif loc == 'c':
            prev_loc = torch.cat([gt_a, gt_d, gt_b], dim=1)
        
        inputs['Y'][loc] = prev_loc
        inputs['U'][loc] = torch.cat([prev_loc, imgs['Y'][loc]], dim=1)
        inputs['V'][loc] = torch.cat([prev_loc, imgs['Y'][loc], imgs['U'][loc]], dim=1)

    return inputs


# Get reference to the network
# Reference is 'a' for every network
def get_refs(imgs, color_names):

    ref_imgs = {}

    for color in color_names:
        ref_imgs[color] = imgs[color]['a']

    return ref_imgs

# Get GTs to the network
# Network for 'd' : GT is 'd'
# Network for 'b' : GT is 'b'
# Network for 'c' : GT is 'c'
def get_gts(imgs, color_names, loc_names):

    gt_imgs = {}

    for color in color_names:
        gt_color = {}
        for loc in loc_names:
            gt_color[loc] = imgs[color][loc]
        gt_imgs[color] = gt_color    

    return gt_imgs

def update_total(bitrates, color_names, loc_names):

    total_bit = 0
    for color in color_names:
        color_bit = 0
        for loc in loc_names:
            color_bit += bitrates[color][loc].val()
        total_bit += color_bit

    bitrates['total'].update(total_bit)

def estimate_bits(sym, pmf, mask):

    pmf = pmf.permute(0,2,3,1)
    L = pmf.shape[-1]
    pmf = pmf.reshape(-1, L)
    sym = sym.reshape(-1, 1)
    assert pmf.shape[0] == sym.shape[0]

    encode_idx = (mask.flatten()==1).nonzero(as_tuple=False)

    pmf = pmf[encode_idx][:,0,:]
    sym = sym[encode_idx][:,:,0]

    relevant_probs = torch.gather(pmf, dim=1, index=sym)
    bits = torch.sum(-torch.log2(relevant_probs.clamp(min=1e-5)))    

    return bits

def save_bitrate_img(sym, pmf, loc, img_name, dir):

    B, C, H, W = sym.shape

    pmf = pmf.permute(0,2,3,1)
    L = pmf.shape[-1]
    pmf = pmf.reshape(-1, L)
    sym = sym.reshape(-1, 1)
    assert pmf.shape[0] == sym.shape[0]
    relevant_probs = -torch.log2((torch.gather(pmf, dim=1, index=sym)).clamp(min=1e-5))

    bitrate = torch.reshape(relevant_probs, (B,C,H,W))
    bitrate = torch.ceil(10*bitrate)[0,0]
    bitrate = bitrate.clamp(min=0, max=255)

    bitrate = bitrate.detach().cpu().numpy()
    save_img = bitrate.astype(np.uint8)

    save_name = dir + img_name[0] + '_' + loc + '.jpg'

    cv2.imwrite(save_name, save_img)

def save_bitrate_img_slice(sym_list, pmf_list, loc, img_name, dir):

    _, _, H_lu, W_lu = sym_list[0].shape
    _, _, H_ru, W_ru = sym_list[1].shape
    _, _, H_ld, W_ld = sym_list[2].shape
    _, _, H_rd, W_rd = sym_list[3].shape

    H_total = H_lu + H_ld
    W_total = W_lu + W_ru

    save_img = np.zeros((H_total, W_total)).astype(np.uint8)

    for idx in range(len(sym_list)):
        B, C, H, W = sym_list[idx].shape

        pmf = pmf_list[idx].permute(0,2,3,1)
        L = pmf.shape[-1]
        pmf = pmf.reshape(-1, L)
        sym = sym_list[idx].reshape(-1, 1)
        assert pmf.shape[0] == sym.shape[0]
        relevant_probs = -torch.log2((torch.gather(pmf, dim=1, index=sym)).clamp(min=1e-5))

        bitrate = torch.reshape(relevant_probs, (B,C,H,W))
        bitrate = torch.ceil(10*bitrate)[0,0]
        bitrate = bitrate.clamp(min=0, max=255)

        bitrate = bitrate.detach().cpu().numpy()
        bitrate = bitrate.astype(np.uint8)

        if idx == 0:
            save_img[:int(H_total/2), :int(W_total/2)] = bitrate
        elif idx == 1:
            save_img[:int(H_total/2), int(W_total/2):] = bitrate
        elif idx == 2:
            save_img[int(H_total/2):, :int(W_total/2)] = bitrate
        elif idx == 3:
            save_img[int(H_total/2):, int(W_total/2):] = bitrate            

    save_name = dir + img_name[0] + '_' + loc + '.jpg'

    cv2.imwrite(save_name, save_img)    


def slice_img(img):

    B, C, H, W = img.size()

    img_lu = img[:,:,:int(H/2), :int(W/2)]
    img_ru = img[:,:,:int(H/2), int(W/2):]
    img_ld = img[:,:,int(H/2):, :int(W/2)]
    img_rd = img[:,:,int(H/2):, int(W/2):]

    slice_img = []
    slice_img.append(img_lu)
    slice_img.append(img_ru)
    slice_img.append(img_ld)
    slice_img.append(img_rd)

    return slice_img    

def get_jpegxl_result(test_dataloader):

    # JPEG-XL result of img a
    jpegxl_bpp = []
    jpegxl_avg_bpp = 0.0
    jpegxl_avg_time = 0.0

    for batch_idx, data in enumerate(test_dataloader):
        
        img_a, _, _, _, _, img_name, _ = data

        img_a = torch.unsqueeze(img_a, dim=2)

        img_a = (img_a.squeeze()).permute(1,2,0)
        img_a = img_a.numpy()
        img_a = YUV2RGB(img_a)
        h,w,c = img_a.shape

        img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)

        savename = img_name[0]

        cv2.imwrite(savename, img_a)

        start = time()
        os.system('jpegxl/build/tools/cjxl "%s" output.jxl -q 100' % (savename))
        end = time()

        filesize = os.stat('output.jxl').st_size

        bpp = 8*filesize / (4*h*w)
        jpegxl_avg_bpp += bpp
        jpegxl_avg_time += (end - start)
        
        jpegxl_bpp.append(bpp)

        logging.info("%s : %.4f" % (img_name, bpp))

        os.system('rm "%s"' % (savename))
        os.system('rm %s' % ('output.jxl'))

    jpegxl_avg_bpp /= len(test_dataloader)
    jpegxl_avg_time /= len(test_dataloader)

    logging.info("JPEGXL Average BPP : %.4f,      Average Time : %.4f" % (jpegxl_avg_bpp, jpegxl_avg_time))    

    return jpegxl_bpp, jpegxl_avg_bpp, jpegxl_avg_time