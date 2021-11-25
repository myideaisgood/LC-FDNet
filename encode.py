import torch

import logging
import os
import sys
from time import time

from config import parse_args
from utils.average_meter import AverageMeter
from utils.log_helpers import *
from utils.helpers import *

######### Configuration #########
######### Configuration #########
######### Configuration #########
args = parse_args()

# Design Parameters
HIDDEN_UNIT = args.hidden_unit

# Session Parameters
GPU_NUM = args.gpu_num
EMPTY_CACHE = args.empty_cache

# Directory Parameters
DATA_DIR = args.data_dir
DATASET = args.dataset
EXP_NAME = args.experiment_name
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
WEIGHTS = args.weights
ENC_DIR = args.encode_dir

# Check if directory does not exist
os.system('rm -rf "%s"' % (ENC_DIR))
create_path(ENC_DIR)

# Set up logger
filename = os.path.join(LOG_DIR, 'logs_encode.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

######### Configuration #########
######### Configuration #########
######### Configuration #########

color_names = ['Y','U','V']
loc_names = ['d', 'b', 'c']

# Set up networks
networks = setup_networks(color_names, loc_names, logging, HIDDEN_UNIT)

# Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the network to GPU if possible
if torch.cuda.is_available():
    network2cuda(networks, device, color_names, loc_names)

# Load the pretrained model if exists
if os.path.exists(os.path.join(CKPT_DIR, WEIGHTS)):
    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, WEIGHTS))

    for color in color_names:
        path_name = 'network_' + color + '_' + WEIGHTS
        checkpoint = torch.load(os.path.join(CKPT_DIR, path_name))
        for loc in loc_names:
            networks[color][loc].load_state_dict(checkpoint['network_' + color + '_' + loc])
    logging.info('Recover completed.')
else:
    logging.info('No model to load')
    sys.exit(1)

# Inference Current Model
# Metric Holders
bitrates = get_AverageMeter(color_names, loc_names)
bitrates['total'] = AverageMeter()

enc_times = {}
for loc in loc_names:
    enc_times[loc] = AverageMeter()

# Change networks to evaluation mode
network_set(networks, color_names, loc_names, set='eval')

# Read in image names
img_names = os.listdir(os.path.join(DATA_DIR, DATASET, 'test'))
img_names = sorted(img_names)

# JPEGXL metrics
jpegxl_avg_bpp = 0
jpegxl_avg_time = 0

from utils.enc_dec import *

with torch.no_grad():
    for img_name in img_names:

        # Read in image
        img_a, img_b, img_c, img_d, ori_img, img_name, padding = read_img(os.path.join(DATA_DIR, DATASET, 'test', img_name))
        H, W, _ = ori_img.shape

        # Modify image name lena.png -> lena
        img_name = modify_imgname(img_name)
        img_name_wo_space = img_name.replace(" ","")

        # Create directory to save compressed file
        create_path(os.path.join(ENC_DIR, img_name_wo_space))

        # Encode padding
        encode_padding(padding, img_name_wo_space, ENC_DIR)

        # Encode img_a by jpegxl
        jpegxl_bpp, jpegxl_time = encode_jpegxl(img_a, img_name, img_name_wo_space, H, W, ENC_DIR)
        
        jpegxl_avg_bpp += jpegxl_bpp
        jpegxl_avg_time += jpegxl_time

        # Data to cuda
        [img_a, img_b, img_c, img_d] = img2cuda([img_a, img_b, img_c, img_d], device)
        imgs = abcd_unite(img_a, img_b, img_c, img_d, color_names)

        # Inputs / Ref imgs / GTs
        inputs = get_inputs(imgs, color_names, loc_names)
        ref_imgs = get_refs(imgs, color_names)
        gt_imgs = get_gts(imgs, color_names, loc_names)

        # Encode img_b, img_c, img_d
        for loc in loc_names:

            start_time = time()

            for color in color_names:
                cur_network = networks[color][loc]
                cur_inputs = inputs[color][loc]
                cur_gt_img = gt_imgs[color][loc]
                cur_ref_img = ref_imgs[color]

                # Feed to network
                _, q_res_L, error_var_map, error_var_th, mask_L, pmf_softmax_L = cur_network(cur_inputs, cur_gt_img, cur_ref_img, frequency='low', mode='eval')
                mask_H = 1-mask_L

                gt_L = mask_L * cur_gt_img
                input_H = torch.cat([cur_inputs, gt_L], dim=1)

                _, q_res_H, pmf_softmax_H = cur_network(input_H, cur_gt_img, cur_ref_img, frequency='high', mode='eval')

                # Encode by torchac
                bpp_L = encode_torchac(pmf_softmax_L, q_res_L, mask_L, img_name_wo_space, color, loc, H, W, ENC_DIR, EMPTY_CACHE, frequency='low')
                bpp_H = encode_torchac(pmf_softmax_H, q_res_H, mask_H, img_name_wo_space, color, loc, H, W, ENC_DIR, EMPTY_CACHE, frequency='high')

                bpp = bpp_L + bpp_H

                bitrates[color][loc].update(bpp)

                if EMPTY_CACHE:
                    del q_res_L, pmf_softmax_L, q_res_H, pmf_softmax_H
                    torch.cuda.empty_cache()
                
            enc_time = time() - start_time
            enc_times[loc].update(enc_time)

        update_total(bitrates, color_names, loc_names)

        # Print Test Img Results
        log_img_info(logging, img_name, bitrates, jpegxl_bpp, color_names, loc_names)

    jpegxl_avg_bpp /= len(img_names)
    jpegxl_avg_time /= len(img_names)

    log_dataset_info(logging, bitrates, jpegxl_avg_bpp, enc_times, jpegxl_avg_time, color_names, loc_names, 'Avg')