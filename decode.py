import torch

import logging
import os
import sys
from time import time

from config import parse_args
from utils.average_meter import AverageMeter
from utils.log_helpers import *
from utils.helpers import *
from utils.quantizer import custom_round

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
DEC_DIR = args.decode_dir

# Check if directory does not exist
os.system('rm -rf "%s"' % (DEC_DIR))
create_path(DEC_DIR)

# Set up logger
filename = os.path.join(LOG_DIR, 'logs_decode.txt')
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
dec_times = AverageMeter()
dec_jpegxl_time = AverageMeter()

# Change networks to evaluation mode
network_set(networks, color_names, loc_names, set='eval')

# Read in image names
img_names = os.listdir(os.path.join(DATA_DIR, DATASET, 'test'))

# JPEGXL metrics
jpegxl_avg_bpp = 0
jpegxl_avg_time = 0

from utils.enc_dec import *

with torch.no_grad():
    for img_name in img_names:

        start_time = time()

        img_name = modify_imgname(img_name)
        img_a = decode_jpegxl(ENC_DIR, img_name)

        jpegxl_time = time()

        img_a = var_or_cuda(torch.unsqueeze(img_a, dim=0))
        _, _, H, W = img_a.shape

        imgs = abcd_unite(img_a, img_a, img_a, img_a, color_names)

        padding = decode_padding(ENC_DIR, img_name)
        pad_w, pad_h = padding[0], padding[1]

        for loc in loc_names:
            for color in color_names:
                # Obtain GT
                if color == 'Y':
                    sym_mean = 255
                else:
                    sym_mean = 510

                inputs = get_inputs(imgs, color_names, loc_names)
                ref_imgs = get_refs(imgs, color_names)
                gt_imgs = get_gts(imgs, color_names, loc_names)

                cur_network = networks[color][loc]
                cur_inputs = inputs[color][loc]
                cur_gt_img = gt_imgs[color][loc]
                cur_ref_img = ref_imgs[color]

                # Feed to network
                pred_L, q_res_L, error_var_map, error_var_th, mask_L, pmf_softmax_L = cur_network(cur_inputs, cur_gt_img, cur_ref_img, frequency='low', mode='eval')
                mask_H = 1-mask_L

                # Decode Low frequency region
                decoded_sym_L = decode_torchac(pmf_softmax_L, img_name, mask_L, color, loc, ENC_DIR, EMPTY_CACHE, frequency='low')
                decode_L = custom_round(pred_L) - decoded_sym_L + sym_mean
                input_H = torch.cat([cur_inputs, decode_L*mask_L], dim=1)

                pred_H, _, pmf_softmax_H = cur_network(input_H, cur_gt_img, cur_ref_img, frequency='high', mode='eval')

                decoded_sym_H = decode_torchac(pmf_softmax_H, img_name, mask_H, color, loc, ENC_DIR, EMPTY_CACHE, frequency='high')
                decode_H = custom_round(pred_H) - decoded_sym_H + sym_mean

                recon = mask_L * decode_L + mask_H * decode_H
                imgs[color][loc] = recon

                if EMPTY_CACHE:
                    del pred_L, pmf_softmax_L, pred_H, pmf_softmax_H
                    torch.cuda.empty_cache()

        output_img = abcd2img(imgs, color_names)

        H, W, _ = output_img.shape
        output_img = output_img[:H-pad_h,:W-pad_w]

        cv2.imwrite(DEC_DIR + '/' + img_name + '.png', output_img)

        end_time = time()

        out_string = '%s, Decode Time = %.4f = %.4f + %.4f'
        out_tuple = (img_name, end_time - start_time, jpegxl_time - start_time, end_time - jpegxl_time)

        logging.info(out_string % out_tuple)

        dec_times.update(end_time - jpegxl_time)
        dec_jpegxl_time.update(jpegxl_time - start_time)

    out_string = 'Average Decode Time = %.4f = %.4f + %.4f'
    out_tuple = (dec_times.avg() + dec_jpegxl_time.avg(), dec_jpegxl_time.avg() + dec_times.avg())        