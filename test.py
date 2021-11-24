from torch.utils.data import DataLoader

import logging
import os
import sys

from config import parse_args
from utils.average_meter import AverageMeter
from utils.data_loaders import Dataset
from utils.log_helpers import *
from utils.helpers import *

from utils.data_transformer import *

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
DATASET = args.dataset
EXP_NAME = args.experiment_name
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
WEIGHTS = args.weights

# Set up logger
filename = os.path.join(LOG_DIR, 'logs_inference.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

######### Configuration #########
######### Configuration #########
######### Configuration #########

# Set up Dataset
test_dataset = Dataset(args, 'test')

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=2,
    shuffle=False
)

# JPEG-XL result of img a
jpegxl_bpp, jpegxl_avg_bpp, jpegxl_avg_time = get_jpegxl_result(test_dataloader)

############################
# Encode : yd - ud - vd
#       => yb - ub - vb
#       => yc - uc - vc
############################

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

with torch.no_grad():
    # Metric Holders
    bitrates = get_AverageMeter(color_names, loc_names)
    bitrates['total'] = AverageMeter()

    enc_times = {}
    for loc in loc_names:
        enc_times[loc] = AverageMeter()

    # Change networks to evaluation mode
    network_set(networks, color_names, loc_names, set='eval')

    # Evaluate for test data
    for batch_idx, data in enumerate(test_dataloader):

        img_a, img_b, img_c, img_d, ori_img, img_name, padding = data

        _, H, W, _ = ori_img.shape

        # Data to cuda
        [img_a, img_b, img_c, img_d] = img2cuda([img_a, img_b, img_c, img_d], device)
        imgs = abcd_unite(img_a, img_b, img_c, img_d, color_names)

        # Inputs / Ref imgs / GTs
        inputs = get_inputs(imgs, color_names, loc_names)
        ref_imgs = get_refs(imgs, color_names)
        gt_imgs = get_gts(imgs, color_names, loc_names)

        for loc in loc_names:

            start_time = time()

            for color in color_names:
                # Feed to network
                cur_network = networks[color][loc]
                cur_inputs = inputs[color][loc]
                cur_gt_img = gt_imgs[color][loc]
                cur_ref_img = ref_imgs[color]

                # Low Frequency Compressor
                _, q_res_L, error_var_map, error_var_th, mask_L, pmf_softmax_L = cur_network(cur_inputs, cur_gt_img, cur_ref_img, frequency='low', mode='eval')
                mask_H = 1-mask_L

                bits_L = estimate_bits(sym=q_res_L, pmf=pmf_softmax_L, mask=mask_L)

                if EMPTY_CACHE:
                    del q_res_L, pmf_softmax_L
                    torch.cuda.empty_cache()

                gt_L = mask_L * cur_gt_img
                input_H = torch.cat([cur_inputs, gt_L], dim=1)

                _, q_res_H, pmf_softmax_H = cur_network(input_H, cur_gt_img, cur_ref_img, frequency='high', mode='eval')

                bits_H = estimate_bits(sym=q_res_H, pmf=pmf_softmax_H, mask=mask_H)
                bits = bits_L.item() + bits_H.item()
                bitrate = bits / (H*W)

                bitrates[color][loc].update(bitrate)

                if EMPTY_CACHE:
                    del q_res_H, pmf_softmax_H
                    torch.cuda.empty_cache()

            enc_time = time() - start_time
            enc_times[loc].update(enc_time)

        update_total(bitrates, color_names, loc_names)

        # Print Test Img Results
        log_img_info(logging, img_name, bitrates, jpegxl_bpp[batch_idx], color_names, loc_names)

    # Print Test Dataset Results
    log_dataset_info(logging, bitrates, jpegxl_avg_bpp, enc_times, jpegxl_avg_time, color_names, loc_names)