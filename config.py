import argparse

def parse_training_args(parser):
    """Add args used for training only.

    Args:
        parser: An argparse object.
    """
    # Design paramters
    parser.add_argument('--crop_size', type=int, default=128,
                        help='Size of input image')             

    parser.add_argument('--hidden_unit', type=int, default=64,
                        help='Number of hidden units in network')             

    parser.add_argument('--lambda_pred', type=float, default=1.0,
                    help='Balance Parameter of Prediction Loss')

    parser.add_argument('--lambda_ev', type=float, default=1.0,
                    help='Balance Parameter of Error Variance Loss')

    parser.add_argument('--lambda_br', type=float, default=1.0,
                    help='Balance Parameter of Bitrate Loss')

    # Session parameters
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='GPU number to use')

    parser.add_argument('--batch_size', type=int, default=24,
                        help='Minibatch size')

    parser.add_argument('--epochs', type=int, default=20000,
                        help='Number of epochs to train')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    parser.add_argument('--decay_step', type=int, default=1000,
                        help='Decay Step')

    parser.add_argument('--decay_rate', type=int, default=0.1,
                        help='Decay Rate')

    parser.add_argument('--print_every', type=int, default=1,
                        help='How many iterations print for loss evaluation')

    parser.add_argument('--save_every', type=int, default=20,
                        help='How many iterations to save')                        

    parser.add_argument('--empty_cache', type=str2bool, default=True,
                        help='Empty cache for efficient memory allocation (speed down)')

    # Directory parameters
    parser.add_argument('--data_dir', type=str, default="dataset/",
                        help='dataset directory')

    parser.add_argument('--dataset', type=str, default="div2k/",
                        help='name of dataset : play, clic_m, clic_p, div2k')

    parser.add_argument('--experiment_name', type=str, default='default/',
                        help='Experiment Name directory')

    parser.add_argument('--ckpt_dir', type=str, default="ckpt/",
                        help='checkpoint directory')

    parser.add_argument('--log_dir', type=str, default="log/",
                        help='log directory')

    parser.add_argument('--weights', type=str, default="ckpt.pth",
                        help='Fill In Fill In')

    parser.add_argument('--encode_dir', type=str, default="encode_results/",
                        help='Directory of where encoded results are stored')

    parser.add_argument('--decode_dir', type=str, default="decode_results/",
                        help='Directory of where decoded images are stored')

def parse_args():
    """Initializes a parser and reads the command line parameters.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """

    parser = argparse.ArgumentParser(description='UNet')
    parse_training_args(parser)

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    args = args.__dict__
    print("Arguments:")

    for key, value in sorted(args.items()):
        print('\t%15s:\t%s' % (key, value))
