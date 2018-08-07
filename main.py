import argparse
import logging
import numpy as np
import random
import os

from run_model import run

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, required=True,
        help='Path to dataset')
    parser.add_argument('-e', '--epochs', type=int, default=10,
        help='Number of passes through the data for training the model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
        help='Float value for network learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=128,
        help='Amount of data passed to the network in each batch')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
        help='Float value multiplier on regularization term')
    parser.add_argument('-rs', '--random_seed', type=int, default=1234,
        help='Random number seed')
    parser.add_argument('-v', '--val', type=float, default=0.1,
        help='Ratio of data in validation set')
    parser.add_argument('-p', '--patience', type=int, default=10,
        help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('-g', '--gpus', type=str, nargs='*',
        help='Index of GPU to use for training')
    parser.add_argument('-sd', '--save_dir', type=str, default='./',
        help='Directory to save model')
    parser.add_argument('-lf', '--log_file', type=str, default='./log.txt',
        help='Path to log file created for training')
    return parser.parse_args()

def main():
    params = get_opts()

    if params.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(params.gpus)

    np.random.seed(params.random_seed)
    random.seed(params.random_seed)

    logging.basicConfig(filename=params.log_file, level=logging.DEBUG)

    run(params)

if __name__ == '__main__':
    main()