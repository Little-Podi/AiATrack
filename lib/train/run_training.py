import argparse
import importlib
import os
import random
import warnings

warnings.filterwarnings('ignore')

import cv2 as cv
import numpy as np
import torch.backends.cudnn
import torch.distributed as dist

torch.backends.cudnn.benchmark = False

import _init_paths
import lib.train.admin.settings as ws_settings


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None, base_seed=None):
    """
    Run the train script.

    Args:
        script_name: Name of experiment in the 'experiments/' folder.
        config_name: Name of the yaml file in the 'experiments/<script_name>'.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    if save_dir is None:
        print('save_dir dir is not given, use the default dir instead')
    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    # Set seed for different process
    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = 'train/{}/{}'.format(script_name, config_name)
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))

    expr_module = importlib.import_module('lib.train.train_script')
    expr_func = getattr(expr_module, 'run')

    # if settings.local_rank in [-1, 0]:
    #     print('script_name: {}.py  config_name: {}.yaml'.format(script_name, config_name))

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='run a train scripts in train_settings')
    parser.add_argument('--script', type=str, default='aiatrack', help='name of the train script')
    parser.add_argument('--config', type=str, default='baseline', help='name of the config file')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='set cudnn benchmark on (1) or off (0) (default is on)')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    parser.add_argument('--save_dir', type=str, default='.', help='the directory to save checkpoints and logs')
    parser.add_argument('--seed', type=int, default=32, help='seed for random numbers')
    args = parser.parse_args()
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark,
                 local_rank=args.local_rank, save_dir=args.save_dir, base_seed=args.seed)


if __name__ == '__main__':
    main()
