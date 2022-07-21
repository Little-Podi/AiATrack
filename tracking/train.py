import argparse
import os
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    """
    Args for training.
    """

    parser = argparse.ArgumentParser(description='parse args for training')
    # For train
    parser.add_argument('--script', type=str, default='aiatrack', help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save', type=str, default='.',
                        help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], default='single',
                        help='train on single gpu or multiple gpus')
    parser.add_argument('--nproc', type=int, help='number of GPUs per node')  # Specify when mode is multiple

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == 'single':
        train_cmd = 'python lib/train/run_training.py --script %s --config %s --save_dir %s ' \
                    % (args.script, args.config, args.save)
    elif args.mode == 'multiple':
        train_cmd = 'python -m torch.distributed.launch --nproc_per_node %d ' \
                    '--master_addr 127.0.0.2 --master_port 6666 lib/train/run_training.py ' \
                    '--script %s --config %s --save_dir %s' \
                    % (args.nproc, args.script, args.config, args.save)
    else:
        raise ValueError("ERROR: mode should be 'single' or 'multiple'")
    os.system(train_cmd)


if __name__ == '__main__':
    main()
