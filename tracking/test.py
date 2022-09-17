import argparse
import os
import sys
import warnings

import torch

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

warnings.filterwarnings('ignore')

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8):
    """
    Run tracker on sequence or dataset.

    Args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run ID.
        dataset_name: Name of dataset (otb, nfs, uav, trackingnet, got_test, got_val, lasot, lasot_ext).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='run tracker on sequence or dataset')
    parser.add_argument('--tracker', type=str, default='aiatrack', help='name of tracking method')
    parser.add_argument('--param', type=str, default='baseline', help='name of config file')
    parser.add_argument('--id', type=int, default=None, help='the run ID')
    parser.add_argument('--dataset', type=str, default='lasot',
                        help='name of dataset (otb, nfs, uav, trackingnet, got_test, got_val, lasot, lasot_ext)')
    parser.add_argument('--seq', type=str, default=None, help='sequence number or name')
    parser.add_argument('--debug', type=int, default=0, help='debug level')
    parser.add_argument('--threads', type=int, default=0, help='number of threads')
    parser.add_argument('--gpus', type=int, default=8, help='num of GPUs you want to use')
    parser.add_argument('--cpus', type=int, default=8, help='num of CPUs you want to use')

    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = str(args.cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.cpus)
    os.environ['MKL_NUM_THREADS'] = str(args.cpus)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.cpus)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.cpus)
    torch.set_num_threads(args.cpus)

    try:
        seq_name = int(args.seq)
    except:
        seq_name = args.seq

    run_tracker(args.tracker, args.param, args.id, args.dataset, seq_name, args.debug,
                args.threads, num_gpus=args.gpus)


if __name__ == '__main__':
    main()
