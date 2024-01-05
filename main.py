import os
import argparse
import datetime
import torch.multiprocessing as mp

from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-tl', default=1, type=int, metavar='N',
                        help='flag pre training/transfer learning')
    args = parser.parse_args()
    
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4444'

    mp.spawn(train, nprocs=args.gpus, args=(args,))

if __name__ == '__main__':
    main()