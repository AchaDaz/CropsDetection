import argparse
import torch.multiprocessing as mp

from prepare import setup, cleanup
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

    setup(args.nr, args.world_size)
    mp.spawn(train,
             nprocs=args.world_size, 
             args=(args,), 
             join=True)

    cleanup()
if __name__ == '__main__':
    main()