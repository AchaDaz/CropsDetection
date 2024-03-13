import argparse

from prepare import set_random_seeds, setup
from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--tl', default=1, type=int, metavar='N',
                        help='flag pre training/transfer learning')
    parser.add_argument("--local_rank", type=int, 
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    args = parser.parse_args()
    
    setup()
    set_random_seeds(random_seed=0)
    train(args.tl, args.epochs, args.local_rank)

if __name__ == "__main__":
    main()