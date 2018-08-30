import argparse
import sys

from .mugen import Mugen


def run():
    parser = argparse.ArgumentParser(description='Music generator')
    subparsers = parser.add_subparsers(dest='mode', help='Mode')

    subparser = subparsers.add_parser('train')
    subparser.add_argument('--data-dir', default='data')

    # subparser = subparsers.add_parser('gen')

    args = parser.parse_args(sys.argv[1:])

    if args.mode == 'train':
        mugen = Mugen(args.data_dir)
        mugen.train()


run()
