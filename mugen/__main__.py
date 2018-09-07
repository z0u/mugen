import argparse
import sys


def run():
    parser = argparse.ArgumentParser(description='Music generator')
    subparsers = parser.add_subparsers(dest='mode', help='Mode')

    subparser = subparsers.add_parser('test-midi')

    subparser = subparsers.add_parser('train')
    subparser.add_argument('--data-dir', default='data')

    # subparser = subparsers.add_parser('gen')

    args = parser.parse_args(sys.argv[1:])

    if args.mode == 'train':
        from .mugen import Mugen
        mugen = Mugen(args.data_dir)
        mugen.train()
    elif args.mode == 'test-midi':
        from .output import play_simple_test
        play_simple_test()


run()
