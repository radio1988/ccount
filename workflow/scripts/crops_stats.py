import argparse, textwrap
from ccount.blob.io import load_crops
from ccount.blob.misc import crops_stat


def parse_cmd_and_prep():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Read crops.npy.gz or locs.npy.gz and give statistics about labels")

    parser.add_argument("-crops", type=str,
                        help="labled blob-crops file, e.g. labeled/labeled.crops.npy.gz")

    args = parser.parse_args()

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    return args

args = parse_cmd_and_prep()
crops = load_crops(args.crops)
crops_stat(crops)