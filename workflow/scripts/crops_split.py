import argparse, textwrap
from ccount.clas.split_data import split_data
from ccount.blob.io import load_crops, save_crops


def parse_cmd_and_prep ():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
    	formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        	>>> Usage: 
        	split_data.py -crops crops.labled.npy.gz -ratio 0.7
        	>>> Output:
        	crops.0.7.npy.gz cropts.0.3.npy.gz
        	'''))
    parser.add_argument("-crops", type=str,
        help="labled blob-crops file, e.g. labeled/labeled.crops.npy.gz")
    parser.add_argument("-ratio", type=float,
        help="ratio of train/val/test, e.g. 0.7")

    args = parser.parse_args()

    if args.ratio > 1 or args.ratio < 0:
    	raise ValueError('ratio should be between 0, 1')

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    return args


args = parse_cmd_and_prep()

crops = load_crops(args.crops)

[crops1, crops2] = split_data(crops, args.ratio)
print(crops1.shape, crops2.shape)
out_name1 = args.crops.replace(".npy.gz", "." + str(args.ratio) + ".npy.gz")
out_name2 = args.crops.replace(".npy.gz", "." + str(round(1-args.ratio, 3)) + ".npy.gz")
if out_name2 == out_name1:
    out_name2 = out_name1.replace("npy.gz", "b.npy.gz")
if out_name2 == out_name1:
    raise ValueError("outname2 == outname1")

save_crops(crops1, out_name1)
save_crops(crops2, out_name2)