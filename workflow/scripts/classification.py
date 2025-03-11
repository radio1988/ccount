# import the necessary packages
from ccount.img.equalize import equalize
from ccount.img.auto_contrast import float_image_auto_contrast
from ccount.img.transform import down_scale

from ccount.blob.io import load_crops, load_locs, save_crops, save_locs
from ccount.blob.mask_image import mask_image
from ccount.blob.misc import crops_stat, parse_crops, crop_width

from ccount.clas.metrics import F1

import sys, argparse, os, re, yaml, keras
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pyimagesearch.cnn.networks.lenet import LeNet
import tensorflow as tf
#from tensorflow.keras.optimizers import Adam


def parse_cmd_and_prep ():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
        description='perfrom classification on crops with trained weights')
    parser.add_argument("-crops", type=str,
                    help="blob-crops file, e.g. res/blob_crops/xxx.crops.npy.gz")
    parser.add_argument("-weight", type=str,
                    help="weights file, e.g. resources/weights/trained.hdf5")
    parser.add_argument("-config", type=str,
                    help="config file, e.g. config.yaml")
    parser.add_argument("-output", type=str,
                    help="output name, e.g. res/xxx.crops.clas.npy.gz")

    args = parser.parse_args()

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Path(args.odir).mkdir(parents=True, exist_ok=True)
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    if config['clas_scaling_factor'] not in [1,2,4]:
        raise Exception(config['clas_scaling_factor'], 'not implemented', 'only support 1,2,4')

    return [args, config]



# Show CPU/GPU info
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# Communication
# print('>>> Command:', sys.argv)
args, config = parse_cmd_and_prep()

crops = load_crops(args.crops)
w = crop_width(crops)
print("crops:", crops[0:3, 0:5])

images, labels, rs = parse_crops(crops)
print("Expanding r by",  config['r_ext_ratio'],  config['r_ext_pixels'])
rs = rs * config['r_ext_ratio'] + config['r_ext_pixels']

print("Downscaling images by ", config['clas_scaling_factor'])
images = np.array([down_scale(image, scaling_factor=config['clas_scaling_factor']) for image in images])
w = int(w/config['clas_scaling_factor'])
rs = rs/config['clas_scaling_factor']

# todo: test skip equalization
# todo: more channels (scaled + equalized + original)
if config['classification_equalization']:
    print("Equalizing images...")
    images = np.array([equalize(image) for image in images])


print("Auto contrasting images...")
images = np.array([float_image_auto_contrast(image) for image in images])

print("Masking images...")
images = np.array([mask_image(image, r=rs[ind]) for ind, image in enumerate(images)])


images = images.reshape((images.shape[0], 2*w, 2*w, 1))
print("max pixel value: ", np.max(images))
print("min pixel value: ", np.min(images))

# Initialize the optimizer and model
# todo: feature normalization (optional)
print("Compiling model...")
#opt = Adam(learning_rate=config['learning_rate'])
opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
model = LeNet.build(numChannels=1, imgRows=2*w, imgCols=2*w, numClasses=config['numClasses'],
                    weightsPath=args.weight)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=[F1])


# Classification process
print('Making classifications...')
probs = model.predict(images)
classifications = probs.argmax(axis=1)
positive_idx = [i for i, x in enumerate(classifications) if x == 1]

# Save 
print("Saving classifications..")
crops[:, 3] = classifications
crops_stat(crops)

save_locs(crops, args.output.replace('crops','locs'))  #todo: fix potential name bug in non-workflow situations
save_crops(crops, args.output)

txt_name = args.output.replace('.npy.gz', '.txt')
np.savetxt(txt_name, classifications.astype(int), fmt='%d')
