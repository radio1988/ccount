# import the necessary packages
from pyimagesearch.cnn.networks.lenet import LeNet
from ccount import *
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter

import sys
import os
import argparse
import numpy as np
import keras
from pathlib import Path

import matplotlib.pyplot as plt  # tk not on hpcc
matplotlib.use('Agg')  # not display on hpcc
import cv2  # not on hpcc


# Show CPU/GPU info
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# Communication
print('example training: python lenet_colonies.py -db mid.strict.npy.gz -s 1 -w ./output/mid.strict.hdf5')
print('example loading: python lenet_colonies.py -db mid.strict.npy.gz -l 1 -w ./output/mid.strict.hdf5')
print('cmd:', sys.argv)
# todo: change format to pandas to count positives for each scanned image (for now, image-> npy -> count)


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-db", "--blobs-db", type=str,
                help="path to blobs-db file, e.g. xxx.labeled.npy.gz")
ap.add_argument("-odir", "--outdir", type=str,
                help="outdir, e.g filter1, filter2")
ap.add_argument("-s", "--save-model", type=int, default=0,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=0,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
                help="(optional) path to weights file")
ap.add_argument("-u", "--undistinguishable", type=str, default="delete",
                help="(optional) treat undistinguishable by delete/convert_to_yes/convert_to_no")
ap.add_argument("-e", "--epochs", type=int, default=30,
                help="(optional) max-epochs, default 30")

args = vars(ap.parse_args())
name = os.path.basename(args['blobs_db'])
name = name.replace(".npy", "")
name = name.replace(".gz", "")
Path(args['outdir']).mkdir(parents=True, exist_ok=True)
name = os.path.join(args['outdir'], name)
print("name", name)
# todo: get wrong ones, reload and refine train

# Parameters
scaling_factor = 2  # input scale down for model training
aug_sample_size = 2000
training_ratio = 0.9  # proportion of data to be in training set
r_ext_ratio = 1.4  # larger (1.4) for better view under augmentation
r_ext_pixels = 30
numClasses=2
batch_size=64  # default 64
epochs = args["epochs"]  # default 500
patience = 3  # default 50
learning_rate = 0.0001  # default 0.0001 (Adam)
verbose = 2  # {0, 1, 2}


# Load Labeled blobs_db
blobs = load_blobs_db(args["blobs_db"])
w = int(sqrt(blobs.shape[1]-6) / 2)  # width/2 of img


# set other laberls as no
if numClasses == 2:
    print("Remove undistinguishable and artifacts")  # todo: user decide
    blobs[blobs[:, 3] == -2, 3] = 0
    blobs[blobs[:, 3] == 9, 3] = 0
    blobs_stat(blobs)



# Parse blobs
Images, Labels, Rs = parse_blobs(blobs)

# Extend Rs
Rs = Rs * r_ext_ratio + r_ext_pixels


# Downscale images
print("Downscaling images by ", scaling_factor)
Images = np.array([down_scale(image, scaling_factor=scaling_factor) for image in Images])

## Downscale w and R
w = int(w/scaling_factor)
Rs = Rs/scaling_factor

# Equalize images (todo: test equalization -> scaling)
# todo: more channels (scaled + equalized + original)
print("Equalizing images...")
Images = np.array([equalize(image) for image in Images])

# Mask images
print("Masking images...")
Images = np.array([mask_image(image, r=Rs[ind]) for ind, image in enumerate(Images)])

# Normalizing images
print("Normalizing images...")
Images = np.array([normalize_img(image) for image in Images])

# Reshape for model
Images = Images.reshape((Images.shape[0], 2*w, 2*w, 1))
print("max pixel value: ", np.max(Images))
print("min pixel value: ", np.min(Images))

# Categorize labels for softmax
Labels2 = np_utils.to_categorical(Labels, numClasses)

# Initialize the optimizer and model
# todo: feature normalization (optional)
print("[INFO] compiling model...")
opt = Adam(lr=learning_rate)
model = LeNet.build(numChannels=1, imgRows=2*w, imgCols=2*w, numClasses=numClasses,
                    weightsPath=args["weights"] if int(args["load_model"]) > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=[F1])  # todo: F1
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0, patience=patience,
                                          verbose=verbose, mode='auto',
                                          baseline=None, restore_best_weights=True)
callbacks_list = [earlystop]



# Classification process
Images_ = Images
Labels_ = Labels
Rs_ = Rs

print("Images_.shape:", Images_.shape)
# Predictions
print('Making predictions...')
probs = model.predict(Images_)
predictions = probs.argmax(axis=1)
positive_idx = [i for i, x in enumerate(predictions) if x == 1]

print("Labels:", Labels.shape, Counter(Labels))
print("predictions:", predictions.shape, Counter(predictions))
print("Manual F1 score: ", F1_calculation(predictions, Labels))


wrong_idx = [i for i, x in enumerate(predictions) if (int(predictions[i]) - int(Labels[i])) != 0]
print("Predictions: mean: {}, count_yes: {} / {};".format(
    np.mean(predictions), np.sum(predictions), len(predictions)))
print("Wrong predictions: {}".format(len(wrong_idx)))

# save predictions
# blobs[:, 3] = predictions  # have effect on Labels[i]
print("saving predictions")
np.savetxt(name +'.pred.txt', predictions.astype(int))
blobs_predict = np.copy(blobs)
blobs_predict[:, 3] = predictions
np.save(name+'.pred.npy', blobs_predict)
os.system('gzip  -f ' + name+'.pred.npy')

# # save yes predictions
# yes_blobs = flat_label_filter(blobs_predict, 1)
# np.save(name+'.yes.npy', yes_blobs)
# os.system('gzip  -f ' + name +'.yes.npy')
