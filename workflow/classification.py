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


# Remove unlabeled (only when training)
if args["load_model"] <= 0:
    print("In training mode")
    print("Removing unlabeled blobs")
    blobs = blobs[blobs[:, 3] != -1, :]
    blobs_stat(blobs)


# set other laberls as no
if numClasses == 2:
    print("Remove undistinguishable and artifacts")  # todo: user decide
    blobs[blobs[:, 3] == -2, 3] = 0
    blobs[blobs[:, 3] == 9, 3] = 0
    blobs_stat(blobs)


# Split train/valid
[trainBlobs, valBlobs] = split_train_valid(blobs, training_ratio)
print("{} Split ratio, split Data into {} training and {} testing".\
      format(training_ratio, trainBlobs.shape[0], valBlobs.shape[0]))

# Balancing Yes/No ratio
if args["load_model"] <= 0:
    print('For training split:')
    trainBlobs = balancing_by_duplicating(trainBlobs)
    print('For validation split:')
    valBlobs = balancing_by_duplicating(valBlobs)  #todo: skip this if F1 working well

# Parse blobs
trainImages, trainLabels, trainRs = parse_blobs(trainBlobs)
valImages, valLabels, valRs = parse_blobs(valBlobs)

# Extend Rs
trainRs = trainRs * r_ext_ratio + r_ext_pixels
valRs = valRs * r_ext_ratio + r_ext_pixels

# Mixed Augmentation (todo: aug into more samples)
## todo: contrast, exposure changes
if args["load_model"] < 0:
    print("Before Aug:", trainImages.shape, trainRs.shape, trainLabels.shape)
    trainImages = augment_images(trainImages, aug_sample_size)  # todo: augment to more samples

    ## match sample size of labels and Rs with augmented images
    while trainRs.shape[0] < aug_sample_size:
        trainRs = np.concatenate((trainRs, trainRs))
        trainLabels = np.concatenate((trainLabels, trainLabels))

    trainRs = trainRs[0:aug_sample_size]
    trainLabels = trainLabels[0:aug_sample_size]
    #todo: randomize again

    print("After Aug:", trainImages.shape, trainRs.shape, trainLabels.shape)
    print('max data', np.max(trainImages), 'min', np.min(trainImages))


# Downscale images
print("Downscaling images by ", scaling_factor)
trainImages = np.array([down_scale(image, scaling_factor=scaling_factor) for image in trainImages])
valImages = np.array([down_scale(image, scaling_factor=scaling_factor) for image in valImages])
## Downscale w and R
w = int(w/scaling_factor)
trainRs = trainRs/scaling_factor
valRs = valRs/scaling_factor

# Equalize images (todo: test equalization -> scaling)
# todo: more channels (scaled + equalized + original)
print("Equalizing images...")
trainImages = np.array([equalize(image) for image in trainImages])
valImages = np.array([equalize(image) for image in valImages])

# Mask images
print("Masking images...")
trainImages = np.array([mask_image(image, r=trainRs[ind]) for ind, image in enumerate(trainImages)])
valImages = np.array([mask_image(image, r=valRs[ind]) for ind, image in enumerate(valImages)])

# Normalizing images
print("Normalizing images...")
trainImages = np.array([normalize_img(image) for image in trainImages])
valImages = np.array([normalize_img(image) for image in valImages])


# # Show Images for model training
# for i in range(20):
#     fig, axes = plt.subplots(1, 2, figsize=(8, 16), sharex=True, sharey=True)
#     ax = axes.ravel()

#     ax[0].set_title("Original contrast")
#     ax[0].imshow(trainImages[i], 'gray', clim=(0.0, 1.0))
#     c = plt.Circle((w - 1, w - 1), trainRs[i], color='yellow', linewidth=1, fill=False)
#     ax[0].add_patch(c)

#     ax[1].set_title('HDR')
#     ax[1].imshow(trainImages[i], 'gray')
#     c = plt.Circle((w - 1, w - 1), trainRs[i], color='yellow', linewidth=1, fill=False)
#     ax[1].add_patch(c)
#     print('save fig', i)
#     plt.savefig(str(i)+'.png')

# Reshape for model
trainImages = trainImages.reshape((trainImages.shape[0], 2*w, 2*w, 1))
valImages = valImages.reshape((valImages.shape[0], 2*w, 2*w, 1))
print("max pixel value: ", np.max(trainImages))
print("min pixel value: ", np.min(trainImages))

# Categorize labels for softmax
trainLabels2 = np_utils.to_categorical(trainLabels, numClasses)
valLabels2 = np_utils.to_categorical(valLabels, numClasses)

# Initialize the optimizer and model
# todo: feature normalization (optional)
print("[INFO] compiling model...")
opt = Adam(lr=learning_rate)
model = LeNet.build(numChannels=1, imgRows=2*w, imgCols=2*w, numClasses=numClasses,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=[F1])  # todo: F1
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0, patience=patience,
                                          verbose=verbose, mode='auto',
                                          baseline=None, restore_best_weights=True)
callbacks_list = [earlystop]


# Train if not loading pre-trained weights
if args["load_model"] <= 0:
    print("[INFO] training...")
    # todo: add radius to model
    # todo: augmentation in batch training

    # todo augmentation here is bad, mask issue
    datagen = ImageDataGenerator(
        featurewise_std_normalization=False,
        rotation_range=90,
        shear_range=0.16,
        zoom_range=0.1,
        width_shift_range=0.03, height_shift_range=0.03,
        horizontal_flip=True, vertical_flip=True
        #todo: blur focus
    )

    datagen.fit(trainImages)

    # model.fit(trainImages, trainLabels, validation_data=(valImagesMsk, valLabels),
    #           batch_size=batch_size, epochs=epochs,
    #           verbose=verbose)

    model.fit_generator(datagen.flow(trainImages, trainLabels2, batch_size=batch_size),
                        validation_data=(valImages, valLabels2),
                        steps_per_epoch=len(trainImages) / batch_size, epochs=epochs,
                        callbacks=callbacks_list,
                        verbose=verbose)


    # Evaluation of the model
    print("[INFO] evaluating...")
    (loss, f1) = model.evaluate(trainImages, trainLabels2,
                                      batch_size=batch_size, verbose=verbose)
    print("[INFO] training F1: {:.2f}%".format(f1 * 100))

    print("[INFO] evaluating...")
    (loss,  f1) = model.evaluate(valImages, valLabels2,
                                      batch_size=batch_size, verbose=verbose)
    print("[INFO] validation F1: {:.2f}%".format(f1 * 100))


    # check to see if the model should be saved to file
    if args["save_model"] > 0:
        print("[INFO] dumping weights to file...")
        model.save_weights(args["weights"], overwrite=True)


# For prediction mode
elif args["load_model"] > 0:
    # randomly select a few testing digits
    # todo: fix tk in hpcc
    # _tkinter.TclError: couldn't connect to display ":0.0"
    # load
    # print('loading...', args["blobs_db"])
    # blobs = load_blobs_db(args["blobs_db"])
    # # unlabeled and uncertain to negative
    # print("set unlabeled and uncertain to negative for correct F1 calculation and popping up")
    # blobs[blobs[:, 3] == -1] = 0
    # blobs[blobs[:, 3] == -2] = 0

    # w = int(sqrt(blobs.shape[1] - 6) / 2)  # width of img
    # # parse
    # Images, Labels, Rs = parse_blobs(blobs)  # for human
    # # equalize
    # Images_ = np.array([equalize(image) for image in Images])
    # # scale down
    # Images_ = np.array([down_scale(image, scaling_factor=scaling_factor) for image in Images_])
    # w_ = int(w / scaling_factor)
    # Rs_ = Rs / scaling_factor * r_extension_ratio  # for machine
    # # mask
    # Images_ = np.array([mask_image(image, r=Rs_[ind]) for ind, image in enumerate(Images_)])
    # # reshape for model
    # Images_ = Images_.reshape((Images_.shape[0], 2 * w_, 2 * w_, 1))

    Images = np.vstack((trainImages, valImages))  # todo, don't get complicated code
    Images_ = Images
    Labels = np.concatenate((trainLabels, valLabels))
    Labels_ = Labels
    Rs = np.concatenate((trainRs, valRs))
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
    
    # save yes predictions
    yes_blobs = flat_label_filter(blobs_predict, 1)
    np.save(name+'.yes.npy', yes_blobs)
    os.system('gzip  -f ' + name +'.yes.npy')

    # # Visualizing random predictions
    # print('Showing samples from', args['blobs_db'])
    # for j in range(len(wrong_idx)):
    #     i = wrong_idx[j]  # only show samples predicted to be positive
        
    #     image = Images_[i]
    #     image_ = Images_[i]
    #     prediction = predictions[i]
    #     label = int(Labels[i])
    #     r = Rs[i]
    #     r_ = Rs_[i]  # todo: fix r at the blob detection stage and don't extend r after wards

    #     image = np.reshape(image, image.shape[0:2])
    #     image_ = np.reshape(image_, image_.shape[0:2])

    #     print("[INFO] Predicted: {}, Label: {}".format(prediction, label))

    #     # image = (Images[i] * 255).astype("uint8")
    #     # image_ = (Images_[i] * 255).astype("uint8")

    #     # visualize original and masked/euqalized blobs
    #     fig, axes = plt.subplots(1, 2, figsize=(8, 16), sharex=False, sharey=False)
    #     ax = axes.ravel()
    #     ax[0].set_title('Pred:{} Label:{}\nradius:{}'.\
    #                     format(int(prediction), int(label), r))
    #     ax[0].imshow(image, 'gray')
    #     c = plt.Circle((w - 1, w - 1), r, color='yellow', linewidth=1, fill=False)
    #     ax[0].add_patch(c)
    #     ax[1].set_title('Pred:{} Label:{}\nradius:{}'.\
    #                     format(int(prediction), int(label), int(r_)))
    #     ax[1].imshow(image_, 'gray')
    #     plt.tight_layout()
    #     plt.show()
