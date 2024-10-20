import os
import pathlib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from os import walk
import re


path_to_train_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/train")
path_to_val_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/val")
path_to_test_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/test")

path_to_train_set_bgr = "/home/jordan/Insect Pest Classification Dataset/classification/trainbgr"
path_to_val_set_bgr = "/home/jordan/Insect Pest Classification Dataset/classification/valbgr"
path_to_test_set_bgr = "/home/jordan/Insect Pest Classification Dataset/classification/testbgr"

def zero_center_normalization(bgr_img):
    mean = [103.939, 116.779, 123.68]
    std = None
    if data_format == 'channels_first':
        if bgr_img.ndim == 3:
            bgr_img[0, :, :] -= mean[0]
            bgr_img[1, :, :] -= mean[1]
            bgr_img[2, :, :] -= mean[2]
            if std is not None:
                bgr_img[0, :, :] /= std[0]
                bgr_img[1, :, :] /= std[1]
                bgr_img[2, :, :] /= std[2]
        else:
            bgr_img[:, 0, :, :] -= mean[0]
            bgr_img[:, 1, :, :] -= mean[1]
            bgr_img[:, 2, :, :] -= mean[2]
            if std is not None:
                bgr_img[:, 0, :, :] /= std[0]
                bgr_img[:, 1, :, :] /= std[1]
                bgr_img[:, 2, :, :] /= std[2]
    else:
        bgr_img[..., 0] -= mean[0]
        bgr_img[..., 1] -= mean[1]
        bgr_img[..., 2] -= mean[2]
        if std is not None:
            bgr_img[..., 0] /= std[0]
            bgr_img[..., 1] /= std[1]
            bgr_img[..., 2] /= std[2]
    return bgr_img


def dataset_to_bgr(origin_image_directory, destination_image_directory):
    for subdir, dirs, files in os.walk(origin_image_directory):
        newdir = re.sub(str(origin_image_directory), destination_image_directory, subdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        for file in files:
            img = cv2.imread(os.path.join(subdir, file))
            bgr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{newdir}/{file}", bgr_img)
            print(f"{newdir}/{file}")

dataset_to_bgr(path_to_val_set, path_to_val_set_bgr)
dataset_to_bgr(path_to_test_set, path_to_test_set_bgr)
dataset_to_bgr(path_to_train_set, path_to_train_set_bgr)


#Loading datasets with keras for EfficientNetB0

from tensorflow.data import Dataset

img_width = 224
img_height = 224
batch_size = 32
num_classes = 102

normalization_layer = tf.keras.layers.Rescaling(1. / 255)


