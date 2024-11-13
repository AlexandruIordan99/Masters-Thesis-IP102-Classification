from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
from keras import layers
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras.applications import EfficientNetB0
import argparse

from matplotlib import pyplot as plt
from sklearn import svm


img_augmentation_layers = [
    layers.Normalization(axis=-1, mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225]),  # ImageNet means
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
    layers.RandomZoom(height_factor=0.1)
]

modelB0= EfficientNetB0(weights='imagenet')

model = tf.keras.Sequential(
    layers=img_augmentation_layers, trainable=True, name="efficientnetb0"
)


model.summary()
modelB0.summary()