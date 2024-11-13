from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import time

from keras import layers
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras.applications import EfficientNetB0
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

# Paths to datasets
path_to_train_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/train")
path_to_val_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/val")
path_to_test_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/test")

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
NUM_CLASSES = 102
epochs = 30

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 8GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# Dataset loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_train_set,
    label_mode="int",
    seed=1,
    batch_size=BATCH_SIZE,
    image_size=size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_val_set,
    label_mode="int",
    seed=1,
    batch_size=BATCH_SIZE,
    image_size=size)

test_ds = tf.keras.utils.image_dataset_from_directory(path_to_test_set,
                                                      label_mode="int",
                                                      seed=1,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=size)

# Image augmentation layers
img_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.2),  #removing it lower accuracy by 0.38%
    # layers.RandomBrightness(factor=0.2)  lowers accuracy by 0.8%
]

def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images


# Preprocessing functions for training and validation datasets
def input_preprocess_train(image, label):
    image = img_augmentation(image)  # Apply augmentation
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def input_preprocess_val_and_test(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


# Apply preprocessing
train_ds = train_ds.map(input_preprocess_train, num_parallel_calls=10)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(input_preprocess_val_and_test, num_parallel_calls=10)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(input_preprocess_val_and_test, num_parallel_calls=10)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


#Transfer Learning from Pre-Trained Weights
def build_feature_extractor():
    base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet")
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D()
    ])
    return model


feature_extractor = build_feature_extractor()


def extract_features(dataset, feature_extractor):
    features = []
    labels = []
    loop_counter = 0
    for batch_images, batch_labels in dataset:
        batch_features = feature_extractor(batch_images)
        features.append(batch_features.numpy())
        labels.append(batch_labels.numpy())
        loop_counter += 1
        print(F"Extracting features on loop {loop_counter}.")
    return np.concatenate(features), np.concatenate(labels)


#Extract features for every set
train_features, train_labels = extract_features(train_ds, feature_extractor)
val_features, val_labels = extract_features(val_ds, feature_extractor)
test_features, test_labels = extract_features(test_ds, feature_extractor)

print("All features extracted.")
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def compute_kernel_entry(i, X, Y, p, constant=1e-8):
    row = np.zeros(Y.shape[0])
    for j in range(Y.shape[0]):
        # Avoid zero values by adding constant
        safe_product = X[i] * Y[j] + constant
        mean_value = np.mean(np.power(safe_product, p))
        # Avoid near-zero mean values
        row[j] = 0 if mean_value < constant else np.power(mean_value, 1 / p)
    return row


def power_mean_kernel(X, Y=None, p=-1.0, epsilon=1e-8):
    if Y is None:
        Y = X

    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    kernel_matrix = np.zeros((n_samples_X, n_samples_Y))
    loop_counter = 0
    for i in range(n_samples_X):
        loop_counter += 1
        print(f"Power mean loop at {loop_counter}.")
        for j in range(n_samples_Y):
            # Avoid zero values by adding epsilon, then apply the power operation
            safe_product = X[i] * Y[j] + epsilon

            # Ensure the mean of the powered terms doesn't approach zero when p < 0
            mean_value = np.mean(np.power(safe_product, p))
            if mean_value < epsilon:  # Check for near-zero mean values
                kernel_matrix[i, j] = 0
            else:
                kernel_matrix[i, j] = np.power(mean_value, 1 / p)

    return kernel_matrix


p = -1 # Adjust this to your desired power

# Compute the training kernel matrix with optimized power-mean kernel function
train_kernel_matrix = power_mean_kernel(train_features, train_features, p=p)
print("Training kernel matrix declared and computed")
# Train SVM with the precomputed kernel matrix
svm_classifier = SVC(kernel="precomputed")
print("Training SVM")
svm_classifier.fit(train_kernel_matrix, np.argmax(train_labels, axis=1))

# Compute kernel matrix for validation set
val_kernel_matrix = power_mean_kernel(val_features, train_features, p=p)
print("Validation kernel matrix declared and computed")
val_preds = svm_classifier.predict(val_kernel_matrix)
print("Validation Accuracy:", accuracy_score(val_labels, val_preds))

# Compute kernel matrix for test set
test_kernel_matrix = power_mean_kernel(test_features, train_features, p=p)
print("Testing kernel matrix declared and computed")
test_preds = svm_classifier.predict(test_kernel_matrix)
print("Test Accuracy:", accuracy_score(test_labels, test_preds))
print(classification_report(test_labels, test_preds))
