from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
from keras import layers
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras.applications import EfficientNetV2L
import argparse
import numpy as np
from keras.src.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import mixed_precision

# Paths to datasets
path_to_train_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/trainbgr")
path_to_val_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/valbgr")
path_to_test_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/testbgr")

IMG_SIZE = 380
IMG_SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
NUM_CLASSES = 102
epochs = 30

# Dataset loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_train_set,
    label_mode="int",
    seed=1,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_val_set,
    label_mode="int",
    seed=1,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)

test_ds = tf.keras.utils.image_dataset_from_directory(path_to_test_set,
                                                      label_mode="int",
                                                      seed=1,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)


mean_ImageNet= [0.485, 0.456, 0.406]
st_dev_ImageNet= [0.229, 0.224, 0.225]



# Image augmentation layers
img_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.2),  #removing it lower accuracy by 0.38%
    # layers.RandomBrightness(factor=0.2)  lowers accuracy by 0.8%
]



img_normalization_layers = [
    layers.Normalization(axis=-1, mean=mean_ImageNet, variance=st_dev_ImageNet)]  # ImageNet means



# Define the normalization parameters for ImageNet
mean_imagenet = [0.485, 0.456, 0.406]  # Mean for R, G, B channels
std_dev_imagenet = [0.229, 0.224, 0.225]  # Standard deviation for R, G, B channels



# Normalization function
def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images


def img_normalization(images):
    for layer in img_normalization_layers:
        images = layer(images)
    return images


# Preprocessing functions for training and validation datasets
def input_preprocess_train(image, label):
    # image = normalize_image(image) #Normalize before augmentation
    image = img_augmentation(image)  # Apply augmentation
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def input_preprocess_val(image, label):
    # image = normalize_image(image)  # Normalize only
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def input_preprocess_test(image, label):
    # image = normalize_image(image)  # Normalize only
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


# Apply preprocessing
train_ds = train_ds.map(input_preprocess_train, num_parallel_calls=10)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(input_preprocess_val, num_parallel_calls=10)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)  #CHECK FUNCTION CALL 25.10.2024

test_ds = test_ds.map(input_preprocess_test, num_parallel_calls=10)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


#Transfer Learning from Pre-Trained Weights
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetV2L(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    for layer in model.layers[:-15]:  # Freeze the first 20 layers
        #changed this from -15 to -10
        layer.trainable = False

    # Rebuild top layers
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    return model



model = build_model(num_classes=NUM_CLASSES)


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


early_stop_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)


epochs = 10  # @param {type: "slider", min:8, max:80}
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks= early_stop_callback)
plot_hist(hist)


def unfreeze_model(model):
    # We unfreeze the top 10 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-15:]:  #changed this from -15 to -10
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(),
                 keras.metrics.Precision(),
                 keras.metrics.Recall()]
    )


unfreeze_model(model)


epochs = 30  # @param {type: "slider", min:4, max:10}
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
plot_hist(hist)
model.save_weights("/home/jordan/modelV2LdataAug.weights.h5")
model.evaluate(test_ds)
model.load_weights("/home/jordan/modelV2LdataAug.weights.h5")
model.evaluate(test_ds)


#Main function for training and saving the model
