from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import tensorflow as tf
from keras._tf_keras.keras.applications import EfficientNetB1




# Paths to datasets
path_to_train_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/train")
path_to_val_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/val")
path_to_test_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/test")

IMG_SIZE = 240
IMG_SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
NUM_CLASSES = 102
epochs = 30


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



model = EfficientNetB1(
    include_top=True,
    weights=None,
    classes=NUM_CLASSES,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

epochs = 30
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
model.evaluate(test_ds)