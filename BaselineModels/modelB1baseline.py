from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import tensorflow as tf
from keras._tf_keras.keras.applications import EfficientNetB1


# Paths to datasets
path_to_train_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/trainbgr")
path_to_val_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/valbgr")
path_to_test_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/testbgr")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 8GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)



IMG_SIZE = 240
BATCH_SIZE = 32
NUM_CLASSES = 102
epochs = 30

def preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)  # Convert labels to one-hot encoding
    return image, label


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


train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)
test_ds = test_ds.map(preprocess)


model = EfficientNetB1(
    include_top=True,
    weights=None,
    classes=NUM_CLASSES,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy', tf.keras.metrics.AUC(),
                                                                          tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model.summary()

epochs = 30
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
model.evaluate(test_ds)