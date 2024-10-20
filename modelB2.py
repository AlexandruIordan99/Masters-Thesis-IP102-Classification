from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
from keras import layers
import tensorflow as tf
from keras._tf_keras.keras.applications import EfficientNetB2
import argparse



path_to_train_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/train")
path_to_val_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/val")
path_to_test_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/test")

#Dataset with images converted to BGR
path_to_train_setbgr = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/trainbgr")
path_to_val_setbgr = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/valbgr")
path_to_test_setbgr = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/testbgr")


IMG_SIZE = 260
size = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
NUM_CLASSES = 102


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


img_augmentation_layers = [
    layers.Normalization(axis =-1, mean= [0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225]), #using ImageNet means
    # and variance
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
    layers.RandomCrop(height=IMG_SIZE, width=IMG_SIZE),
    layers.RandomBrightness(factor=0.1),
    layers.RandomZoom(height_factor=0.1),
]



def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images


# One-hot / categorical encoding
def input_preprocess_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def input_preprocess_val(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


# train_ds = train_ds.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
#
# train_ds = train_ds.batch(batch_size=BATCH_SIZE,
#                           drop_remainder=True,
#                           num_parallel_calls=None,
#                           deterministic=None,
#                           name=None).unbatch()


model_B2 = EfficientNetB2(include_top=True,
                          weights=None,
                          classes = NUM_CLASSES,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))

model_B2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model_B2.summary()

epochs = 30


def main():
    global model_B2
    global epochs
    global train_ds
    global val_ds
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Pass a trained model to skip training")
    parser.add_argument("-s", "--save", help ="Saves a trained model")
    args = parser.parse_args()

    if args.model:
        print(f"Loading {args.model}...")
        model_B2.load_weights(args.model)
        model_B2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        history = model_B2.predict(train_ds, validation_data=test_ds, epochs=epochs)
    else:
        print(f"Training {args.save}...")
        model_B2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        hist_1= model_B2.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size= 64)

    if args.save:
        print(f"Saving {args.save}...")
        model_B2.save_weights(args.save)


main()

