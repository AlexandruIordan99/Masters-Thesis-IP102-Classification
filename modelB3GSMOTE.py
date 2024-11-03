from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
from keras import layers
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras.applications import EfficientNetB3
import argparse
import numpy as np
from keras.src.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from geometric_smote import GeometricSMOTE

# Paths to datasets
path_to_train_setbgr = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/trainbgr")
path_to_val_setbgr = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/valbgr")
path_to_test_setbgr = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/testbgr")
IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
NUM_CLASSES = 102
epochs = 30

# Dataset loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_train_setbgr,
    label_mode="int",
    seed=1,
    batch_size=BATCH_SIZE,
    image_size=size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_val_setbgr,
    label_mode="int",
    seed=1,
    batch_size=BATCH_SIZE,
    image_size=size)

test_ds = tf.keras.utils.image_dataset_from_directory(path_to_test_setbgr,
                                                      label_mode="int",
                                                      seed=1,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=size)

print("Datasets are loaded")


#Pre-Processing Data for G-Smote, need to flatten the images
#batches are used to avoid running out of memory
def dataset_to_numpy_in_batches(dataset, batch_size):
    images, labels = [], []
    current_batch_size = 0
    i = 0
    for img_batch, label_batch in dataset:
        print("Processing image + label pair...")
        i += 1
        # Accumulate data
        images.append(img_batch.numpy())
        labels.append(label_batch.numpy())
        current_batch_size += img_batch.shape[0]  # Track the current number of samples

        # If we have enough data for a complete batch, yield it
        if current_batch_size >= batch_size:
            print(f"Yielding Batch number {i}")
            # Concatenate and truncate to ensure the batch is exactly batch_size
            X = np.concatenate(images)[:batch_size]
            y = np.concatenate(labels)[:batch_size]

            yield X, y

            # Reset images, labels, and the current batch size for the next batch
            remaining_images = np.concatenate(images)[batch_size:]
            remaining_labels = np.concatenate(labels)[batch_size:]

            images = [remaining_images] if len(remaining_images) > 0 else []
            labels = [remaining_labels] if len(remaining_labels) > 0 else []
            current_batch_size = len(remaining_images)

    # Yield any remaining data if it's smaller than batch_size
    if images and current_batch_size > 0:
        print("Yielding final incomplete batch")
        X = np.concatenate(images)
        y = np.concatenate(labels)
        yield X, y


def batch_g_smote_resampling(dataset, batch_size):
    """Apply G-SMOTE to dataset in smaller batches with safe k_neighbors adjustment."""
    X_resampled_list = []
    y_resampled_list = []

    # Process dataset in chunks/batches
    for X_batch, y_batch in dataset_to_numpy_in_batches(dataset, batch_size):
        # Flatten images for G-SMOTE
        X_batch_flattened = X_batch.reshape(X_batch.shape[0], -1)

        # Find the minimum number of samples in any class for the current batch
        class_counts = np.bincount(y_batch)
        min_samples_in_class = min(class_counts[class_counts > 0])  # Ignore zero counts

        # Initialize variables to avoid UnboundLocalError
        X_resampled_batch, y_resampled_batch = None, None

        # Check if there are enough samples to perform G-SMOTE
        if min_samples_in_class > 1:
            adjusted_k_neighbors = max(1, min(5, min_samples_in_class - 1))
            g_smote = GeometricSMOTE(k_neighbors=adjusted_k_neighbors)

            # Apply G-SMOTE to the current batch
            X_resampled_batch, y_resampled_batch = g_smote.fit_resample(X_batch_flattened, y_batch)

            # Reshape images back to original format and collect the data
            X_resampled_batch = X_resampled_batch.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        else:
            # If there aren't enough samples, use the original batch
            print(f"Skipping resampling: Not enough samples in a class, min_samples_in_class={min_samples_in_class}")
            X_resampled_batch, y_resampled_batch = X_batch, y_batch

        # Add the resampled or original data to the lists
        X_resampled_list.append(X_resampled_batch)
        y_resampled_list.append(y_resampled_batch)

    # Concatenate all resampled batches
    X_resampled = np.concatenate(X_resampled_list, axis=0)
    y_resampled = np.concatenate(y_resampled_list, axis=0)

    return X_resampled, y_resampled


print("Dataset to numpy and resampling functions are defined....")
print("Resampling...")
X_resampled, y_resampled = batch_g_smote_resampling(train_ds, batch_size=64)

print("X_resampled and y_resampled are loaded")

print("Loading Scaler")
scaler = StandardScaler()
print("Scaler is loaded, scaling X_resampled and y_resampled and changing their shapes....")

print(f"X_resampled is of type" + f"{type(X_resampled)}")
print(f"y_resampled is of type" + f"{type(y_resampled)}")

X_resampled = scaler.fit_transform(X_resampled.reshape(X_resampled.shape[0], -1)).reshape(X_resampled.shape)
print("Scaling and reshaping successful.")

print(f"X_resampled after scaling is of type {type(X_resampled)}")
print(f"X_resampled is of shape {X_resampled.shape}")  #Dataset is too big to load into a tensor slice at once


def dataset_reshaper(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]


train_ds = tf.data.Dataset.from_generator(
    lambda: dataset_reshaper(X_resampled, y_resampled, BATCH_SIZE),
    output_signature=(tf.TensorSpec(shape=(None, 300, 300, 3), dtype=tf.float32),  #CHANGE RESOLUTION BASED ON MODEL
                      tf.TensorSpec(shape=(None,), dtype=tf.float32)
                      ))

train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print(f"train_ds is batched, its type is {type(train_ds)} and its shape is {train_ds.element_spec}")

#Callback to stop training if val loss does not improve after 3 epochs
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


def one_hot_encode(image, label):
    label = tf.cast(label, tf.int32)
    return image, tf.one_hot(label, depth=NUM_CLASSES)


# Apply the one-hot encoding to datasets
train_ds = train_ds.map(one_hot_encode,  num_parallel_calls=10)
val_ds = val_ds.map(one_hot_encode, num_parallel_calls=10)
test_ds = test_ds.map(one_hot_encode, num_parallel_calls=10)


#Transfer Learning from Pre-Trained Weights
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB3(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    for layer in model.layers[:-20]:  # Freeze the first 20 layers
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
            keras.metrics.AUC(),
            keras.metrics.Precision(),
            keras.metrics.Recall()
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
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=early_stop_callback)
plot_hist(hist)


def unfreeze_model(model):
    # Unfreeze top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
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
model.save_weights("/home/jordan/modelV1B3GSMOTE.weights.h5")
model.evaluate(test_ds)

