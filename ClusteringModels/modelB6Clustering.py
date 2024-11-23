from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import numpy as np
import tensorflow as tf
from keras._tf_keras import keras
from tensorflow.keras import layers, models
from keras._tf_keras.keras.applications import EfficientNetB6
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from keras._tf_keras.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras import mixed_precision

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
#physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = 260
BATCH_SIZE = 16
NUM_CLASSES = 102
max_clusters = 6
threshold = 0.2

# Paths to datasets
path_to_train_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/trainbgr")
path_to_val_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/valbgr")
path_to_test_set = pathlib.PosixPath(
    "/home/jordan/Insect Pest Classification Dataset/classification/testbgr")

# Load and preprocess datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_train_set, label_mode="int", seed=1, batch_size=BATCH_SIZE, image_size=(IMG_SIZE, IMG_SIZE))

val_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_val_set, label_mode="int", seed=1, batch_size=BATCH_SIZE, image_size=(IMG_SIZE, IMG_SIZE))

test_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_test_set, label_mode="int", seed=1, batch_size=BATCH_SIZE, image_size=(IMG_SIZE, IMG_SIZE))

# Mean and standard deviation for ImageNet normalization
mean_imagenet = [0.485, 0.456, 0.406]
std_dev_imagenet = [0.229, 0.224, 0.225]

# Image augmentation layers
img_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.2),
]

# Data augmentation and normalization functions
def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images

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
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)  #CHECK FUNCTION CALL 25.10.2024

test_ds = test_ds.map(input_preprocess_val_and_test, num_parallel_calls=10)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Feature extractor model using EfficientNetB6
base_model = EfficientNetB6(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor_model = models.Model(inputs=base_model.input, outputs=x)


mixed_precision.set_global_policy('mixed_float16') #using half precision variables to save memory and avoid OOM errors
classifier_head = layers.Dense(NUM_CLASSES, activation='softmax', dtype ='float32')(x) #need to specify output data as float 32
classifier_model = models.Model(inputs=base_model.input, outputs=classifier_head)
classifier_model.compile(optimizer=Adam(learning_rate=1e-4),
                         loss='categorical_crossentropy',
                         metrics=['accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])

# Adaptive clustering setup
num_allowed_clusters = [1] * NUM_CLASSES
flags = [0] * NUM_CLASSES

# Training loop with clustering and classifier training
converged = False
while not converged:
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    loop_counter = 0
    # Extract features for training images with augmentation
    for images, labels in train_ds:
        features = feature_extractor_model(images, training=False)
        train_features.append(features)
        train_labels.append(labels)
        loop_counter += 1
        print(f"Extracting train features, loop number at {loop_counter}.")

    train_features = tf.concat(train_features, axis=0)
    train_labels = tf.concat(train_labels, axis=0)
    pseudo_labels = to_categorical(train_labels, NUM_CLASSES)

    pca = PCA(n_components=32)
    train_features = pca.fit_transform(train_features.numpy()) #reducing train dimensionality from 1280 to 48
                                                               #this should help save memory to avoid an OOM error

    # Extract features for validation images without augmentation
    loop_counter = 0
    for images, labels in val_ds:
        features = feature_extractor_model(images, training=False)
        val_features.append(features)
        val_labels.append(labels)
        loop_counter += 1
        print(f"Extracting validation features, loop number at {loop_counter}.")

    val_features = tf.concat(val_features, axis=0)
    val_labels = tf.concat(val_labels, axis=0)
    val_pseudo_labels = to_categorical(val_labels, NUM_CLASSES)

    # Train the classifier model
    classifier_model.fit(train_ds, validation_data=val_ds, epochs=3)

    # Validate and calculate confusion matrix
    val_predictions = classifier_model.predict(val_ds)
    val_predictions = np.argmax(val_predictions, axis=1)
    val_labels = np.argmax(val_labels, axis=1)

    cm = confusion_matrix(val_labels, val_predictions)

    # Normalize and display confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')
    plt.show()


    # Adjust clusters based on confusion matrix
    for class_id in range(NUM_CLASSES):
        false_negatives = cm[class_id].sum() - cm[class_id][class_id]
        if false_negatives > threshold and flags[class_id] == 0:
            num_allowed_clusters[class_id] = min(num_allowed_clusters[class_id] + 1, max_clusters)
            if num_allowed_clusters[class_id] == max_clusters:
                flags[class_id] = 1
        elif false_negatives > threshold and flags[class_id] == 1:
            num_allowed_clusters[class_id] = max(num_allowed_clusters[class_id] - 1, 1)
            if num_allowed_clusters[class_id] == 1:
                flags[class_id] = 0


    # Cluster using K-Means for each class
    clusters_per_class = {}
    for class_id in range(NUM_CLASSES):
        class_indices = tf.where(train_labels == class_id)[:, 0]
        class_features = tf.gather(train_features, class_indices)
        if class_features.shape[0] > 0:
            kmeans_instance = KMeans(n_clusters=num_allowed_clusters[class_id], random_state=42)
            kmeans_instance.fit(class_features)
            clusters_per_class[class_id] = kmeans_instance.labels_

    # Convert one-hot encoded `train_labels` back to class indices
    train_labels_indices = np.argmax(train_labels, axis=1) #reshaping array to match the shape
                                                           # of the cluster pseudo labels

    cluster_labels = np.full(train_labels_indices.shape, -1)
    for class_id, clusters in clusters_per_class.items():
        # Get indices for the current class
        class_indices = np.where(train_labels == class_id)[0]
        # Assign clusters to the corresponding class indices
        cluster_labels[class_indices] = clusters  # Assign directly as integers

    pseudo_labels = to_categorical(cluster_labels, num_classes=NUM_CLASSES)
    print(f"Current -allowed-clusters: {num_allowed_clusters}")
    del train_features, val_features
    tf.keras.backend.clear_session()



    if all(flag == 1 for flag in flags):
        print("Training completed with clustering and data augmentation.")
        print("Evaluating the model after clustering adjustments:")
        classifier_model.evaluate(test_ds)
        converged = True
