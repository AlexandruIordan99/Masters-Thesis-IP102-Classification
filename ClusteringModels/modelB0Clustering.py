from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import numpy as np
import tensorflow as tf
from keras._tf_keras import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from keras._tf_keras.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#You can replace B0 with B1, B2.. and so on, or V2B0, V2B1 etc., or just run the files from their folders
#B0 is mentioned on lines 17 & 122
from keras._tf_keras.keras.applications import EfficientNetB0


#Either set a hard limit, which actually takes up your whole memory UPFRONT, instead of taking VRAM as needed
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     #Never more than 9gb so you can run the rest of your computer at an okay speed
#     #Adjust as needed
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]) # Notice here
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

#OR use an adaptible memory setting, but it may cause memory fragmentation
#When using one, comment the other one out, as such:

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#The reccomendation is as follows:
#Use memory growth for lower models, B0, B1 etc to about B4
#Use a hard value for large models such as B5 or V2S etc.


IMG_SIZE = 224
BATCH_SIZE = 16 #16 for clustering, 32 for augmentation to reduce 00M errors
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


#Add augmentation to train set
def input_preprocess_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


#Extract the labels from the validation and test sets
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


#Extracting features while using ImageNet weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor_model = models.Model(inputs=base_model.input, outputs=x)


classifier_head = layers.Dense(NUM_CLASSES, activation='softmax')(x)
classifier_model = models.Model(inputs=base_model.input, outputs=classifier_head)
classifier_model.compile(optimizer=Adam(learning_rate=1e-4),
                         loss='categorical_crossentropy',
                         metrics=['accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])



#Clustering setup
num_allowed_clusters = [1] * NUM_CLASSES
flags = [0] * NUM_CLASSES

print("Beginning clustering loop.")
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

    train_features = PCA(n_components=64).fit_transform(train_features.numpy()) #having train dimensionality at 1280 leads
    #to an OOM error
    #reducing train dimensionality from 1280 to 128, 64, 48 or even 32 to avoid OOM errors
    #Change the value depending on the model

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

    # Train the classifier model
    classifier_model.fit(train_ds, validation_data=val_ds, epochs=6)

    # Validate and calculate confusion matrix
    val_predictions = classifier_model.predict(val_ds)
    val_predictions = np.argmax(val_predictions, axis=1)
    val_labels = np.argmax(val_labels, axis=1)

    cm = confusion_matrix(val_labels, val_predictions)

    #Show confusion matrix subset, the entire one is unreadable
    #Entire cm can be shown by just switching the variables out
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_subset = cm_normalized[:10, :10]
    sns.heatmap(cm_subset, annot=True, fmt='.2f', cmap='Blues')
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
        cluster_labels[class_indices] = clusters

    #Checkpoint to see how many clusters there are at a given time
    print(f"Current num-allowed-clusters: {num_allowed_clusters}")

    print("Checking Flag value, if flag is not 1, the loop restarts")
    if all(flag == 1 for flag in flags):
        print("Training completed with clustering and data augmentation.")
        print("Evaluating the model after clustering adjustments:")
        classifier_model.evaluate(test_ds)
        converged = True