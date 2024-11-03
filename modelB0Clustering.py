from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import numpy as np
import tensorflow as tf
from keras._tf_keras import keras
from tensorflow.keras import layers, models
from keras._tf_keras.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from keras._tf_keras.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Paths to datasets
path_to_train_setbgr = pathlib.Path("/home/jordan/Insect Pest Classification Dataset/classification/trainbgr")
path_to_val_setbgr = pathlib.Path("/home/jordan/Insect Pest Classification Dataset/classification/valbgr")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 102
max_clusters = 10  # Maximum number of clusters for each class
threshold = 0.2  # Threshold for adjusting num-allowed-clusters

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_train_setbgr, label_mode="int", seed=1, batch_size=BATCH_SIZE, image_size=(IMG_SIZE, IMG_SIZE))
val_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_val_setbgr, label_mode="int", seed=1, batch_size=BATCH_SIZE, image_size=(IMG_SIZE, IMG_SIZE))

#Feature extractor
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor_model = models.Model(inputs=base_model.input, outputs=x)  # Model to extract features

#Classifier model
classifier_head = layers.Dense(NUM_CLASSES, activation='softmax')(x)
classifier_model = models.Model(inputs=base_model.input, outputs=classifier_head)
classifier_model.compile(optimizer=Adam(learning_rate=1e-4),
                         loss='categorical_crossentropy',
                         metrics=['accuracy',
                                  keras.metrics.AUC(), keras.metrics.Precision(),
                                  keras.metrics.Recall(), keras.metrics.F1Score])

print("Models compiled")

# Adaptive clustering setup
num_allowed_clusters = [1] * NUM_CLASSES  # Initialize number of clusters for each class
flags = [0] * NUM_CLASSES  # Flag for each class indicating whether max clusters have been reached

# Training loop with feature extraction and clustering
converged = False
while not converged:
    # Step 1: Extract features for training and validation images
    train_features = []
    train_labels = []
    train_loop_counter = 0
    val_features = []
    val_labels = []
    val_loop_counter = 0

    # Extract features for training images in batches
    for images, labels in train_ds:
        features = feature_extractor_model(images, training=False)  # Use the feature extractor
        train_features.append(features)
        train_labels.append(labels)

    # Concatenate train features and labels, checking for dimensional consistency
    train_features = tf.concat(train_features, axis=0)
    train_labels = tf.concat(train_labels, axis=0)

    # Debug: Print shapes to check alignment
    print(f"train_features shape: {train_features.shape}")
    print(f"train_labels shape: {train_labels.shape}")

    # Create tf.data.Dataset objects from features and labels in batches
    pseudo_labels = to_categorical(train_labels, NUM_CLASSES)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, pseudo_labels))
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Extract features for validation images in batches
    for images, labels in val_ds:
        features = feature_extractor_model(images, training=False)  # Use the feature extractor
        val_features.append(features)
        val_labels.append(labels)

    # Concatenate validation features and labels, checking for dimensional consistency
    val_features = tf.concat(val_features, axis=0)
    val_labels = tf.concat(val_labels, axis=0)

    # Debug: Print shapes to check alignment
    print(f"val_features shape: {val_features.shape}")
    print(f"val_labels shape: {val_labels.shape}")

    # Prepare validation dataset in batches
    val_pseudo_labels = to_categorical(val_labels, NUM_CLASSES)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_pseudo_labels))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Step 2: Train the classifier model
    # Convert labels to categorical for training
    classifier_model.fit(train_ds, validation_data=val_ds, epochs=3)

    # Step 3: Validate and calculate confusion matrix
    val_predictions = classifier_model.predict(val_ds)  # Use classifier model
    val_predictions = np.argmax(val_predictions, axis=1)
    cm = confusion_matrix(val_labels, val_predictions)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # Step 4: Adjust num-allowed-clusters based on confusion matrix
    for class_id in range(NUM_CLASSES):
        false_negatives = cm[class_id].sum() - cm[class_id][class_id]
        if false_negatives > threshold and flags[class_id] == 0:
            num_allowed_clusters[class_id] = min(num_allowed_clusters[class_id] + 1, max_clusters)
            if num_allowed_clusters[class_id] == max_clusters:
                flags[class_id] = 1  # Set flag to 1 if max clusters reached
        elif false_negatives > threshold and flags[class_id] == 1:
            num_allowed_clusters[class_id] = max(num_allowed_clusters[class_id] - 1, 1)
            if num_allowed_clusters[class_id] == 1:
                flags[class_id] = 0  # Reset flag to 0 if min clusters reached

    # Step 5: Cluster using K-Means
    clusters_per_class = {}
    for class_id in range(NUM_CLASSES):
        class_features = train_features[train_labels == class_id]

        # Ensure class_features is 2D
        if class_features.ndim > 2:
            class_features = class_features.reshape(-1, class_features.shape[-1])

        if len(class_features) > 0:
            # Create a KMeans instance with the allowed number of clusters
            kmeans_instance = KMeans(n_clusters=num_allowed_clusters[class_id], random_state=42)
            kmeans_instance.fit(class_features)
            clusters = kmeans_instance.labels_  # Get the cluster labels for each feature
            clusters_per_class[class_id] = clusters

    # Assign cluster labels to training data
    cluster_labels = np.full(train_labels.shape, -1)  # Initialize with -1 for unassigned clusters
    for class_id, clusters in clusters_per_class.items():
        for idx in range(len(clusters)):
            if idx < len(train_labels) and train_labels[idx] == class_id:
                cluster_labels[idx] = clusters[idx]  # Assign cluster ID

    # Convert cluster labels to categorical format for softmax classifier
    pseudo_labels = to_categorical(cluster_labels, num_classes=NUM_CLASSES)

    print(f"Current num-allowed-clusters: {num_allowed_clusters}")

    # Check for convergence
    if all(flag == 1 for flag in flags):
        converged = True

print("Training completed with the trained model.")
