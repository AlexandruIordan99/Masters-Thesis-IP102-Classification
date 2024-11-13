from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import numpy as np
import tensorflow as tf
from keras import Model
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
from sklearn.decomposition import PCA

gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 8GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]) # Notice here
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


IMG_SIZE = 224
BATCH_SIZE = 32
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

def TripletLoss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
    negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), -1)

    loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(loss_1, 0.0))

    return loss


# Apply preprocessing
train_ds = train_ds.map(input_preprocess_train, num_parallel_calls=10)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(input_preprocess_val_and_test, num_parallel_calls=10)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)  #CHECK FUNCTION CALL 25.10.2024

test_ds = test_ds.map(input_preprocess_val_and_test, num_parallel_calls=10)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Define the feature extractor model using EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor_model = models.Model(inputs=base_model.input, outputs=x)



# Now, define three inputs for anchor, positive, and negative images
anchor_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
positive_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
negative_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Generate embeddings for anchor, positive, and negative
anchor_embedding = feature_extractor_model(anchor_input)
positive_embedding = feature_extractor_model(positive_input)
negative_embedding = feature_extractor_model(negative_input)

# Model that outputs all three embeddings
triplet_model = models.Model(inputs=[anchor_input, positive_input, negative_input],
                             outputs=[anchor_embedding, positive_embedding, negative_embedding])

# Compile the model with triplet loss
triplet_model.compile(optimizer=Adam(learning_rate=1e-4), loss=TripletLoss,
                         metrics=['accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])

print("Triplet model compiled.")

def create_triplet_batch(dataset, batch_size):
    anchor_images = []
    positive_images = []
    negative_images = []

    for images, labels in dataset:
        for i in range(batch_size):
            anchor_image = images[i]
            label = labels[i]

            # Find a positive sample (same class)
            positive_image = images[tf.random.categorical(tf.expand_dims(tf.cast(label, tf.float32), 0), 1)[0, 0]]

            # Find a negative sample (different class)
            negative_image = images[tf.random.categorical(tf.expand_dims(tf.cast(1 - label, tf.float32), 0), 1)[0, 0]]

            # Collect triplet samples
            anchor_images.append(anchor_image)
            positive_images.append(positive_image)
            negative_images.append(negative_image)

    return np.array(anchor_images), np.array(positive_images), np.array(negative_images)

@tf.function
def select_triplet_images(images, labels, i, label):
    images_on_gpu = tf.identity(images)  # Move images to GPU
    anchor_image = images_on_gpu[i]
    positive_image = images_on_gpu[tf.random.categorical(tf.expand_dims(tf.cast(label, tf.float32), 0), 1)[0, 0]]

    batch_size = tf.shape(images_on_gpu)[0]
    anchor_label = label

    negative_indices = tf.where(tf.not_equal(labels, anchor_label))[:, 0]
    num_negatives = tf.shape(negative_indices)[0]

    if num_negatives > 0:
        random_neg_index = tf.random.uniform(shape=(), maxval=num_negatives, dtype=tf.int32)
        negative_image = images_on_gpu[negative_indices[random_neg_index]]
    else:
        random_neg_index = tf.random.uniform(shape=(), maxval=batch_size, dtype=tf.int32)
        negative_image = images_on_gpu[random_neg_index]

    return anchor_image, positive_image, negative_image


# Adaptive clustering setup
num_allowed_clusters = [1] * NUM_CLASSES
flags = [0] * NUM_CLASSES

print("Beginning Clustering Loop.")
# Training loop with clustering and classifier training
converged = False
while not converged:
    train_anchor_images = []
    train_positive_images = []
    train_negative_images = []
    train_labels = []

    val_anchor_images = []
    val_positive_images = []
    val_negative_images = []
    val_labels = []

    loop_counter = 0

    # Extract triplet features for training images with augmentation
    for images, labels in train_ds:
        # Sample triplets (anchor, positive, negative)
        for i in range(BATCH_SIZE):
            anchor_image, positive_image, negative_image = select_triplet_images(images, labels, i, labels)
            label = labels[i]

            # Find a positive sample (same class)
            batch_size = tf.shape(images)[0]
            random_index = tf.random.uniform(shape=(), maxval=batch_size, dtype=tf.int32)

            # Find a negative sample (different class)
            negative_indices = tf.where(tf.not_equal(labels, label))[:, 0]
            num_negatives = tf.shape(negative_indices)[0]


            # Collect triplet samples
            train_anchor_images.append(anchor_image)
            train_positive_images.append(positive_image)
            train_negative_images.append(negative_image)
            train_labels.append(labels[i])  # Can still collect labels for analysis, not used in triplet loss

        loop_counter += 1
        print(f"Extracting train triplets, loop number at {loop_counter}.")

    # Convert to numpy arrays or tensors
    train_anchor_images = np.array(train_anchor_images)
    train_positive_images = np.array(train_positive_images)
    train_negative_images = np.array(train_negative_images)
    train_labels = np.array(train_labels)

    # Train the triplet model with triplet loss
    triplet_model.fit(
        [train_anchor_images, train_positive_images, train_negative_images],  # Inputs for triplet model
        train_labels,  # Dummy labels (not used by the triplet loss function)
        epochs=3,
        batch_size=BATCH_SIZE
    )

    # Extract triplet features for validation images without augmentation
    loop_counter = 0
    for images, labels in val_ds:
        # Sample triplets (anchor, positive, negative)
        for i in range(BATCH_SIZE):
            anchor_image, positive_image, negative_image = select_triplet_images(images, labels, i, labels)
            label = labels[i]

            # Find a positive sample (same class)
            batch_size = tf.shape(images)[0]
            random_index = tf.random.uniform(shape=(), maxval=batch_size, dtype=tf.int32)

            # Find a negative sample (different class)
            negative_indices = tf.where(tf.not_equal(labels, label))[:, 0]
            num_negatives = tf.shape(negative_indices)[0]


        loop_counter += 1
        print(f"Extracting validation triplets, loop number at {loop_counter}.")

    # Convert to numpy arrays or tensors
    val_anchor_images = np.array(val_anchor_images)
    val_positive_images = np.array(val_positive_images)
    val_negative_images = np.array(val_negative_images)
    val_labels = np.array(val_labels)

    # Predict using the triplet model (triplet embeddings)
    val_predictions = triplet_model.predict(
        [val_anchor_images, val_positive_images, val_negative_images]
    )

    # You can extract the embeddings from the model's output
    # Here we just use the embeddings from the triplet model for evaluation
    # Note: The model returns a list of embeddings, one for each input (anchor, positive, negative)
    anchor_embedding, positive_embedding, negative_embedding = val_predictions

    # You can evaluate these embeddings, compute distances, or perform clustering based on your needs
    # For simplicity, let's assume we want to compare the anchor and positive embeddings for accuracy
    distances = np.linalg.norm(anchor_embedding - positive_embedding, axis=1)

    # You might want to define a threshold for determining positive matches
    threshold = 0.5  # Example threshold, adjust as needed
    predicted_labels = (distances < threshold).astype(int)

    # Now you can calculate the accuracy or use the embeddings for clustering, etc.
    accuracy = np.mean(predicted_labels == val_labels)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Calculate confusion matrix if desired
    cm = confusion_matrix(val_labels, predicted_labels)
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


    # Clustering with K-Means for each class using triplet embeddings
clusters_per_class = {}
train_anchor_embeddings = []
train_loop_counter = 0

# Generate embeddings for each training sample using the anchor output of the triplet model
for images, labels in train_ds:
    train_loop_counter += 1
    # Get embeddings from the triplet model's anchor output
    embeddings = triplet_model.predict([images, images, images])[0]  # Only anchor embeddings
    train_anchor_embeddings.append(embeddings)
    print(f"Train embedding extraction loop counter: {train_loop_counter}")

# Concatenate all anchor embeddings into a single tensor
train_anchor_embeddings = tf.concat(train_anchor_embeddings, axis=0)

# Generate class-wise clusters using K-Means
kmeans_loop_counter = 0
for class_id in range(NUM_CLASSES):
    kmeans_loop_counter += 1
    # Find indices of samples belonging to the current class
    class_indices = tf.where(train_labels == class_id)[:, 0]
    class_features = tf.gather(train_anchor_embeddings, class_indices)

    if class_features.shape[0] > 0:
        # Apply K-Means clustering for the current class
        kmeans_instance = KMeans(n_clusters=num_allowed_clusters[class_id], random_state=42)
        kmeans_instance.fit(class_features)
        clusters_per_class[class_id] = kmeans_instance.labels_
    print(f"K-Means clustering loop counter for class {class_id}: {kmeans_loop_counter}")

# Convert one-hot encoded `train_labels` back to class indices for pseudo-label creation
train_labels_indices = np.argmax(train_labels, axis=1)

# Initialize array for cluster pseudo-labels
cluster_labels = np.full(train_labels_indices.shape, -1)
cluster_assignment_loop_counter = 0

# Assign cluster labels to samples according to the clusters_per_class results
for class_id, clusters in clusters_per_class.items():
    cluster_assignment_loop_counter += 1
    # Get indices of samples belonging to the current class
    class_indices = np.where(train_labels_indices == class_id)[0]
    # Assign clusters as pseudo-labels
    cluster_labels[class_indices] = clusters
    print(f"Cluster assignment loop counter for class {class_id}: {cluster_assignment_loop_counter}")

# Convert cluster labels to one-hot encoding for training
pseudo_labels = to_categorical(cluster_labels, num_classes=NUM_CLASSES)

# Check cluster counts and convergence status
print(f"Current num-allowed-clusters: {num_allowed_clusters}")

# Check if the convergence condition is met based on flags
print("Checking Flag value; if any flag is not set, the loop will continue.")
if all(flag == 1 for flag in flags):
    print("Training completed with clustering and data augmentation.")
    print("Evaluating the model after clustering adjustments:")

    # Evaluate the triplet model on test data (optional)
    test_anchor_embeddings = []
    test_labels = []
    test_loop_counter = 0

    # Generate test embeddings for evaluation
    for test_images, test_labels_batch in test_ds:
        test_loop_counter += 1
        embeddings = triplet_model.predict([test_images, test_images, test_images])[0]
        test_anchor_embeddings.append(embeddings)
        test_labels.append(test_labels_batch)
        print(f"Test embedding extraction loop counter: {test_loop_counter}")

    test_anchor_embeddings = tf.concat(test_anchor_embeddings, axis=0)
    test_labels = tf.concat(test_labels, axis=0)

    # Evaluate model performance (optional metric calculation here)
    print("Triplet model evaluation completed.")

    converged = True
