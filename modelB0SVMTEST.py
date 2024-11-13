import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from keras._tf_keras.keras.applications import EfficientNetB0
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from pyclustering.cluster.xmeans import xmeans
from imblearn.over_sampling import GeometricSMOTE

# Paths to datasets
path_to_train_setbgr = pathlib.PosixPath("/home/jordan/Insect Pest Classification Dataset/classification/trainbgr")
path_to_val_setbgr = pathlib.PosixPath("/home/jordan/Insect Pest Classification Dataset/classification/valbgr")
path_to_test_setbgr = pathlib.PosixPath("/home/jordan/Insect Pest Classification Dataset/classification/testbgr")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 102

# Dataset loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    path_to_train_setbgr, label_mode="int", seed=1, batch_size=BATCH_SIZE, image_size=(IMG_SIZE, IMG_SIZE)
)

# Image augmentation layers
img_augmentation_layers = [
    layers.Normalization(axis=-1, mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225]),  # ImageNet means
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
    layers.RandomZoom(height_factor=0.1)
]


def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images


# Preprocessing function
def input_preprocess_train(image, label):
    image = img_augmentation(image)
    return image, label


# Apply preprocessing
train_ds = train_ds.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# Load EfficientNetB0 without the top classification layer
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Add global average pooling to convert features into a 1D vector
x = layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=x)

# Extract features from the training dataset
features_train = []
labels_train = []

for images, labels in train_ds:
    features = feature_extractor(images, training=False)
    features_train.append(features.numpy())
    labels_train.append(labels.numpy())

# Concatenate all features and labels
train_features = np.concatenate(features_train, axis=0)
train_labels = np.concatenate(labels_train, axis=0)


# Power mean function
# def power_mean(features, p):
#     return np.sign(features) * np.abs(features) ** p

# train_features = power_mean(train_features, p=-1)

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

# Train an SVM on the extracted features
base_svm = SVC()
OVR_SVM = OneVsRestClassifier(base_svm)

param_grid = {
    "estimator__C": [0.01],
    "estimator__kernel": ["rbf"],
    "estimator__gamma": ["scale"]
}

# Set up the GridSearchCV with the OvR classifier
grid_search = GridSearchCV(OVR_SVM, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=10)

# Fit the GridSearchCV
grid_search.fit(train_features, train_labels)

# Display best parameters and score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
