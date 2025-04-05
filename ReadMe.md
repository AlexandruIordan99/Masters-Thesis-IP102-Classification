**Overview**

This repository contains code for a Master's Thesis on Pest Insect Classification using the IP102 dataset.
The paper focuses on improving classification performance on this dataset by addressing two of its major issues,
namely class imbalance and intra-class variance. The former is addressed through the use of data augmentation,
and transfer-learning using ImageNet weights, while the latter is addressed through the use of a clustering
algorithm. These techniques are implemented using EfficientNet models across versions 1 and 2.

In the main folder, the Dataset_to_BGR lets users quickly change and save the IP102 dataset in a BGR format.
The F1Calculator lets users calculate F1-Scores using the accuracy and recall metrics from the model evaluation
output. Finally, geometric_smote is a local copy of the code from https://github.com/georgedouzas/imbalanced-learn-extra.
This local copy was necessary because the geometric_smote import was not compatible with Python 3.12.6. The authors are credited
at the top of that file.


**To run:**

Use a Linux machine 

Install up to date machine learning drivers from NVIDIA

Prefferably use at least an NVIDIA RTX 4080 (10GB of RAM). Otherwise many of the models will stop working due to not enough VRAM.
If you understandably do not own such an expensive graphics card, lower the batch size of the model. This will however lower its accuracy.

Clone the repo

Obtain the dataset from [its authors or from ](https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset). This is linked instead of the authors page because their google drive link is broken.

Update the paths to the locations of your respective training, validation and testing sets.


