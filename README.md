# CHEST_XRAY_PNUMONIA_COVID_CLASSIFICATION
Pneumonia Classification from Chest X-ray Images using CNNs and PyTorch.
Project Description

This project focuses on automatic detection of pneumonia in  chest X-ray (CXR) images using Convolutional Neural Networks (CNNs) implemented in PyTorch. The dataset consists of labeled images divided into Normal , Pneumonia and Covid classes. Advanced data augmentation techniques are applied to improve model generalization.

The main goal is to provide a reliable AI-based tool for early diagnosis of pneumonia in Adults, which can be especially useful in low-resource healthcare settings.
Source: https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images
classes :
-Pnemonia : 1800 images
-Covid : 1626 images
-Normal: 1802 images

Technologies & Libraries:

-Deep Learning: PyTorch

-Data Handling & Augmentation: torchvision, PIL, numpy

-Visualization: matplotlib

-Others: tqdm, random, os

Features:

Custom CNN for  pneumonia classification

Data augmentation including grayscale conversion, random cropping, rotation, flipping, and Gaussian blur

Training and validation loops with accuracy and loss tracking

Model checkpointing to save the best performing model

Visual evaluation of predictions on validation/test images

Results

-Achieved 96% accuracy on the validation dataset

-Loss and accuracy curves are plotted to monitor training

-Sample predictions with true vs predicted labels are visualized

Future Work:

-Implement transfer learning with ResNet or EfficientNet for better performance

-Add explainable AI features like Grad-CAM for model interpretability

-Extend to multi-class pneumonia detection including COVID-19
