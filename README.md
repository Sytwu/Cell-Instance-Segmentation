# NYCU Computer Vision 2025 Spring HW3
StudentID: 111550159 \
Name: Li-Zhong Szu-Tu (司徒立中)

## Introduction
In this task, the dataset is designed for instance segmentation of cells, consisting of 209 training and validation images and 101 test images with four different categories. Each image in the training and validation sets is accompanied by masks for each class. In these masks, a value of zero represents the background, while non-zero values indicate individual cell instances, where the pixel values correspond to instance IDs. The objective is to perform instance segmentation by predicting both the class and precise mask for each cell instance. The performance is evaluated solely using mean Average Precision (mAP). \
\
In this assignment, only the Mask R-CNN model is allowed; however, it is acceptable to modify its backbone, neck (Region Proposal Network), and head. In my implementation, I use ConvNeXt with pretrained weights as the backbone, and adjust the maximum number of box proposals considered per image to retain more candidate detections during inference.\
\
The dataset can be downloaded Here!

## How to install
How to install dependences
```
conda env create -f environment.yml
conda activate env
```

## How to run
How to execute the code
```
# Training & Inference
python main.py
```
My model weights can be downloaded Here!

## Performance snapshot
A shapshot of the leaderboard
\
Last Update: 2025/05/07 01:35 a.m. (GMT+8)
