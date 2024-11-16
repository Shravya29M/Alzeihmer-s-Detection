# Alzheimer's Detection Using Deep Learning

This repository implements a hybrid deep learning model to classify Alzheimer's disease into four categories: **Mild Demented**, **Moderate Demented**, **Non Demented**, and **Very Mild Demented**. The model combines **DenseNet201** and **VGG16** architectures with advanced data augmentation techniques to achieve high accuracy.

## Overview

Key features of this project include:
- **Hybrid Architecture**: Combining DenseNet201 and VGG16 for feature extraction.
- **Advanced Data Augmentation**: Rotation, zooming, shifting, and flipping to improve model robustness.
- **Fine-Tuning**: Leveraging pre-trained weights and additional training for improved performance.

**Achieved Accuracy**: **95.15%**

---

## Project Structure

- **`alzheimer_main.py`**: Contains the complete implementation of the model.

- **`alzheimer_result.py`**: Contains the another version of the model.

---

## Dataset

The dataset has been taken from kaggle.

Each folder contains the corresponding class images.

If the dataset is compressed, you can unzip it using the following lines in the script:

-!unzip /path/to/MildDemented.zip
-!unzip /path/to/ModerateDemented.zip
 -!unzip /path/to/NonDemented.zip
 -!unzip /path/to/VeryMildDemented.zip

## Model Architecture

### 1. Data Preprocessing
- Splitting the dataset into training, validation, and test sets.
- Applying data augmentation with:
  - **Rotation range**: 40 degrees
  - **Zoom range**: 0.2
  - **Width and height shifts**: 0.2
  - **Shear**: 0.2
  - **Horizontal flipping**

### 2. Hybrid Model
- **DenseNet201**:
  - Pre-trained on ImageNet.
  - Used for feature extraction.
  - Global Average Pooling (GAP) and Batch Normalization layers added.
- **VGG16**:
  - Pre-trained on ImageNet.
  - Used for feature extraction.
  - GAP and Batch Normalization layers added.
- **Fusion**:
  - DenseNet201 and VGG16 outputs concatenated.
  - Fully connected layer with Dropout (50%).
  - Final layer with 4 output nodes and softmax activation for classification.

### 3. Training
- **Phase 1: Initial Training**
  - Freeze DenseNet201 and VGG16 layers.
  - Optimizer: Adam with exponential decay.
  - Learning rate: `0.001`.
- **Phase 2: Fine-Tuning**
  - Unfreeze all layers.
  - Optimizer: Adam with reduced learning rate (`1e-5`).

### 4. Evaluation
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

---

## Results

### Test Accuracy
- **95.15%**

### Performance Metrics
The model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### Training and Validation Curves
Accuracy and loss curves are plotted for both initial training and fine-tuning phases.



