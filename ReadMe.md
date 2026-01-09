# Handwritten Digit Recognition using MNIST

This project implements a machine learning model to recognize handwritten digits (0–9) using the MNIST dataset. It demonstrates the core concepts of supervised learning and image classification using a Convolutional Neural Network (CNN).

## Project Overview

The system is trained on grayscale images of handwritten digits and learns to identify patterns such as edges, curves, and shapes. Once trained, the model can accurately predict the digit represented in an unseen image.

## Objectives

- Train a model to classify handwritten digit images into ten classes (0–9)
- Apply image preprocessing techniques such as normalization and reshaping
- Implement a Convolutional Neural Network for image classification
- Evaluate model performance using accuracy, loss curves, and confusion matrix

## Dataset

- **MNIST Dataset**
- 60,000 training images
- 10,000 test images
- Image size: 28×28 pixels
- Grayscale images

## Model Architecture

- Convolutional layers with ReLU activation
- Max pooling layers
- Fully connected dense layers
- Softmax output layer for multi-class classification

## Features

- Automatic loading and preprocessing of MNIST data
- CNN-based digit classification
- Model evaluation on unseen test data
- Visualization of confusion matrix and loss curves

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn
