#!/bin/bash
#####################################################
# This master script explains how to run everything #
#####################################################

# Spit out basic dataset properties
python analyze_dataset.py

# Train Multi-layer Perceptron model on images
python train_mlp.py -lr=0.1 -totEpochs=200 -width=64 -height=64 -plot
# Use it to make predictions
python predict_mlp.py '/path/to/unclassified/images/'

# Train Convolutional model on images

# Train Multi-layer Perceptron on features

# Train Multi-layer Perceptron on images+features

# Train Random Forest on features

