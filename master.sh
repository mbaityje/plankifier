#!/bin/bash
#####################################################
# This master script explains how to run everything #
#####################################################

# Spit out basic dataset properties
python analyze_dataset.py

# Train Convolutional model on images
python train_imgmodel.py -lr=0.0001 -totEpochs=5000 -width=128 -height=128 -datapath=./data/zooplankton_trainingset_15oct/ -model='conv2' -bs=32  -aug

# Train Multi-layer Perceptron model on images
python train_imgmodel.py -lr=0.001 -totEpochs=2000 -width=128 -height=128 -datapath=./data/zooplankton_trainingset_15oct/ -model='mlp' -bs=32  -aug

# Train Multi-layer Perceptron on features
python train_features_mlp.py -lr=0.01 -totEpochs=2000 -layers 256 128 -datapath=./data/zooplankton_trainingset_15oct/ -bs=32 -plot


