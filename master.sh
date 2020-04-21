#!/bin/bash
#####################################################
# This master script explains how to run everything #
#####################################################

# Spit out basic dataset properties
python analyze_dataset.py -datapath='./data/2020.02.02_zooplankton_trainingset_EWA/' -kind='mixed'

# Train Convolutional model on images
python train_imgmodel.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/2020.02.02_zooplankton_trainingset_EWA/' -datakind='mixed'
python train_imgmodel.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/2019.11.20_zooplankton_trainingset_15oct_TOM/' -datakind='image'

# Train Multi-layer Perceptron model on images
python train_imgmodel.py -lr=0.001 -totEpochs=2000 -width=128 -height=128 -datapath=./data/2019.11.20_zooplankton_trainingset_15oct_TOM/ -model='mlp' -bs=32  -aug

# Train Multi-layer Perceptron on features
python train_features_mlp.py -lr=0.01 -totEpochs=2000 -layers 256 128 -bs=32 -plot

# Train model on combination of features and images
python train_mixed.py -totEpochs=500 -width=128 -height=128 -model=mlp -resize=keep_proportions -bs=16 -lr=0.0001 -opt=sgd -datapath='./data/2019.11.20_zooplankton_trainingset_15oct_TOM/'







# Make predictions
python predictions.py
cat predict.txt
rm predict.txt

# Validation tests
cd val
bash validation.sh        # Validate on Tommy-validation
bash validation-counts.sh # Validate on populations (compare total counts, not single images)
cd ..
