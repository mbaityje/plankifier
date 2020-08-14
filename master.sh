#!/bin/bash
########################################################
# This master script explains how to launch everything #
########################################################

# Spit out basic dataset properties
python analyze_dataset.py -datapath ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06

# Train Convolutional model on images, using only few classes (bosmina, hydra, dirt)
python train.py -datapaths ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06/ -outpath ./dummy_out -opt=adam -class_select bosmina hydra dirt -lr=0.001 -bs=32 -aug -model_image=conv2 -L 128 -datakind=image -ttkind=image -totEpochs=20 -earlyStopping=10

# Train Multi-layer Perceptron on features, using only few classes (bosmina, hydra, dirt)
python train.py -datapaths ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06/ -outpath ./dummy_out -opt=adam -class_select bosmina hydra dirt -lr=0.001 -bs=32 -aug -model_feat=mlp -layers 128 48 -L 128 -datakind=feat -ttkind=feat -totEpochs=20 -earlyStopping=10

# Train model on combination of features and images, using only few classes (bosmina, hydra, dirt)
python train.py -datapaths ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06/ -outpath ./dummy_out -opt=adam -class_select bosmina hydra dirt -lr=0.001 -bs=32 -aug -model_feat=mlp -model_image=conv2 -layers 128 48 -L 128 -datakind=mixed -ttkind=mixed -totEpochs=20 -earlyStopping=10







# Make predictions
python predictions.py
cat predict.txt
rm predict.txt

# Validation tests
cd val
bash validation.sh        # Validate on Tommy-validation
bash validation-counts.sh # Validate on populations (compare total counts, not single images)
cd ..
