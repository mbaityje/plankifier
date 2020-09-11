#!/bin/bash
########################################################
# This master script explains how to launch everything #
########################################################

cd ..

outdir_feat='./dummy_out/feat/'
outdir_images='./dummy_out/images/'
outdir_mixed='./dummy_out/mixed/'


#
# ANALYZE DATASET
#

# Spit out basic dataset properties
python analyze_dataset.py -datapath ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06



#
# TRAIN MODELS
#

# Train Convolutional model on IMAGES, using only few classes (bosmina, hydra, dirt)
python train.py -datapaths ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06/ -outpath $outdir_images -opt=adam -class_select bosmina hydra dirt -lr=0.001 -bs=32 -aug -model_image=conv2 -L 128 -datakind=image -ttkind=image -totEpochs=20 -earlyStopping=10

# Train Multi-layer Perceptron on FEATURES, using only few classes (bosmina, hydra, dirt)
python train.py -datapaths ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06/ -outpath $outdir_feat -opt=adam -class_select bosmina hydra dirt -lr=0.001 -bs=32 -aug -model_feat=mlp -layers 128 48 -L 128 -datakind=feat -ttkind=feat -totEpochs=20 -earlyStopping=10

# Train model on combination of features and images (MIXED), using only few classes (bosmina, hydra, dirt)
python train.py -datapaths ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06/ -outpath $outdir_mixed -opt=adam -class_select bosmina hydra dirt -lr=0.001 -bs=32 -aug -model_feat=mlp -model_image=conv2 -layers 128 48 -L 128 -datakind=mixed -ttkind=mixed -totEpochs=20 -earlyStopping=10




#
# MAKE PREDICTIONS 
# Predictions here will be poor, because we use the very poorly trained models of this script, and because they are trained on only few classes
#

# testdir: 			directory with the images whose class we want to predict
# modelfullname: 	names of the models that are used for the predictions
# thresholds: 		abstention thresholds
# ensMethods: 		ensembling methods
# predname:			path and name of the output file

#The following testdir only works for images, due to a wrong structure of the validation set
testdirs='./data/1_zooplankton_0p5x/validation/tommy_validation/images/bosmina/'
modelfullnames="$outdir_images/keras_model.h5"

python predict.py -testdirs $testdirs -modelfullname $modelfullnames -predname $outdir_images/predict


# Here, I want to show how to mix predictions from different models.
# In general it is enough to put more than one name in modelfullnames, but I
# want to test loading both mixed, feat and image models at the same time. This
# requires dealing with some path BS that it doesn't make sense to deal with until
# the ecologists figured out how to store the data properly.
testdirs='./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/bosmina/training_data/'
modelfullnames="$outdir_images/keras_model.h5"
python predict.py -testdirs $testdirs -modelfullname $modelfullnames -predname dummy_out/predictions/predict


# Predict with a better model
modelfullnames="trained-models/conv2/keras_model.h5"
testdirs='./data/1_zooplankton_0p5x/validation/tommy_validation/images/bosmina/ ./data/1_zooplankton_0p5x/validation/tommy_validation/images/uroglena/'
python predict.py -testdirs $testdirs -modelfullname $modelfullnames -predname $outdir_images/predict



#
# VALIDATION
#

# Validation tests

python validation.py 		# Validate on Tommy-validation
python validation-counts.py # Validate on populations (compare total counts, not single images)

