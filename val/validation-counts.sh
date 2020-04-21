#!/bin/bash

#
# The count folders are folders in which the taxonomists counted the number of each species,
# but without labeling the images one by one. We can only compare the total population measured
# by the taxonomists, with that of the classifier. This is the ultimate task we will give to the classifier,
# so it is an excellent validation check.
#
#
HERE=$PWD
PROGSDIR=$PWD/..


cd $PROGSDIR
paths=$(ls -d data/1_zooplankton_0p5x/validation/counts/year_*/*/0000000000_subset_static_html/)

# In each data directory, make a file with the model's predictions
for path in $paths
do
	ls -d $path
	python predict.py -testdir $path/images/00000/
	cp predict.txt $path/
	ls $path/*
done


# Compare predictions with taxonomists' counts
cd $HERE
python validation-counts.py
