#!/bin/bash
#
# This script goes through the different datasets contained in the
# training data dir, and checks whether some training data is
# contained in more than one folder (which we don't want).
#
#######################################################################

ZOODIR=../data/1_zooplankton_0p5x/

echo "For each dataset, we search for duplicates in the other datasets"
for dataset in $(ls $ZOODIR/training/* -d)
do
    echo ""
    echo dataset: $dataset
    echo "The following images are also found in other datasets"
    for image in $(ls $dataset/*/training_data/*.jpeg)
    do
	for other_dataset in $(ls -d $ZOODIR/training/*| grep -v $dataset)
	do
	    other_image=$(echo $image | sed "s-$dataset-$other_dataset-")

	    # If the ls finds something, it is a duplicate
	    if [ $(ls $other_image 2>/dev/null) ]
	    then
		echo $image
		echo "-- $other_image"
	    fi
	    
	done
    done
		 
done

