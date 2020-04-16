#!/bin/bash

echo "For each dataset, we search for duplicates in the other datasets"
for dataset in $(ls training/* -d)
do
    echo dataset: $dataset
    echo "The following images are also found in other datasets"
    for image in $(ls $dataset/*/training_data/*.jpeg)
    do
	for other_dataset in $(ls -d training/*| grep -v $dataset)
	do
	    other_image=$(echo $image | sed "s-$dataset-$other_dataset-")

	    # If the ls finds something, it is a duplicate
	    ls $other_image 2>/dev/null
	done
    done
		 
done
		     
