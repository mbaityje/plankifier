#!/bin/bash
#
# This script updates the dataset with the contents of the Q folder.
# Requires access to the Eawag intranet, and the right mounting point for Q.
#

QDIR=../../Q-AQUASCOPE # Root of the disk with the ECO plankton images
ZOODIR=../data/1_zooplankton_0p5x/ # Dir where we want to place the plankton images

# These are the two zooplankton dirs that we want to update
mkdir -p $ZOODIR/validation $ZOODIR/training


#####################################################
# Recovery of only the training images and features #
#####################################################
# Loops allow to change the directory structure in the desired way

#
# Loop over training datasets
#
for dataset in $(ls -d $QDIR/pictures/annotation_classifier/1_zooplankton_0p5x/training/*)
do
    echo $dataset
    # Loop over classes
    let tot=0 # Count how many images we have
    for classdir in $(ls -d  $dataset/*)
    do
	class=$(basename $classdir)
	dir_here=$ZOODIR/training/$(basename $dataset)/$class                                            # The local version of the folder on Q
	mkdir -p $dir_here/training_data                                                         # Create the folder in case it does not exist
	rsync --exclude 'Thumbs.db' -au $classdir/features.tsv $dir_here/                        # Update the features.tsv file
	rsync --exclude 'Thumbs.db' -au $classdir/training_data/*.jpeg $dir_here/training_data/  # Update the jpeg images
	if [ $(ls $classdir/training_data/*.jpg  2>/dev/null | wc -l) -gt 0 ]; then echo "THERE ARE SOME JPG FILES THAT HAVE NOT BEEN TRANSFERRED"; fi
	if [ $(ls $classdir/training_data/*.png  2>/dev/null | wc -l) -gt 0 ]; then echo "THERE ARE SOME PNG FILES THAT HAVE NOT BEEN TRANSFERRED"; fi

	n=$(ls $dir_here/training_data/*.jpeg 2>/dev/null |wc -l)
	let tot=$tot+$n
	printf "$class:\t$n\n"
    done
#    n=$(ls $dataset/*/training_data/*.jpeg | wc -l)
    printf "In total contains $tot jpeg images\n\n"
done



#######################
# Validation datasets #
#######################

# Validation folder
rsync -auvr --exclude 'Thumbs.db' $QDIR/pictures/annotation_classifier/1_zooplankton_0p5x/validation/zooplankton_validationset_????.??.?? $ZOODIR/validation/


# Counts folder
rsync -auvr --exclude 'Thumbs.db' $QDIR/pictures/annotation_classifier/1_zooplankton_0p5x/validation/counts $ZOODIR/validation/
countsdir=$QDIR/pictures/annotation_classifier/1_zooplankton_0p5x/validation/counts/
counts_rel_path=$(grep LINUX-PATH-REL $ZOODIR/validation/counts/path-to-directory.txt|cut -f2 -d' ')
ls $QDIR/pictures/annotation_classifier/1_zooplankton_0p5x/validation/counts//../../../../../lab/2020/annotation_workshop/pictures//year_2018/1530403202/0000000000_subset_static_html/images/00000/| grep -v rawcolor | grep -v binary
countsdir_here=$ZOODIR/validation/counts/

# Copy one by one to local the folders from which the counts were taken (excluding most of the rubbish that they contain)
# I should have used the rsync options better and done this in a line
for year in $(ls -d $countsdir/$counts_rel_path/*)
do

    echo $year
    for subdir in $(ls -d $year/*)
    do
	echo $subdir
	feat_path=$subdir/0000000000_subset_static_html/
	img_path=$feat_path/images/00000/
	feat_path_here=$countsdir_here/$(basename $year)/$(basename $subdir)/0000000000_subset_static_html
	img_path_here=$feat_path_here/images/00000/
	mkdir -p $img_path_here
	
	features=$feat_path/features.tsv	
	rsync -auvr --exclude 'Thumbs.db' $features $feat_path_here
	rsync -auvr --exclude 'Thumbs.db' --exclude '*rawcolor*' --exclude '*binary*' $img_path/* $img_path_here/ 
    done
done


