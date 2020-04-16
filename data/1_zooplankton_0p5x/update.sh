#!/bin/bash
#
# This script updates the dataset with the contents of the Q folder.
# Requires access to the Eawag intranet, and the right mounting point for Q.
#

# These are the two dirs that we want to update
mkdir -p validation training

#############################################################################
# Full recovery of the directory (UNDESIRED, since includes backups folder) #
#############################################################################

# rsync -auvr ../../../Q-AQUASCOPE/pictures/annotation_classifier/1_zooplankton_0p5x/* ./


###########################################################################################################
# Full recovery of training and validation folders (UNDESIRED at least now, since it includes raw images) #
###########################################################################################################

#rsync -auvr ../../../Q-AQUASCOPE/pictures/annotation_classifier/1_zooplankton_0p5x/training/* ./training/
#rsync -auvr ../../../Q-AQUASCOPE/pictures/annotation_classifier/1_zooplankton_0p5x/validation/* ./validation/


#####################################################
# Recovery of only the training images and features #
#####################################################
#exit

#
# Loop over training datasets
#
for dataset in $(ls -d ../../../Q-AQUASCOPE/pictures/annotation_classifier/1_zooplankton_0p5x/training/*)
do
    echo $dataset
    # Loop over classes
    let tot=0 # Count how many images we have
    for classdir in $(ls -d  $dataset/*)
    do
	class=$(basename $classdir)
	dir_here=training/$(basename $dataset)/$class                        # The local version of the folder on Q
	mkdir -p $dir_here/training_data                                     # Create the folder in case it does not exist
	rsync -au $classdir/features.tsv $dir_here/                        # Update the features.tsv file
	rsync -au $classdir/training_data/*.jpeg $dir_here/training_data/  # Update the jpeg images
	if [ $(ls $classdir/training_data/*.jpg  2>/dev/null | wc -l) -gt 0 ]; then echo "THERE ARE SOME JPG FILES THAT HAVE NOT BEEN TRANSFERRED"; fi
	if [ $(ls $classdir/training_data/*.png  2>/dev/null | wc -l) -gt 0 ]; then echo "THERE ARE SOME PNG FILES THAT HAVE NOT BEEN TRANSFERRED"; fi

	n=$(ls $dir_here/training_data/*.jpeg 2>/dev/null |wc -l)
	let tot=$tot+$n
	printf "$class:\t$n\n"
    done
#    n=$(ls $dataset/*/training_data/*.jpeg | wc -l)
    printf "In total contains $tot jpeg images\n\n"
done

#
# Loop over validation datasets
#

# Validation datasets are small, so we just do it brute force
rsync -auvr ../../../Q-AQUASCOPE/pictures/annotation_classifier/1_zooplankton_0p5x/validation/* ./validation/

