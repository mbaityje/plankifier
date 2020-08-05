#!/bin/bash

#
# Uses predict.py to make predictions on the test data contained in validation.txt
#
# validation.txt contains a list of lines. Each line has a class name, and a directory with images from that class
#
   


HERE=$PWD
PROGSDIR=$PWD/..
OUTDIR=$HERE/pred_temp/
FILE_RESULTS=$OUTDIR/results.txt
models="out/trained-models/conv2_image_adam_aug_b8_lr1e-3_L128_t500/keras_model.h5 \
out/trained-models/conv2_image_sgd_aug_b32_lr5e-5_L128_t1000/keras_model.h5 \
out/trained-models/smallvgg_image_adam_aug_b32_lr1e-3_L128_t5000/keras_model.h5 \
out/trained-models/smallvgg_image_adam_aug_b32_lr5e-5_L128_t5000/keras_model.h5 \
out/trained-models/smallvgg_image_sgd_aug_b8_lr5e-6_L192_t5000/keras_model.h5 \
out/trained-models/smallvgg_image_sgd_aug_b32_lr5e-5_L128_t5000/keras_model.h5
"

# Loop over the directories contained in validation.txt
echo "class good tot ratio" > $FILE_RESULTS
while IFS= read -r line; do

    # Each line in validation.txt contains a class name, and a path to images of that class
    read class file <<<$line
    echo $class

    cd $PROGSDIR

    python predict.py -testdir=$file -outpath=$OUTDIR -predname=$class -modelfullname $models -em='leader' -absthres=0.8 2>/dev/null

    predfile=$(echo $OUTDIR/$class.txt)
    temp=$(awk -vclass=$class 'BEGIN{good=0; tot=0}{tot+=1.0; if(class==$2){good+=1;}}END{print good,tot,good/tot}' $predfile)

    echo $class $temp >> $FILE_RESULTS
    cd $HERE	
done < validation.txt 

# Make a classification report based on the outcome of the previous loop (results.txt)
cd $OUTDIR
python $HERE/class_report.py >> $FILE_RESULTS


