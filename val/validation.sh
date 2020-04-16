#!/bin/bash

#
# Uses predict.py to make predictions on thee test data contained in validation.txt
#
# validation.txt contains a list of lines. Each line has a class name, and a directory with images from that class
#
   

FILE_RESULTS=results.txt
HERE=$PWD
PROGSDIR=$PWD/..


echo "class good tot ratio" > $FILE_RESULTS
while IFS= read -r line; do

    # Each line in validation.txt contains a class name, and a path to images of that class
    read class file <<<$line
    cd $PROGSDIR
    preds=$(python predict.py -testdir=$file | grep .jpeg | awk '{print $2}')

    temp=$(echo "$preds" | awk -vclass=$class 'BEGIN{good=0; tot=0}{tot+=1.0; if(class==$1) {good+=1;}}END{print good,tot,good/tot}')
#    good=$(echo $temp | cut -d' ' -f1)
#    tot=$(echo $temp | cut -d' ' -f2)
#    ratio=$(echo $temp |cut -d' ' -f3)
    echo $class $temp

    cd $HERE
done < validation.txt >> $FILE_RESULTS

python class_report.py
