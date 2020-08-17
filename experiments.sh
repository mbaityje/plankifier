

# EXPERIMENT 1

datapaths="./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06/"
opt='adam'
lr=0.001
bs=8
L=192
model_feat=''
model_image='conv2'
ttkind='image'
datakind='image'
totEpochs=5000
earlyStopping=1000
outpath="out/experiments/set1/${ttkind}_${model_image}${model_feat}_${opt}_lr${lr}bs${bs}L${L}totEp${totEpochs}es${earlyStopping}"

echo $outpath

mkdir -p $outpath
python train.py -datapaths $datapaths -outpath $outpath -opt=$opt -lr=$lr -bs=$bs -aug -model_image=$model_image -L $L -datakind=$datakind -ttkind=$ttkind -totEpochs=$totEpochs -earlyStopping=$earlyStopping

