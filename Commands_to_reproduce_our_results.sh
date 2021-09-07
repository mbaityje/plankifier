#!/bin/sh
export CUDA_VISIBLE_DEVICES=1

## 

for i in 0 1 2 3;
do
python train.py -datapaths ./data/zooplankton_0p5x/ -outpath ./trained-models/Aquascope_Mixed_models/ -classifier multi -aug -datakind image -ttkind image -save_data yes -resize_images 2 -L 128 -finetune 1 -valid_set yes -training_data True -hp_tuning yes -models_image eff0 eff1 eff2 eff3 eff4 eff5 eff6 eff7 incepv3 res50 dense121 mobile -max_trials 10 -executions_per_trial 1  -compute_extrafeat yes -stacking_ensemble yes -finetune_epochs 400 -Bayesian_epoch 100 -epochs 200 -balance_weight no -init_name Init_${i}
done