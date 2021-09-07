# Imports
'''
Loads trained ensemble of models and generates prediction on custom data.

Launch as:
	# python predict_ensemble.py  -testdir './data/zooplankton_0p5x_all/validation/counts/year_2018/15304*/0000000000_subset_static_html/' -model_path '/local/kyathasr/plankifier/trained-models/Aquascope_corrected_Data_with_Mixed/' -models_image eff2 mobile -init_names Iteration_01 Iteration_02 -path_to_save './out/Ensemble_prediction3' -ens_type 1

The program can predict using either avg_ensemble, or stack ensemble or both ensemble together. Choose ens_type=1 for average ensemble, ens_type=2 for stack_ensemble and ens_type=3 for both average and stack ensemble

'''


import os, sys, pathlib, glob, time, datetime, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_models as hm, helper_data as hd, helper_tts as htts
from PIL import Image
import numpy as np, pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def predict_ensemble(testdir,model_path,models_image,
                     init_names,ens_type,stack_path,path_to_save,ttkind):
    
    model_path=''.join(model_path)
    testdir=''.join(testdir)
    stack_path=''.join(stack_path)
    path_to_save=''.join(path_to_save)
    
    ## Get the parameters from the trained model files
    params=np.load(model_path+'/params.npy', allow_pickle=True).item()
    classes = np.load(model_path+'/classes.npy')
    L=params.L
    resize_images=params.resize_images
    finetune=params.finetune
    
#     ttkind=params.ttkind
    
    if ttkind=='mixed':
        Mixed=1
    else:
        Mixed=0

    compute_extrafeat=params.compute_extrafeat
    mixed_from_finetune=params.mixed_from_finetune
    mixed_from_notune=params.mixed_from_notune
    mixed_from_scratch=params.mixed_from_scratch

    
    if ttkind=='mixed':
        alsoImages= True
        finetune=0
        test_features	= glob.glob(testdir+'/*.tsv')
        df= hd.LoadMixedData(test_features,L,resize_images,
                             alsoImages,compute_extrafeat)
        
        im_col = [i for i,col in enumerate(df.columns) if col == 'npimage'][0]
        X_image=np.array([df.to_numpy()[i, im_col] for i in range( len(df.index) )])

        Xf=df.drop(columns=['npimage','rescaled','classname','url','filename','file_size','timestamp'], errors='ignore')
        X_feat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
        
        im_names=df['filename']
        
        for_mixed=models_image
        Data = [X_image, X_feat]
        
        ###### Prepare for models_image ######
        model_names = list()

        if mixed_from_finetune == 1: 
            for i in range(len(models_image)):
                Mixed_model_name1='FromFinetuneMixed_'+models_image[i]+'_and_MLP'
                model_names.append(Mixed_model_name1)
        elif mixed_from_notune == 1: 
            for i in range(len(models_image)):
                Mixed_model_name1='FromNotuneMixed_'+models_image[i]+'_and_MLP'
                model_names.append(Mixed_model_name1)
        elif mixed_from_scratch == 1: 
            for i in range(len(models_image)):
                Mixed_model_name1='FromScratchMixed_'+models_image[i]+'_and_MLP'
                model_names.append(Mixed_model_name1)
        
    else:
        im_names= glob.glob(testdir+'/*.jp*g')
        Data = hd.LoadImageList(im_names,L,resize_images, show=False)
        
        model_names=models_image
        

    if ens_type==1:
        Avg_Probs_and_predictions = hm.avg_ensemble_selected_on_unlabelled(im_names,Data,classes,model_names,
                                                                      model_path,finetune,
                                                                      Mixed,models_image,
                                                                      init_names,path_to_save)
    elif ens_type==2:
        Stack_predictions = hm.stacking_ensemble_selected_on_unlabelled(im_names,Data,classes,model_names,
                                                                   model_path,finetune,Mixed,
                                                                   init_names,stack_path,
                                                                   for_mixed,path_to_save)
    elif ens_type==3:
        Avg_Probs_and_predictions = hm.avg_ensemble_selected_on_unlabelled(im_names,Data,classes,model_names,
                                                                      model_path,finetune,
                                                                      Mixed,models_image,
                                                                      init_names,path_to_save)
        
        Stack_predictions = hm.stacking_ensemble_selected_on_unlabelled(im_names,Data,classes,model_names,
                                                                   model_path,finetune,Mixed,
                                                                   init_names,stack_path,
                                                                   for_mixed,path_to_save)
        
if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Load ensemble model and use it to make predictions on images')
	parser.add_argument('-testdir', nargs='*', default=['./data/zooplankton_0p5x_all/validation/counts/year_2018/15304*/0000000000_subset_static_html/'], help="Directories with the data to be checked.")
	parser.add_argument('-model_path', nargs='*', default=['/local/kyathasr/plankifier/trained-models/Aquascope_corrected_Data_with_Mixed/'], help='main directory path of the trained models')
	parser.add_argument('-models_image', nargs='*', default=['eff2','eff7','incepv3','mobile','eff3','dense121'], help='select the models that were trained and of interest from these: conv,mobile,eff0,eff1,eff2,eff3,eff4,eff5,eff6,eff7,incepv3,res50,dense121')
	parser.add_argument('-init_names', nargs='*', default=['Iteration_01','Iteration_02','Iteration_02','Iteration_02','Iteration_03','Iteration_04'], help='Initializations names corresponding to the selected models')
	parser.add_argument('-path_to_save', default=['/local/kyathasr/For_paper/out/Ensemble_prediction4'], help="directory where you want the output saved")
	parser.add_argument('-stack_path', default=['/local/kyathasr/plankifier/trained-models/Aquascope_corrected_Data_with_Mixed/'\
'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/Mixed/Stacking_Ensemble/'\
'Ens_of_eff2_eff7_incepv3_mobile_eff3_dense121/'], help="path of the saved stacking ensemble")
	parser.add_argument('-ens_type', type=int, default=1, help="Choose ens_type=1 for average ensemble, ens_type=2 for stack_ensemble and ens_type=3 for both average and stack ensemble")
	parser.add_argument('-ttkind', default=['image'], help="select either 'mixed' or 'image' models")
    
	args=parser.parse_args()


	predict_ensemble(testdir=args.testdir,model_path=args.model_path,models_image=args.models_image,
                     init_names=args.init_names,stack_path=args.stack_path,path_to_save=args.path_to_save,
                     ens_type=args.ens_type, ttkind=args.ttkind)
    


