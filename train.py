#!/usr/bin/env python3
# 
# Script that trains a model
# 
# 

# DA FARE
# 
# - Data preprocessing should be consistent
# - helper_data: make sure that images are always divided by 255
# - Scan (includendo architettura e data Augmentation)
# - implement logging
# - condizioni iniziali
# - binary classifier
# - learning rate schedule
# - maxout activation on the dense layer should reduce overfitting


###########
# IMPORTS #
###########

import os, sys, pathlib, glob, time, datetime, argparse, numpy as np, pandas as pd
import tensorflow.keras as keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_models as hm, helper_data as hd, helper_tts as htts
from PIL import Image
from sklearn.metrics import classification_report
# import pdb

from keras.callbacks import LearningRateScheduler
import pickle
from sklearn.preprocessing import StandardScaler
from joblib import dump
import joblib
from pathlib import Path

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


###############
# TRAIN CLASS #
###############

class Ctrain:
	def __init__(self, initMode='default', verbose=True):
		self.data=None
		self.trainSize=None
		self.testSize=None
		self.model=None
		self.opt=None
		self.verbose=False
		self.SetParameters(mode=initMode)

		return

	def SetParameters(self, mode='default'):
		''' default, from args'''
		if mode == 'default':
			self.ReadArgs(string=None)
		elif mode == 'args':
			self.ReadArgs(string=sys.argv[1:])
		else:
			print('Unknown parameter mode',mode)
			raise NotImplementedError
		return

	def ReadArgs(self, string=None):

		if string is None: 
			string=""

		parser = argparse.ArgumentParser(description='Train different models on plankton images')
        
		parser.add_argument('-datapaths', nargs='*', default=['./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/'], help="Directories with the data.")
		parser.add_argument('-outpath', default='./out/', help="directory where you want the output saved")
		parser.add_argument('-load_weights', default=None, help='Model weights that should be loaded.')
		parser.add_argument('-saveModelName', default='keras_model.h5', help='Name of the model when it is saved at the end of the run.')
		# User experience
		parser.add_argument('-verbose', action='store_true', help="Print many messages on screen.")
		parser.add_argument('-plot', action='store_true', help="Plot loss and accuracy during training once the run is over.")
		# Hyperparameters
		parser.add_argument('-opt', choices=['sgd','adam'], default='sgd', help="Choice of the minimization algorithm (sgd,adam)")
		parser.add_argument('-bs', type=int, default=32, help="Batch size")
		parser.add_argument('-lr', type=float, default=0.00005, help="Learning Rate")
		parser.add_argument('-aug', action='store_true', help="Perform data augmentation. Augmentation parameters are hard-coded.")
		parser.add_argument('-modelfile', default=None, help='The name of the file where a model is stored (to be loaded with keras.models.load_model() )')
		parser.add_argument('-model_image', choices=['conv2','smallvgg','mobile','eff0','eff1','eff2','eff3','eff4','eff5','eff6','eff7','incepv3','res50','dense121'], default=None, help='For mixed data models, tells what model to use for the image branch. For image models, it is the whole model')
		parser.add_argument('-model_feat', choices=['mlp'], default=None, help='For mixed data models, tells what model to use for the feature branch. For feat models, it is the whole model.')
		parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers for MLP")
		# Data
		parser.add_argument('-L', type=int, default=128, help="Images are resized to a square of LxL pixels")
		parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the test set")
		parser.add_argument('-class_select', nargs='*', default=None, help='List of classes to be looked at (put the class names one by one, separated by spaces). If None, all available classes are studied.')
		parser.add_argument('-classifier', choices=['binary','multi','versusall'], default='multi', help='Choose between "binary","multi","versusall" classifier')
		parser.add_argument('-lr_scheduler', choices=['yes','no'], default='no', help='Choose "yes" or "no" for scheduling learning rate')
		parser.add_argument('-balance_weight', choices=['yes','no'], default='no', help='Choose "yes" or "no" for balancing class weights for imbalance classes')
		parser.add_argument('-datakind', choices=['mixed','feat','image'], default=None, help="Which data to load: features, images, or both")
		parser.add_argument('-ttkind', choices=['mixed','feat','image'], default=None, help="Which data to use in the test and training sets: features, images, or both")
		parser.add_argument('-training_data', choices=['True','False'], default='True', help="This is to cope with the different directory structures")
		# Training time
		parser.add_argument('-totEpochs', type=int, default=5, help="Total number of epochs for the training")
		parser.add_argument('-initial_epoch', type=int, default=0, help='Initial epoch of the training')
		parser.add_argument('-earlyStopping', type=int, default=100, help='If >0, we do early stopping, and this number is the patience (how many epochs without improving)')
		# Preprocessing Images 
		parser.add_argument('-resize_images', type=int, default=1, help="Images are resized to a square of LxL pixels by keeping the initial image proportions if resize=1. If resize=2, then the proportions are not kept but resized to match the user defined dimension")

## For HyperParameter Tuning       

		# Model related
		parser.add_argument('-models_image', nargs='*', default='mobile', help='select models to train from: conv,mobile,eff0,eff1,eff2,eff3,eff4,eff5,eff6,eff7,incepv3,res50,dense121')
		# Ensemble related
		parser.add_argument('-stacking_ensemble', choices=['yes','no'], default='no', help='Choose "yes" or "no" for running Stacking ensemble')
		parser.add_argument('-avg_ensemble', choices=['yes','no'], default='no', help='Choose "yes" or "no" for running Average ensemble ')
		parser.add_argument('-finetune' , type=int, default=0, help='Choose "0" or "1" for finetuning') 
		parser.add_argument('-finetune_epochs', type=int, default=100, help="Total number of epochs for the funetune training")
# 		parser.add_argument('-finetuned' , type=int, default=0, help='Choose "0" or "1" for selecting finetune model--Used only for Mixed model/Average ensemble/Stacking Ensemble') 
		parser.add_argument('-hp_tuning', choices=['yes','no'], default=None, help="Whether to perform hyperparameter optimization")
		parser.add_argument('-max_trials', type=int, default=5, help="Total number of trials for hyperparameter optimization")
		parser.add_argument('-epochs', type=int, default=100, help="Total number of epochs for the training")
		parser.add_argument('-executions_per_trial', type=int, default=1, help="Total number of trials for each hyperparameter optimization trial")        
		parser.add_argument('-bayesian_epoch', type=int, default=50, help="Total number of epochs for the training")
        
		parser.add_argument('-saved_data', choices=['yes','no'], default=None, help="Whether to load saved data or not")  
		parser.add_argument('-save_data', choices=['yes','no'], default=None, help="Whether to save the data for later use or not")  
        
		parser.add_argument('-compute_extrafeat', choices=['yes','no'], default=None, help="Whether to compute extra features or not")  
		parser.add_argument('-only_ensemble', choices=['yes','no'], default=None, help="Whether to train only ensemble models")     
		parser.add_argument('-mixed_from_finetune', type=int, default=0, help="Whether to train mixed models using finetuned image and feature models")    
		parser.add_argument('-mixed_from_notune', type=int, default=0, help="Whether to train mixed models using Not tuned image and feature models")    
		parser.add_argument('-mixed_from_scratch', type=int, default=0, help="Whether to train mixed models using image and feature models from scratch")      
		parser.add_argument('-valid_set', choices=['yes','no'], default='no', help="Select to have validation set. Choose from Yes or No")
		parser.add_argument('-init_name', default='Initialization_01', help="directory name where you want the Best models to be saved")


		args=parser.parse_args(string)

		# Add a trailing / to the paths, just for safety
		for i,elem in enumerate(args.datapaths):
			args.datapaths[i] = elem +'/'
		
		args.outpath  = args.outpath +'/'
		args.training_data = True if args.training_data == 'True' else False

		self.ArgsCheck(args)
		self.params=args

		if self.verbose:
			print(args)

		return

	def ArgsCheck(self, args):
		''' Consistency checks for command line arguments '''
		if args.L<8:
			raise ValueError('Linear size of the images <8 pixels is too small to be wanted. Abort.')

		if args.initial_epoch>=args.totEpochs:
			print('The initial epoch is already equal or larger than the target number of epochs, so there is no need to do anything. Exiting...')
			raise SystemExit

		if args.initial_epoch<0:
			print('The initial epoch cannot be negative. Exiting...')
			raise SystemExit

		if args.totEpochs<0:
			print('The total number of epochs cannot be negative. Exiting...')
			raise SystemExit

		if args.verbose:
			ngpu=len(keras.backend.tensorflow_backend._get_available_gpus())
			print('We have {} GPUs'.format(ngpu))


		#
		# Set compatible flags for model and data types
		#

		# Set a default model when none are defined
		if args.model_image is None and args.model_feat is None and args.hp_tuning is None:
			args.model_image = 'conv2'
			args.datakind == 'image'
			args.ttkind = 'image'
			if self.verbose:
				print('No model was specified by the user, so we analyze images with {}'.format(args.model_image))
		
		elif args.model_image is None and args.hp_tuning is None:
			args.datakind == 'feat'
			args.ttkind = 'feat'

		elif args.model_feat is None and args.hp_tuning is None:
			args.datakind == 'image'
			args.ttkind = 'image'

		elif args.hp_tuning is None:
			args.datakind == 'mixed'
			args.ttkind = 'mixed'

		if args.ttkind != 'image' and args.aug==True: 
			print('User asked for data augmentation, but we set it to False, because we only do it for `image` models')
			args.aug=False
            
		if args.ttkind == 'image':
			args.compute_extrafeat='no'   
			print('User asked for computing extra features, but we set it to False, because we only do it for `mixed` models')
            
		return


	def CreateOutDir(self):
		''' Create a unique output directory, and put inside it a file with the simulation parameters '''
		pathlib.Path(self.params.outpath).mkdir(parents=True, exist_ok=True)
		self.WriteParams()
		return

	def WriteParams(self):
		''' Writes a txt file with the simulation parameters '''
		self.fsummary=open(self.params.outpath+'/params.txt','w')
		print(self.params, file=self.fsummary); 
		self.fsummary.flush()

		''' Writes the same simulation parameters in binary '''
		np.save(self.params.outpath+'/params.npy',self.params)
		return

	def UpdateParams(self, **kwargs):
		''' Updates the parameters given in kwargs, and updates params.txt'''
		self.paramsDict = vars(self.params)
		if kwargs is not None:
			for key, value in kwargs.items():
				self.paramsDict[key] = value
		self.CreateOutDir()
		self.WriteParams()

		return

	def LoadData(self, datapaths=None, L=None, class_select=-1,classifier = None,compute_extrafeat=None, resize_images=None, balance_weight=None, datakind=None, training_data=None):
		''' 
		Loads dataset using the function in the Cdata class.
		Acts differently in case it is the first time or not that the data is loaded

		The flag `training_data` is there because of how the taxonomists created the data directories. In the folders that I use for training there is an extra subfolder called `training_data`. This subfolder is absent, for example, in the validation directories.
		'''

		# Default values
		if 	   datapaths == None:    datapaths = self.params.datapaths
		if 			  L == None:             L = self.params.L
		if class_select == -1: 	  class_select = self.params.class_select # class_select==None has the explicit meaning of selecting all the classes
		if 	   classifier == None:      classifier = self.params.classifier
		if 	   compute_extrafeat == None:      compute_extrafeat = self.params.compute_extrafeat
		if 	   resize_images == None:      resize_images = self.params.resize_images
		if 	   balance_weight == None:      balance_weight = self.params.balance_weight
		if 	   datakind == None:      datakind = self.params.datakind
		if training_data== None: training_data = self.params.training_data

		# Initialize or Load Data Structure
		if self.data is None:
			self.data = hd.Cdata(datapaths, L, class_select,classifier,compute_extrafeat, resize_images,balance_weight, datakind, training_data=training_data)
		else:
			self.data.Load(datapaths, L, class_select , classifier,compute_extrafeat, resize_images,balance_weight, datakind, training_data=training_data)

		# Reset parameters	
		self.params.datapaths = self.data.datapath
		self.params.L        = self.data.L
		self.params.class_select = self.data.class_select
		self.params.datakind = self.data.kind
		self.params.classifier = self.data.classifier
		self.params.compute_extrafeat = self.data.compute_extrafeat
		self.params.balance_weight = self.data.balance_weight
		return

    
    
	def CreateTrainTestSets(self, ttkind=None, classifier=None, save_data=None, balance_weight=None, testSplit=None, valid_set=None,compute_extrafeat=None,random_state=12345):
		'''
		Creates train and test sets using the CtrainTestSet class
		'''

		# Set default value for ttkind
		if ttkind is None:
			ttkind = self.params.ttkind
		else:
			self.params.ttkind = ttkind

		# Set default value for testSplit
		if testSplit == None:
			testSplit = self.params.testSplit
		else:
			self.params.testSplit = testSplit

		if valid_set == None:
			valid_set = self.params.valid_set
		else:
			self.params.valid_set = valid_set
                        
		if classifier == None:
			classifier = self.params.classifier
            
		if save_data == None:
			save_data = self.params.save_data
            
		if balance_weight == None:
			balance_weight = self.params.balance_weight

		if compute_extrafeat == None:
			compute_extrafeat = self.params.compute_extrafeat   
            
		self.tt=htts.CTrainTestSet(self.data.X, self.data.y,self.data.filenames,
                                   ttkind=ttkind,classifier=classifier,balance_weight=balance_weight,
                                   testSplit=testSplit,valid_set=valid_set,
                                   compute_extrafeat=compute_extrafeat,random_state=random_state)
		self.params.ttkind=self.tt.ttkind

                
# To save the data
		if self.params.ttkind == 'mixed':
			scaler = StandardScaler()
			scaler.fit(self.tt.trainXfeat)
			dump(scaler,self.params.outpath +'/Features_scaler_used_for_MLP.joblib')
			self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
			self.tt.testXfeat = scaler.transform(self.tt.testXfeat)
			if self.params.valid_set=='yes':
				self.tt.valXfeat = scaler.transform(self.tt.valXfeat)
				Data=[self.tt.trainFilenames,self.tt.trainXimage,self.tt.trainY,
                      self.tt.testFilenames,self.tt.testXimage,self.tt.testY,
                      self.tt.valFilenames,self.tt.valXimage,self.tt.valY,
                      self.tt.trainXfeat,self.tt.testXfeat,self.tt.valXfeat]   
			elif self.params.valid_set=='no':   
				Data=[self.tt.trainFilenames,self.tt.trainXimage,self.tt.trainY,
                  self.tt.testFilenames,self.tt.testXimage,self.tt.testY,
                  self.tt.trainXfeat,self.tt.testXfeat]      
    
		elif self.params.ttkind == 'feat':        
			scaler = StandardScaler()
			scaler.fit(self.tt.trainX)
			dump(scaler,self.params.outpath +'/Features_scaler_used_for_MLP.joblib')
			self.tt.trainX = scaler.transform(self.tt.trainX)
			self.tt.testX = scaler.transform(self.tt.testX)
            
			if self.params.valid_set=='yes':
				self.tt.valXfeat = scaler.transform(self.tt.valXfeat)
				Data=[self.tt.trainFilenames,self.tt.trainX,self.tt.trainY,
                      self.tt.testFilenames,self.tt.testX,self.tt.testY,
                      self.tt.valFilenames,self.tt.valX,self.tt.valY]   
			elif self.params.valid_set=='no':
				Data=[self.tt.trainFilenames,self.tt.trainX,self.tt.trainY,
                  self.tt.testFilenames,self.tt.testX,self.tt.testY] 
            
		elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'no':
			if self.params.valid_set=='yes':
				Data=[self.tt.trainFilenames,self.tt.trainX,self.tt.trainY,
                      self.tt.testFilenames,self.tt.testX,self.tt.testY,
                      self.tt.valFilenames,self.tt.valX,self.tt.valY]  
			elif self.params.valid_set=='no' :
				Data=[self.tt.trainFilenames,self.tt.trainX,self.tt.trainY,
                      self.tt.testFilenames,self.tt.testX,self.tt.testY]       
                
		elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'yes':
			scaler = StandardScaler()
			scaler.fit(self.tt.trainXfeat)
			dump(scaler,self.params.outpath +'/Aqua_Features_scaler_used_for_MLP.joblib')
			self.tt.trainXfeat = scaler.transform(self.tt.trainXfeat)
			self.tt.testXfeat = scaler.transform(self.tt.testXfeat)
			if self.params.valid_set=='yes':
				self.tt.valXfeat = scaler.transform(self.tt.valXfeat)
                
				Data=[self.tt.trainFilenames,self.tt.trainXimage,self.tt.trainY,
                      self.tt.testFilenames,self.tt.testXimage,self.tt.testY,
                      self.tt.valFilenames,self.tt.valXimage,self.tt.valY,
                      self.tt.trainXfeat,self.tt.testXfeat,self.tt.valXfeat]   
			elif self.params.valid_set=='no':   
				Data=[self.tt.trainFilenames,self.tt.trainXimage,self.tt.trainY,
                  self.tt.testFilenames,self.tt.testXimage,self.tt.testY,
                  self.tt.trainXfeat,self.tt.testXfeat]   
                
		if self.params.save_data == 'yes': 
			with open(self.params.outpath+'/Data.pickle', 'wb') as a:
				pickle.dump(Data,a)  
                
		if self.params.balance_weight == 'yes': 
			with open(self.params.outpath+'/class_weights.pickle', 'wb') as cw:
				pickle.dump(self.tt.class_weights,cw)             

            # To Save classes and filenames
		np.save(self.params.outpath+'/classes.npy', self.tt.lb.classes_)
            
		Filenames_for_Ensemble=[self.tt.trainFilenames, self.tt.testFilenames]
		with open(self.params.outpath+'/Filenames_for_Ensemble_training.pickle', 'wb') as b:
			pickle.dump(Filenames_for_Ensemble,b)
                   
		return
                 

	def Train(self, train=True):
        
		if self.params.hp_tuning is None: # if hyperparameter tuning is not chosen

			# Save classes
			if train:
				np.save(self.params.outpath+'/classes.npy', self.tt.lb.classes_)

			# Callbacks
			checkpointer= keras.callbacks.ModelCheckpoint(filepath=self.params.outpath+'/bestweights.hdf5', monitor='val_loss', verbose=0, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy    
			logger= keras.callbacks.CSVLogger(self.params.outpath+'/epochs.log', separator=' ', append=False)
			callbacks=[checkpointer, logger]
        
			if self.params.earlyStopping>0:
				earlyStopping   = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.params.earlyStopping, restore_best_weights=True)
				callbacks.append(earlyStopping)
            
			if self.params.lr_scheduler=='yes':        
					# learning schedule callback
				lrate = keras.callbacks.LearningRateScheduler(hd.step_decay)    
				callbacks.append(lrate)

			self.aug = None if (self.params.aug == False) else ImageDataGenerator(
							rotation_range=90,
							vertical_flip=True,
							horizontal_flip=True,
							shear_range=10
							)

			self.trainParams=hm.CreateParams(
										layers 		= self.params.layers, 
										lr 			= self.params.lr,
										bs 			= self.params.bs,
										optimizer 	= self.params.opt,
										classifier 	= self.params.classifier,
										totEpochs 	= self.params.totEpochs,
										dropout 	= self.params.dropout,
										callbacks 	= callbacks,
										aug 		= self.aug,
										modelfile 	= self.params.modelfile,
										model_image = self.params.model_image,
										model_feat  = self.params.model_feat,
										load_weights 	= self.params.load_weights,
										initial_epoch	= self.params.initial_epoch,
										train=train
										)
                   
			if self.params.saved_data is None:
				# train the neural network
				start=time.time()
				# If train==False the wrapper will only load the model. The other branch is for mixed vs non-mixed models
				if train==False:
					trX, trY, teX, teY = (None, None, None, None)
				elif (self.params.ttkind == 'mixed'):
					trX, trY, teX, teY,class_weights = ([self.tt.trainXimage,self.tt.trainXfeat], self.tt.trainY, [self.tt.testXimage,self.tt.testXfeat], self.tt.testY,self.tt.class_weights)
				else:
# 					trX, trY, teX, teY = (self.tt.trainX, self.tt.trainY, self.tt.testX, self.tt.testY)
					trX, trY, teX, teY,class_weights = (self.tt.trainX, self.tt.trainY, self.tt.testX, self.tt.testY,self.tt.class_weights)
    
				wrapper = hm.CModelWrapper(trX, trY, teX, teY,class_weights, self.trainParams, numclasses=len(self.data.classes))     
    
			elif self.params.saved_data == 'yes':
				Data = pd.read_pickle(self.params.outpath+'/Data.pickle')
				classes = np.load(self.params.outpath+'/classes.npy')           
				if self.params.balance_weight == 'yes': 
					class_weights=pd.read_pickle(self.params.outpath+'/class_weights.pickle')                
                
				if self.params.valid_set=='no' :
					if self.params.ttkind == 'mixed':
						trainFilenames=Data[0]
						trainXimage=Data[1]
						trY=Data[2]
						testFilenames=Data[3]
						testXimage=Data[4]               
						teY=Data[5]              
						trainXfeat=Data[6]               
						testXfeat=Data[7]  
						trX=[trainXimage,trainXfeat]             
						teX=[testXimage,testXfeat]  
					elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'yes':
						trainFilenames=Data[0]
						trainXimage=Data[1]
						trY=Data[2]
						testFilenames=Data[3]
						testXimage=Data[4]               
						teY=Data[5]              
						trainXfeat=Data[6]               
						testXfeat=Data[7]  
						trX=[trainXimage,trainXfeat]             
						teX=[testXimage,testXfeat]                   
					else:
						trainFilenames=Data[0]
						trX=Data[1]
						trY=Data[2]
						testFilenames=Data[3]
						teX=Data[4]               
						teY=Data[5]  
                    
				elif self.params.valid_set=='yes':
					if self.params.ttkind == 'mixed':
						trainFilenames=Data[0]
						trainXimage=Data[1]
						trY=Data[2]
						testFilenames=Data[3]
						testXimage=Data[4]
						teY=Data[5]
						valFilenames=Data[6]
						valXimage=Data[7]
						veY=Data[8]
						trainXfeat=Data[9]
						testXfeat=Data[10]
						valXfeat=Data[11]
						trX=[trainXimage,trainXfeat]             
						teX=[testXimage,testXfeat]    
						veX=[valXimage,valXfeat]    
					elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'yes':
						trainFilenames=Data[0]
						trainXimage=Data[1]
						trY=Data[2]
						testFilenames=Data[3]
						testXimage=Data[4]
						teY=Data[5]
						valFilenames=Data[6]
						valXimage=Data[7]
						veY=Data[8]
						trainXfeat=Data[9]
						testXfeat=Data[10]
						valXfeat=Data[11]
						trX=[trainXimage,trainXfeat]             
						teX=[testXimage,testXfeat]    
						veX=[valXimage,valXfeat] 
					else:
						trainFilenames=Data[0]
						trX=Data[1]
						trY=Data[2]
						testFilenames=Data[3]
						teX=Data[4]               
						teY=Data[5]                   
						valFilenames=Data[6]
						veX=Data[7]               
						veY=Data[8]        
                      
				wrapper = hm.CModelWrapper(trX, trY, teX, teY,class_weights, self.trainParams, numclasses=len(classes))
        
# 			wrapper = hm.CModelWrapper(trX, trY, teX, teY, self.trainParams, numclasses=len(self.data.classes))
			self.model, self.history = wrapper.model, wrapper.history

			if train:
				trainingTime=time.time()-start
				print('Training took',trainingTime/60,'minutes')

				print('Saving the last model. These are not the best weights, they are the last ones. For the best weights use the callback output (bestweights.hdf5)]')
				self.SaveModel()
                
      
		elif self.params.hp_tuning=='yes': 
                # Select loss based on the classifier chosen
			if self.params.classifier == 'binary' or self.params.classifier == 'versusall':
				self.loss="binary_crossentropy"
			elif self.params.classifier == 'multi':
				self.loss="categorical_crossentropy"
			if self.params.saved_data is None:
				classes=self.tt.lb.classes_        
				if self.params.balance_weight == 'yes': 
					class_weights=self.tt.class_weights    
				else:
					class_weights=None 
				if self.params.valid_set=='no' :   
					if self.params.ttkind == 'mixed':
						trainFilenames=self.tt.trainFilenames
						trainXimage=self.tt.trainXimage
						trainY=self.tt.trainY
						testFilenames=self.tt.testFilenames
						testXimage=self.tt.testXimage              
						testY=self.tt.testY             
						trainXfeat=self.tt.trainXfeat             
						testXfeat=self.tt.testXfeat     
					elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'yes':
						trainFilenames=self.tt.trainFilenames
						trainXimage=self.tt.trainXimage
						trainY=self.tt.trainY
						testFilenames=self.tt.testFilenames
						testXimage=self.tt.testXimage              
						testY=self.tt.testY             
						trainXfeat=self.tt.trainXfeat             
						testXfeat=self.tt.testXfeat 
					else:
						trainFilenames=self.tt.trainFilenames
						trainX=self.tt.trainX
						trainY=self.tt.trainY
						testFilenames=self.tt.testFilenames
						testX=self.tt.testX              
						testY=self.tt.testY 
				elif self.params.valid_set=='yes' :
					if self.params.ttkind == 'mixed':
						trainFilenames=self.tt.trainFilenames
						trainXimage=self.tt.trainXimage
						trainY=self.tt.trainY
						testFilenames=self.tt.testFilenames
						testXimage=self.tt.testXimage              
						testY=self.tt.testY  
						valFilenames=self.tt.valFilenames
						valXimage=self.tt.valXimage              
						valY=self.tt.valY 
						trainXfeat=self.tt.trainXfeat             
						testXfeat=self.tt.testXfeat 
						valXfeat=self.tt.valXfeat 
					elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'yes':
						trainFilenames=self.tt.trainFilenames
						trainXimage=self.tt.trainXimage
						trainY=self.tt.trainY
						testFilenames=self.tt.testFilenames
						testXimage=self.tt.testXimage              
						testY=self.tt.testY  
						valFilenames=self.tt.valFilenames
						valXimage=self.tt.valXimage              
						valY=self.tt.valY 
						trainXfeat=self.tt.trainXfeat             
						testXfeat=self.tt.testXfeat 
						valXfeat=self.tt.valXfeat 
					else:
						trainFilenames=self.tt.trainFilenames
						trainX=self.tt.trainX
						trainY=self.tt.trainY
						testFilenames=self.tt.testFilenames
						testX=self.tt.testX              
						testY=self.tt.testY                
						valFilenames=self.tt.valFilenames
						valX=self.tt.valX              
						valY=self.tt.valY                      
                    
			elif self.params.saved_data == 'yes':
				Data = pd.read_pickle(self.params.outpath+'/Data.pickle')
				classes = np.load(self.params.outpath+'/classes.npy')   
				if self.params.balance_weight == 'yes': 
					class_weights=pd.read_pickle(self.params.outpath+'/class_weights.pickle')   
				else:
					class_weights=None 
				if self.params.valid_set=='yes':
					if self.params.ttkind == 'mixed':
						trainFilenames=Data[0]
						trainXimage=Data[1]
						trainY=Data[2]
						testFilenames=Data[3]
						testXimage=Data[4]
						testY=Data[5]
						valFilenames=Data[6]
						valXimage=Data[7]
						valY=Data[8]
						trainXfeat=Data[9]
						testXfeat=Data[10]
						valXfeat=Data[11]
# 						trX=[trainXimage,trainXfeat]             
# 						teX=[testXimage,testXfeat]    
# 						veX=[valXimage,valXfeat]  
					elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'yes':
						trainFilenames=Data[0]
						trainXimage=Data[1]
						trainY=Data[2]
						testFilenames=Data[3]
						testXimage=Data[4]
						testY=Data[5]
						valFilenames=Data[6]
						valXimage=Data[7]
						valY=Data[8]
						trainXfeat=Data[9]
						testXfeat=Data[10]
						valXfeat=Data[11]
					else:
						trainFilenames=Data[0]
						trainX=Data[1]
						trainY=Data[2]
						testFilenames=Data[3]
						testX=Data[4]               
						testY=Data[5]                   
						valFilenames=Data[6]
						valX=Data[7]               
						valY=Data[8]  
                        
				elif self.params.valid_set=='no':
					if self.params.ttkind == 'mixed':
						trainFilenames=Data[0]
						trainXimage=Data[1]
						trainY=Data[2]
						testFilenames=Data[3]
						testXimage=Data[4]
						testY=Data[5]
						trainXfeat=Data[6]
						testXfeat=Data[7]
# 						trX=[trainXimage,trainXfeat]             
# 						teX=[testXimage,testXfeat]    
					elif self.params.ttkind == 'image' and self.params.compute_extrafeat == 'yes':
						trainFilenames=Data[0]
						trainXimage=Data[1]
						trainY=Data[2]
						testFilenames=Data[3]
						testXimage=Data[4]
						testY=Data[5]
						trainXfeat=Data[6]
						testXfeat=Data[7]
# 						trX=[trainXimage,trainXfeat]             
# 						teX=[testXimage,testXfeat] 
					else:
						trainFilenames=Data[0]
						trainX=Data[1]
						trainY=Data[2]
						testFilenames=Data[3]
						testX=Data[4]               
						testY=Data[5]                                     
                        
			if self.params.valid_set=='no':  
				valXimage,valXfeat,valX,valY=[],[],[],[]

                # IMAGE ONLY
			if self.params.ttkind == 'image':
				if self.params.only_ensemble == 'no' or self.params.only_ensemble is None:
					for i in range(len(self.params.models_image)):
						hm.get_and_train_best_models(X_train=trainX,y_train=trainY,
                                                     X_test=testX, y_test=testY,
                                                     X_val=valX, y_val=valY,
                                                     outpath=self.params.outpath,
                                                     model_to_train=self.params.models_image[i],
                                                     epochs=self.params.epochs,aug=self.params.aug,
                                                     classes=classes,
                                                     bayesian_epoch=self.params.bayesian_epoch,
                                                     max_trials=self.params.max_trials,
                                                     executions_per_trial=self.params.executions_per_trial,
                                                     loss=self.loss, finetune=self.params.finetune,
                                                     ttkind=self.params.ttkind,Mixed=0,
                                                     finetune_epochs=self.params.finetune_epochs,
                                                     class_weight=class_weights,
                                                     valid_set=self.params.valid_set,
                                                     init_name=self.params.init_name) 

					if self.params.avg_ensemble=='yes':
						hm.avg_ensemble(X_test=testX, y_test=testY,
                                        X_val=valX, y_val=valY,
                                        classes=classes,
                                        models_image=self.params.models_image,
                                        outpath=self.params.outpath,
                                        finetune=self.params.finetune,
                                        Mixed=0,valid_set=self.params.valid_set,
                                        for_mixed=self.params.models_image,
                                        init_name=self.params.init_name)  
                
					if self.params.stacking_ensemble=='yes':
						hm.stacking_ensemble(X_train=trainX, y_train=trainY,
                                             X_test=testX, y_test=testY,
                                             X_val=valX, y_val=valY,
                                             classes=classes,
                                             models_image=self.params.models_image,
                                             outpath=self.params.outpath,
                                             finetune=self.params.finetune,
                                             valid_set=self.params.valid_set,
                                             for_mixed=self.params.models_image,
                                             init_name=self.params.init_name,Mixed=0)
                    
				elif self.params.only_ensemble == 'yes':
                    
					if self.params.avg_ensemble=='yes':
						hm.avg_ensemble(X_test=testX, y_test=testY,
                                        X_val=valX, y_val=valY,
                                        classes=classes,
                                        models_image=self.params.models_image,
                                        outpath=self.params.outpath,
                                        finetune=self.params.finetune,
                                        Mixed=0,valid_set=self.params.valid_set,
                                        for_mixed=self.params.models_image,
                                        init_name=self.params.init_name)  
                
					if self.params.stacking_ensemble=='yes':
						hm.stacking_ensemble(X_train=trainX, y_train=trainY,
                                             X_test=testX, y_test=testY,
                                             X_val=valX, y_val=valY,
                                             classes=classes,
                                             models_image=self.params.models_image,
                                             outpath=self.params.outpath,
                                             finetune=self.params.finetune,
                                             valid_set=self.params.valid_set,
                                             for_mixed=self.params.models_image,
                                             init_name=self.params.init_name,Mixed=0)
                        
             # FEATURE
			elif self.params.ttkind == 'feat':
				hm.get_and_train_best_models(X_train=trainX,y_train=trainY,
                                             X_test=testX, y_test=testY,
                                             X_val=valX, y_val=valY,
                                             outpath=self.params.outpath,
                                             model_to_train='mlp',
                                             epochs=self.params.epochs,aug=0,
                                             classes=classes,
                                             bayesian_epoch=self.params.bayesian_epoch,
                                             max_trials=self.params.max_trials,
                                             executions_per_trial=self.params.executions_per_trial,
                                             loss=self.loss, finetune=self.params.finetune,
                                             ttkind=self.params.ttkind,Mixed=0,
                                             finetune_epochs=self.params.finetune_epochs,
                                             class_weight=class_weights,
                                             valid_set=self.params.valid_set,
                                             init_name=self.params.init_name) 

             # MIXED
			elif self.params.ttkind == 'mixed':
				if self.params.only_ensemble == 'no' or self.params.only_ensemble is None:
######################### Mixed from Scratch ###########################  
					if self.params.mixed_from_scratch == 1:  
						X_train=[trainXimage,trainXfeat]
					# Concatenate Image and Feature     
						for i in range(len(self.params.models_image)):
# 							FeaturePath=self.params.outpath+'BestModelsFromBayesianSearch/mlp'
							members=hm.Select_Mixed_Model_from_scratch(X_train,classes,
                                                                       self.params.models_image[i])
							Mixed_model = hm.define_Mixed_model_from_scratch(members,len(classes),self.loss)
							Mixed_model_name='FromScratchMixed_'+self.params.models_image[i]+'_and_MLP'
                    # Compile feature and image models from scratch
							hm.compile_and_train(Mixed_model,self.params.outpath,
                                                 Mixed_model_name,1e-05,
                                                 [trainXimage,trainXfeat],trainY,
                                                 [testXimage,testXfeat],testY,
                                                 [valXimage,valXfeat],valY,
                                                 self.params.epochs,0,
                                                 self.loss,0,classes,self.params.ttkind,
                                                 Mixed=1,class_weight=class_weights,
                                                 valid_set=self.params.valid_set,
                                                 init_name=self.params.init_name)
                
						Mixed_models_name = list()         
						for i in range(len(self.params.models_image)):
							Name='FromScratchMixed_'+self.params.models_image[i]+'_and_MLP'
							Mixed_models_name.append(Name)

						if self.params.avg_ensemble=='yes':
							if len(Mixed_models_name) > 1:      
								hm.avg_ensemble(X_test=[testXimage,testXfeat],y_test=testY,
                                                X_val=[valXimage,valXfeat],y_val=valY,
                                                classes=classes,
                                                models_image=Mixed_models_name,
                                                outpath=self.params.outpath,finetune=0,
                                                Mixed=1,valid_set=self.params.valid_set,
                                                for_mixed=self.params.models_image,
                                                init_name=self.params.init_name)   
							else :
								print('For Ensemble, models selected should be greater than 1')

						if self.params.stacking_ensemble=='yes':
							if len(Mixed_models_name) > 1: 
								hm.stacking_ensemble(X_train=[trainXimage,trainXfeat], 
                                                     y_train=trainY,
                                                     X_test=[testXimage,testXfeat],
                                                     y_test=testY,
                                                     X_val=[valXimage,valXfeat],
                                                     y_val=valY,
                                                     classes=classes,
                                                     models_image=Mixed_models_name,
                                                     outpath=self.params.outpath,
                                                     finetune=0,valid_set=self.params.valid_set,
                                                     for_mixed=self.params.models_image,
                                                     init_name=self.params.init_name,Mixed=1)  
							else :
								print('For Ensemble, models selected should be greater than 1')
                 
					if self.params.mixed_from_notune == 1 or self.params.mixed_from_finetune == 1:
# 						print('Mixed from fine tune ==1')
# 						Xtrain=[trainXimage,trainXfeat]
    
                # Feature
						feature_filepath=self.params.outpath+'BestModelsFromBayesianSearch/'+ self.params.init_name +'/Feature/mlp'
						if os.path.isdir(feature_filepath) ==  False:           
							hm.get_and_train_best_models(X_train=trainXfeat,y_train=trainY,
                                                         X_test=testXfeat, y_test=testY,
                                                         X_val=valXfeat,y_val=valY,
                                                         outpath=self.params.outpath,
                                                         model_to_train='mlp',
                                                         epochs=1000,aug=0,
                                                         classes=classes,
                                                         bayesian_epoch=self.params.bayesian_epoch,
                                                         max_trials=self.params.max_trials,
                                                         executions_per_trial=self.params.executions_per_trial,
                                                         loss=self.loss, finetune=self.params.finetune,
                                                         ttkind=self.params.ttkind,Mixed=0,
                                                         finetune_epochs=self.params.finetune_epochs,
                                                         class_weight=class_weights,
                                                         valid_set=self.params.valid_set,
                                                         init_name=self.params.init_name) 

                # Image
						for i in range(len(self.params.models_image)):
							image_filepath=self.params.outpath+'BestModelsFromBayesianSearch/'+ self.params.init_name +'/Image/'+ self.params.models_image[i]
							if os.path.isdir(image_filepath) ==  False:           
								hm.get_and_train_best_models(X_train=trainXimage,y_train=trainY,
                                                             X_test=testXimage,y_test=testY,
                                                             X_val=valXimage,y_val=valY,
                                                             outpath=self.params.outpath,
                                                             model_to_train=self.params.models_image[i],
                                                             epochs=self.params.epochs,aug=1,
                                                             classes=classes,
                                                             bayesian_epoch=self.params.bayesian_epoch,
                                                             max_trials=self.params.max_trials,
                                                             executions_per_trial=self.params.executions_per_trial,
                                                             loss=self.loss, finetune=self.params.finetune,
                                                             ttkind=self.params.ttkind,Mixed=0,
                                                             finetune_epochs=self.params.finetune_epochs,
                                                             class_weight=class_weights,
                                                             valid_set=self.params.valid_set,
                                                             init_name=self.params.init_name)                
                        
					if self.params.mixed_from_notune == 1 :  
                        # Concatenate Image and Feature     
						for i in range(len(self.params.models_image)):
# 							FeaturePath=self.params.outpath+'BestModelsFromBayesianSearch/mlp'
							members=hm.Select_Mixed_Model(self.params.outpath,
                                                          self.params.models_image[i],
                                                          'mlp',finetune=0,Mixed=1)
							Mixed_model = hm.define_Mixed_model(members,len(classes),self.loss)
							Mixed_model_name='FromNotuneMixed_'+self.params.models_image[i]+'_and_MLP'
            
            # Using best trained feature and image models
							hm.compile_and_train(Mixed_model,self.params.outpath,
                                                 Mixed_model_name,1e-05,
                                                 [trainXimage,trainXfeat],trainY,
                                                 [testXimage,testXfeat],testY,
                                                 self.params.epochs,0,
                                                 self.loss,0,classes,self.params.ttkind,
                                                 Mixed=1,class_weight=class_weights,
                                                 valid_set=self.params.valid_set,
                                                 init_name=self.params.init_name)
                
						Mixed_models_name = list()         
						for i in range(len(self.params.models_image)):
							Name='FromNotuneMixed_'+self.params.models_image[i]+'_and_MLP'
							Mixed_models_name.append(Name)
                
						if self.params.avg_ensemble=='yes':
							if len(Mixed_models_name) > 1:      
								hm.avg_ensemble(X_test=[testXimage,testXfeat], y_test=testY,
                                                X_val=[valXimage,valXfeat], y_val=valY,
                                                classes=classes,
                                                models_image=Mixed_models_name,
                                                outpath=self.params.outpath,
                                                finetune=0,Mixed=1,valid_set=self.params.valid_set,
                                                for_mixed=self.params.models_image,
                                                init_name=self.params.init_name)  
                                ### If you want average ensemble for image as well, then uncomment this
# 								hm.avg_ensemble(X_test=testXimage, y_test=testY,
#                                                 X_val=valXimage, y_val=valY,
#                                                 classes=classes,
#                                                 models_image=self.params.models_image,
#                                                 outpath=self.params.outpath,
#                                                 finetune=0,Mixed=0,valid_set=self.params.valid_set,
#                                                 for_mixed=self.params.models_image,init_name=self.params.init_name)  
							else :
								print('For Ensemble, models selected should be greater than 1')

						if self.params.stacking_ensemble=='yes':
							if len(Mixed_models_name) > 1: 
								hm.stacking_ensemble(X_train=[trainXimage,trainXfeat],
                                                     y_train=trainY,
                                                     X_test=[testXimage,testXfeat],
                                                     y_test=testY,
                                                     X_val=[valXimage,valXfeat],
                                                     y_val=valY,
                                                     classes=classes,
                                                     models_image=Mixed_models_name,
                                                     outpath=self.params.outpath,
                                                     finetune=0,valid_set=self.params.valid_set,
                                                     for_mixed=self.params.models_image,
                                                     init_name=self.params.init_name,Mixed=1)  
                                
                                ### If you want ensemble for image as well, then uncomment this       
# 								hm.stacking_ensemble(X_train=trainXimage, y_train=trainY,
#                                                      X_test=testXimage, y_test=testY,
#                                                      X_val=valXimage, y_val=valY,
#                                                      classes=classes,
#                                                      models_image=self.params.models_image,
#                                                      outpath=self.params.outpath,
#                                                      finetune=0,valid_set=self.params.valid_set,
#                                                      for_mixed=self.params.models_image,init_name=self.params.init_name) 
							else :
								print('For Ensemble, models selected should be greater than 1')
                            
					if self.params.mixed_from_finetune == 1 : 
                        # Concatenate Image and Feature     
						for i in range(len(self.params.models_image)):
# 							FeaturePath=self.params.outpath+'BestModelsFromBayesianSearch/mlp'
							members=hm.Select_Mixed_Model(self.params.outpath,
                                                          self.params.models_image[i],
                                                          'mlp',self.params.finetune)
							Mixed_model = hm.define_Mixed_model(members,len(classes),self.loss)
							Mixed_model_name='FromFinetuneMixed_'+self.params.models_image[i]+'_and_MLP'
            
            # Using best trained feature and image models
							hm.compile_and_train(Mixed_model,self.params.outpath,
                                                 Mixed_model_name,1e-05,
                                                 [trainXimage,trainXfeat],trainY,
                                                 [testXimage,testXfeat],testY,
                                                 [valXimage,valXfeat],valY,
                                                 self.params.epochs,0,
                                                 self.loss,0,classes,self.params.ttkind,
                                                 Mixed=1,class_weight=class_weights,
                                                 valid_set=self.params.valid_set,
                                                 for_mixed=self.params.models_image,
                                                 init_name=self.params.init_name)
                
						Mixed_models_name = list()         
						for i in range(len(self.params.models_image)):
							Name='FromFinetuneMixed_'+self.params.models_image[i]+'_and_MLP'
							Mixed_models_name.append(Name)
                
						if self.params.avg_ensemble=='yes':
							if len(Mixed_models_name) > 1:      
								hm.avg_ensemble(X_test=[testXimage,testXfeat],y_test=testY,
                                                X_val=[valXimage,valXfeat],y_val=valY,
                                                classes=classes,
                                                models_image=Mixed_models_name,
                                                outpath=self.params.outpath,
                                                finetune=0,Mixed=1,valid_set=self.params.valid_set,
                                                for_mixed=self.params.models_image,
                                                init_name=self.params.init_name)  
                                
                                ### If you want ensemble for image as well, then uncomment this       
# 								hm.avg_ensemble(X_test=testXimage, y_test=testY,
#                                                 X_val=valXimage, y_val=valY,
#                                                 classes=classes,
#                                                 models_image=self.params.models_image,
#                                                 outpath=self.params.outpath,
#                                                 finetune=0,Mixed=0,valid_set=self.params.valid_set,
#                                                  for_mixed=self.params.models_image,init_name=self.params.init_name)  
							else :
								print('For Ensemble, models selected should be greater than 1')

						if self.params.stacking_ensemble=='yes':
							if len(Mixed_models_name) > 1: 
								hm.stacking_ensemble(X_train=[trainXimage,trainXfeat], 
                                                     y_train=trainY,
                                                     X_test=[testXimage,testXfeat],
                                                     y_test=testY,
                                                     X_val=[valXimage,valXfeat],
                                                     y_val=valY,
                                                     classes=classes,
                                                     models_image=Mixed_models_name,
                                                     outpath=self.params.outpath,
                                                     finetune=0,valid_set=self.params.valid_set,
                                                     for_mixed=self.params.models_image,
                                                     init_name=self.params.init_name,Mixed=1)  
                                
                                ### If you want ensemble for image as well, then uncomment this       
# 								hm.stacking_ensemble(X_train=trainXimage,y_train=trainY,
#                                                      X_test=testXimage, y_test=testY,
#                                                      X_val=valXimage, y_val=valY,
#                                                      classes=classes,
#                                                      models_image=self.params.models_image,
#                                                      outpath=self.params.outpath,
#                                                      finetune=0,valid_set=self.params.valid_set,
#                                                      for_mixed=self.params.models_image,init_name=self.params.init_name,Mixed=1) 
							else :
								print('For Ensemble, models selected should be greater than 1')

                            
				elif self.params.only_ensemble == 'yes':  
                                      
					if self.params.mixed_from_scratch == 1:                                     
						Mixed_models_name = list()         
						for i in range(len(self.params.models_image)):
							Name='FromScratchMixed_'+self.params.models_image[i]+'_and_MLP'
							Mixed_models_name.append(Name)
                
						if self.params.avg_ensemble=='yes':
							hm.avg_ensemble(X_test=[testXimage,testXfeat],y_test=testY,
                                            X_val=[valXimage,valXfeat],y_val=valY,
                                            classes=classes,
                                            models_image=Mixed_models_name,
                                            outpath=self.params.outpath,
                                            finetune=0,Mixed=1,valid_set=self.params.valid_set,
                                            for_mixed=self.params.models_image,
                                            init_name=self.params.init_name)  

						if self.params.stacking_ensemble=='yes':
							hm.stacking_ensemble(X_train=[trainXimage,trainXfeat],
                                                 y_train=trainY,
                                                 X_test=[testXimage,testXfeat],
                                                 y_test=testY,
                                                 X_val=[valXimage,valXfeat],
                                                 y_val=valY,
                                                 classes=classes,
                                                 models_image=Mixed_models_name,
                                                 outpath=self.params.outpath,
                                                 finetune=0,valid_set=self.params.valid_set,
                                                 for_mixed=self.params.models_image,
                                                 init_name=self.params.init_name,Mixed=1)  
                            
					if self.params.mixed_from_notune == 1 :  

						Mixed_models_name = list()         
						for i in range(len(self.params.models_image)):
							Name='FromNotuneMixed_'+self.params.models_image[i]+'_and_MLP'
							Mixed_models_name.append(Name)
                
						if self.params.avg_ensemble=='yes':
							if len(Mixed_models_name) > 1: 
								hm.avg_ensemble(X_test=[testXimage,testXfeat],y_test=testY,
                                                classes=classes,models_image=Mixed_models_name,
                                                outpath=self.params.outpath,finetune=0,Mixed=1,
                                                valid_set=self.params.valid_set,
                                                for_mixed=self.params.models_image,
                                                init_name=self.params.init_name)  

                                ### If you want ensemble for image as well, then uncomment this       
# 								hm.avg_ensemble(X_test=testXimage, y_test=testY,
#                                             X_val=valXimage, y_val=valY,
#                                             classes=classes,
#                                             models_image=self.params.models_image,
#                                             outpath=self.params.outpath,
#                                             finetune=0,Mixed=0,valid_set=self.params.valid_set,
#                                             for_mixed=self.params.models_image,init_name=self.params.init_name)  
							else:
									print('For Ensemble, models selected should be greater than 1')
                                
                                
						if self.params.stacking_ensemble=='yes':
							if len(Mixed_models_name) > 1: 
								hm.stacking_ensemble(X_train=[trainXimage,trainXfeat],
                                                     y_train=trainY,
                                                     X_test=[testXimage,testXfeat],
                                                     y_test=testY,
                                                     X_val=[valXimage,valXfeat],
                                                     y_val=valY,
                                                     classes=classes,
                                                     models_image=Mixed_models_name,
                                                     outpath=self.params.outpath,
                                                     finetune=0,valid_set=self.params.valid_set,
                                                     for_mixed=self.params.models_image,
                                                     init_name=self.params.init_name,Mixed=1) 
                                
                                ### If you want ensemble for image as well, then uncomment this       
# 								hm.stacking_ensemble(X_train=trainXimage, y_train=trainY,
#                                                      X_test=testXimage, y_test=testY,
#                                                      X_val=valXimage, y_val=valY,
#                                                      classes=classes,
#                                                      models_image=self.params.models_image,
#                                                      outpath=self.params.outpath,
#                                                      finetune=0,valid_set=self.params.valid_set,
#                                                      for_mixed=self.params.models_image,init_name=self.params.init_name,Mixed=1) 
							else:
								print('For Ensemble, models selected should be greater than 1')
          
					if self.params.mixed_from_finetune == 1 : 
						Mixed_models_name = list()         
						for i in range(len(self.params.models_image)):
							Name='FromFinetuneMixed_'+self.params.models_image[i]+'_and_MLP'
							Mixed_models_name.append(Name)
                
						if self.params.avg_ensemble=='yes':
							if len(Mixed_models_name) > 1: 
								hm.avg_ensemble(X_test=[testXimage,testXfeat],y_test=testY,
                                                X_val=[valXimage,valXfeat],y_val=valY,
                                                classes=classes,
                                                models_image=Mixed_models_name,
                                                outpath=self.params.outpath,
                                                finetune=0,Mixed=1,valid_set=self.params.valid_set,
                                                for_mixed=self.params.models_image,
                                                init_name=self.params.init_name)  
                                
                                ### If you want ensemble for image as well, then uncomment this       
# 								hm.avg_ensemble(X_test=testXimage,y_test=testY,
#                                                 X_val=valXimage,y_val=valY,
#                                                 classes=classes,
#                                                 models_image=self.params.models_image,
#                                                 outpath=self.params.outpath,
#                                                 finetune=0,Mixed=0,valid_set=self.params.valid_set,
#                                                 for_mixed=self.params.models_image,init_name=self.params.init_name)  
							else :
								print('For Ensemble, models selected should be greater than 1')
                                
                                
						if self.params.stacking_ensemble=='yes':
							if len(Mixed_models_name) > 1: 
								hm.stacking_ensemble(X_train=[trainXimage,trainXfeat],
                                                     y_train=trainY,
                                                     X_test=[testXimage,testXfeat],
                                                     y_test=testY,
                                                     X_val=[valXimage,valXfeat],
                                                     y_val=valY,
                                                     classes=classes,
                                                     models_image=Mixed_models_name,
                                                     outpath=self.params.outpath,
                                                     finetune=0,valid_set=self.params.valid_set,
                                                     for_mixed=self.params.models_image,
                                                     init_name=self.params.init_name,Mixed=1)  
                                
### If you want ensemble for image as well, then uncomment this       
# 								hm.stacking_ensemble(X_train=trainXimage,y_train=trainY,
#                                                      X_test=testXimage,y_test=testY,
#                                                      X_val=valXimage,y_val=valY,
#                                                      classes=classes,
#                                                      models_image=self.params.models_image,
#                                                      outpath=self.params.outpath,
#                                                      finetune=0,valid_set=self.params.valid_set,
#                                                      for_mixed=self.params.models_image,init_name=self.params.init_name,Mixed=1) 
							else :
								print('For Ensemble, models selected should be greater than 1')                               
		return                            


	def LoadModel(self, modelfile=None, bestweights=None):
		'''
		Loads model `modelfile` (.h5 file), and if present, loads weights form `bestweights` (.hdf5 file)
		'''
		
		# Model

		if modelfile is not  None:
			self.params.modelfile = modelfile

		self.model = keras.models.load_model(self.params.modelfile)


		# Best weights

		if bestweights is not None:
			self.params.bestweights = bestweights

		try:
			self.model.load_weights(self.params.bestweights)
		except:
			print('Did not load bestweights (file: {})'.format(self.params.bestweights))

		return


	def Report(self):
		predictions = self.Predict()
		clrep=classification_report(self.tt.testY.argmax(axis=1), predictions.argmax(axis=1), 
									target_names=self.tt.lb.classes_,
									labels = range(len(self.data.classes))
									)
		print(clrep)        
		return


	def Predict(self):

		bs = self.params.bs if (self.params.aug == False) else None
		testX = [self.tt.testXimage, self.tt.testXfeat] if (self.tt.ttkind=='mixed') else self.tt.testX

		predictions = self.model.predict(testX, batch_size=bs)

		return predictions


	def IdentifyWorsePrediction(self, predictions):
		''' Identify the easiest prediction and the worse mistake '''

		i_maxconf_right=-1; i_maxconf_wrong=-1
		maxconf_right  = 0; maxconf_wrong  = 0
		
		for i in range(len(self.tt.testY)):
			# Spot easiestly classified image (largest confidence, and correct classification)
			if self.tt.testY.argmax(axis=1)[i]==predictions.argmax(axis=1)[i]: # correct classification
				if predictions[i][predictions[i].argmax()]>maxconf_right: # if the confidence on this prediction is larger than the largest seen until now
					i_maxconf_right = i
					maxconf_right   = predictions[i][predictions[i].argmax()]
			# Spot biggest mistake (largest confidence, and incorrect classification)
			else: # wrong classification
				if predictions[i][predictions[i].argmax()]>maxconf_wrong:
					i_maxconf_wrong=i
					maxconf_wrong=predictions[i][predictions[i].argmax()]
		return i_maxconf_right, i_maxconf_wrong, maxconf_right, maxconf_wrong

	def Confidences(self, predictions):

		# Confidences of right and wrong predictions
		confidences = predictions.max(axis=1) # confidence of each prediction made by the classifier
		whether = np.array([1 if testY.argmax(axis=1)[i]==predictions.argmax(axis=1)[i] else 0 for i in range(len(predictions))]) #0 if wrong, 1 if right
		self.confidences_right = confidences[np.where(testY.argmax(axis=1)==predictions.argmax(axis=1))[0]]
		self.confidences_wrong = confidences[np.where(testY.argmax(axis=1)!=predictions.argmax(axis=1))[0]]

	def Abstention(self, thresholds=[0.999], print=True, save=True, plot=False):

		if thresholds is None:
			thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.98,0.99,0.995,0.997,0.999,0.9995,0.9999,0.99995,0.99999]

		accs,nconfident = np.ndarray(len(thresholds), dtype=np.float), np.ndarray(len(thresholds), dtype=np.int)
		for i,thres in enumerate(thresholds):
			confident     = np.where(confidences>thres)[0]
			nconfident[i] = len(confident)
			accs      [i] = whether[confident].sum()/nconfident[i] if nconfident[i]>0 else np.nan

	def Finalize(self):
		'''
		Clean-up operations for the end of a run
		'''

		# print("Training-time: {:} seconds".format(trainingTime), file=self.fsummary)
		self.fsummary.close()


	def SaveModel(self):

		self.model.save(self.params.outpath+'/'+self.params.saveModelName, overwrite=True, include_optimizer=True)
		return


if __name__=='__main__':
	print('\nRunning',sys.argv[0],sys.argv[1:])
	sim=Ctrain(initMode='args')
	sim.LoadData()
	sim.CreateOutDir()
	sim.CreateTrainTestSets()
	sim.Train()
# 	sim.Report()
	sim.Finalize()