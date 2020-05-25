#!/usr/bin/env python3
# 
# Script that trains a model
# 
# 

# DA FARE
# 
# - Scan (includendo architettura e data Augmentation)
# - implement logging
# - condizioni iniziali
# - binary classifier
# - learning rate schedule
# - 


###########
# IMPORTS #
###########

import os, sys, pathlib, glob, time, datetime, argparse, numpy as np, pandas as pd
import keras
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



###############
# TRAIN CLASS #
###############

class Ctrain:
	def __init__(self, initMode='default'):
		self.data=None
		self.trainSize=None
		self.testSize=None
		self.model=None
		self.opt=None
		self.SetParameters(mode=initMode)

		return

	def SetParameters(self,mode='default'):
		''' default, from args'''
		if mode == 'default':
			self.ReadArgs(string=None)
		elif mode == 'args':
			self.ReadArgs(string=sys.argv[1:])
		else:
			print('Unknown ')
		return

	def ReadArgs(self, string=None):

		if string is None: 
			string=""

		parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
		# Paths
		parser.add_argument('-datapath', default='./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/', help="Print many messages on screen.")
		parser.add_argument('-outpath', default='./out/', help="Print many messages on screen.")
		# User experience
		parser.add_argument('-verbose', action='store_true', help="Print many messages on screen.")
		parser.add_argument('-plot', action='store_true', help="Plot loss and accuracy during training once the run is over.")
		# Hyperparameters
		parser.add_argument('-opt', choices=['sgd','adam'], default='sgd', help="Choice of the minimization algorithm (sgd,adam)")
		parser.add_argument('-bs', type=int, default=32, help="Batch size")
		parser.add_argument('-lr', type=float, default=0.00005, help="Learning Rate")
		parser.add_argument('-aug', action='store_true', help="Perform data augmentation. Augmentation parameters are hard-coded.")
		parser.add_argument('-model', choices=['mlp','conv2','smallvgg'], default='mlp', help='The model. MLP gives decent results, conv2 is the best, smallvgg overfits (*validation* accuracy oscillates).')
		parser.add_argument('-model_image', choices=['mlp','conv2','smallvgg'], default='mlp', help='For mixed data models, tells what model to use for the image branch.')
		parser.add_argument('-model_feat', choices=['mlp'], default='mlp', help='For mixed data models, tells what model to use for the feature branch.')
		parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers for MLP")
		# Data
		parser.add_argument('-L', type=int, default=128, help="Images are resized to a square of LxL pixels")
		parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
		parser.add_argument('-class_select', nargs='*', default=None, help='List of classes to be looked at (put the class names one by one, separated by spaces). If None, all available classes are studied.')
		parser.add_argument('-datakind', choices=['mixed','feat','image'], default='mixed', help="Which data to load: features, images, or both")
		parser.add_argument('-ttkind', choices=['mixed','feat','image'], default='mixed', help="Which data to use in the test and training sets: features, images, or both")
		# Training time
		parser.add_argument('-totEpochs', type=int, default=5, help="Total number of epochs for the training")
		parser.add_argument('-initial_epoch', type=int, default=0, help='Initial epoch of the training')

		# parser.add_argument('-load', default=None, help='Path to a previously trained model that should be loaded.')
		# parser.add_argument('-override_lr', action='store_true', help='If true, when loading a previously trained model it discards its LR in favor of args.lr')
		# parser.add_argument('-augtype', default='standard', help='Augmentation type')
		# parser.add_argument('-augparameter', type=float, default=0, help='Augmentation parameter')
		# parser.add_argument('-cpu', default=False, help='performs training only on cpus')
		# parser.add_argument('-gpu', default=False, help='performs training only on gpus')
		# args=parser.parse_args()
		args=parser.parse_args(string)

		# Add a trailing / to the paths, just for safety
		args.datapath = args.datapath+'/'
		args.outpath  = args.outpath +'/'
				
		self.ArgsCheck(args)
		self.params=args

		print(args)


		return

	@staticmethod
	def ArgsCheck(args):
		''' Consistency checks for command line arguments '''
		if args.L<8:
			raise ValueError('Linear size of the images <8 pixels is too small to be wanted. Abort.')

		# flatten_image = True if args.model in ['mlp'] else False
		if args.initial_epoch>=args.totEpochs:
			print('The initial epoch is already equalr or larger than the target number of epochs, so there is no need to do anything. Exiting...')
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

		if args.datakind == 'image':
			args.ttkind = 'image'
		elif args.datakind == 'feat':
			args.ttkind = 'feat'

		return

	def CreateOutDir(self):
		''' Create a unique output directory, and put inside it a file with the simulation parameters '''
		outDir = self.params.outpath+'/'+self.params.model+'_mix/'+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")+'/'
		pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)
		self.fsummary=open(outDir+'params.txt','w')
		print(self.params, file=self.fsummary); 
		self.fsummary.flush()
		return

	def LoadData(self, datapath=None, L=None, class_select=None, datakind=None):
		''' 
		Loads dataset using the function in the Cdata class.
		Acts differently in case it is the first time or not that the data is loaded
		'''

		# Default values
		if 	   datapath == None:     datapath = self.params.datapath
		if 			  L == None:            L = self.params.L
		if class_select == None: class_select = self.params.class_select
		if 	   datakind == None:     datakind = self.params.datakind

		# Initialize or Load Data Structure
		if self.data==None:
			self.data = hd.Cdata(datapath, L, class_select, datakind)
		else:
			self.data.Load(datapath, L, class_select ,datakind)

		# Reset parameters	
		self.params.datapath = self.data.datapath
		self.params.L        = self.data.L
		self.params.class_select = self.data.class_select
		self.params.datakind = self.data.kind

		return

	def CreateTrainTestSets(self, ttkind=None, random_state=12345):
		
		if ttkind is None:
			ttkind= self.params.ttkind
		self.tt=htts.CTrainTestSet(self.data.X, self.data.y, ttkind=ttkind)
		self.params.ttkind=self.tt.ttkind

		return


	def Train(self):

		# Callbacks
		checkpointer    = keras.callbacks.ModelCheckpoint(filepath=sim.params.outpath+'/bestweights.hdf5', monitor='val_loss', verbose=0, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy
		coitointerrotto = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
		logger          = keras.callbacks.callbacks.CSVLogger(sim.params.outpath+'epochs.log', separator=' ', append=False)


		self.aug = None if (self.params.aug == False) else ImageDataGenerator(
																				rotation_range=90,      # 
																				vertical_flip=True,
																				horizontal_flip=True,
																				shear_range=30,			# 
																				width_shift_range=0.5,	# 
																				height_shift_range=0.5, # 
																				fill_mode='constant', 
																				validation_split=self.params.testSplit,
																				#From here on it's the default values
																				featurewise_center=False, samplewise_center=False,
																				featurewise_std_normalization=False, samplewise_std_normalization=False,
																				zca_whitening=False, zca_epsilon=1e-06, brightness_range=None, zoom_range=0.0,
																				channel_shift_range=0.0, cval=0.0, rescale=None, preprocessing_function=None,
																				data_format=None,  dtype=None
																			)






		self.trainParams=hm.CreateParams(
									layers=sim.params.layers, 
									lr=sim.params.lr,
        							bs=sim.params.bs,
        							totEpochs=sim.params.totEpochs,
        							callbacks= [checkpointer, logger, coitointerrotto],
        							aug = self.aug,
        							model = self.params.model,
        							model_image = self.params.model_image,
        							model_feat  = self.params.model_feat
        							)

		# train the neural network
		start=time.time()

		if self.params.ttkind == 'mixed':
			self.history, self.model = hm.MixedModel([self.tt.trainXimage,self.tt.trainXfeat], self.tt.trainY, [self.tt.testXimage,self.tt.testXfeat], self.tt.testY, self.trainParams)
		else:
			self.history, self.model = hm.PlainModel(self.tt.trainX, self.tt.trainY, self.tt.testX, self.tt.testY, self.trainParams)

		trainingTime=time.time()-start
		print('Training took',trainingTime/60,'minutes')

		return

	def Report(self):

		predictions = self.Predict()


		clrep=classification_report(self.tt.testY.argmax(axis=1), predictions.argmax(axis=1), target_names=self.tt.lb.classes_)
		print(clrep)

		return


	def Predict(self):

		bs = self.trainParams['bs'] if (self.params.aug == False) else None
		testX =  [self.tt.testXimage, self.tt.testXfeat] if (self.tt.ttkind=='mixed') else self.tt.testX

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
		print("Training-time: {:} seconds".format(trainingTime), file=self.fsummary)
		self.fsummary.close()


	def LoadModel(self):
		raise NotImplementedError
		return

	def SaveModel(self):
		raise NotImplementedError
		return












print('\nRunning',sys.argv[0],sys.argv[1:])

if __name__=='__main__':
	sim=Ctrain()
	sim.SetParameters('args')
	sim.data = hd.Cdata(sim.params.datapath, sim.params.L, sim.params.class_select, sim.params.datakind)
	sim.CreateOutDir()
	sim.CreateTrainTestSets()
	sim.Train()

	sim.Report()








