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

		parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
		# I/O
		parser.add_argument('-datapath', default='./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/', help="Directory with the data.")
		parser.add_argument('-outpath', default='./out/', help="directory where you want the output saved")
		parser.add_argument('-load', default=None, help='Path to a previously trained model that should be loaded.')
		parser.add_argument('-saveModelName', default='keras_model.h5', help='Name of the model when it is saved.')
		# parser.add_argument('-override_lr', action='store_true', help='If true, when loading a previously trained model it discards its LR in favor of args.lr')
		# User experience
		parser.add_argument('-verbose', action='store_true', help="Print many messages on screen.")
		parser.add_argument('-plot', action='store_true', help="Plot loss and accuracy during training once the run is over.")
		# Hyperparameters
		parser.add_argument('-opt', choices=['sgd','adam'], default='sgd', help="Choice of the minimization algorithm (sgd,adam)")
		parser.add_argument('-bs', type=int, default=32, help="Batch size")
		parser.add_argument('-lr', type=float, default=0.00005, help="Learning Rate")
		parser.add_argument('-aug', action='store_true', help="Perform data augmentation. Augmentation parameters are hard-coded.")
		# parser.add_argument('-model', choices=['mlp','conv2','smallvgg'], default='mlp', help='The model. MLP gives decent results, conv2 is the best, smallvgg overfits (*validation* accuracy oscillates).')
		parser.add_argument('-model_image', choices=['mlp','conv2','smallvgg'], default='mlp', help='For mixed data models, tells what model to use for the image branch.')
		parser.add_argument('-model_feat', choices=['mlp'], default='mlp', help='For mixed data models, tells what model to use for the feature branch.')
		parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers for MLP")
		parser.add_argument('-dropout', type=float, default=None, help="Drop out")
		# Data
		parser.add_argument('-L', type=int, default=128, help="Images are resized to a square of LxL pixels")
		parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
		parser.add_argument('-class_select', nargs='*', default=None, help='List of classes to be looked at (put the class names one by one, separated by spaces). If None, all available classes are studied.')
		parser.add_argument('-datakind', choices=['mixed','feat','image'], default='mixed', help="Which data to load: features, images, or both")
		parser.add_argument('-ttkind', choices=['mixed','feat','image'], default='mixed', help="Which data to use in the test and training sets: features, images, or both")
		parser.add_argument('-training_data', choices=[True,False], default=True, help="This is to cope with the different directory structures that I was given. Sometimes the class folder has an extra folder inside, called training_data. For the moment, this happens in the training images they gave me, but not with the validation images.")
		# Training time
		parser.add_argument('-totEpochs', type=int, default=5, help="Total number of epochs for the training")
		parser.add_argument('-initial_epoch', type=int, default=0, help='Initial epoch of the training')
		parser.add_argument('-earlyStopping', type=int, default=100, help='If >0, we do early stopping, and this number is the patience (how many epochs without improving)')

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

		if args.datakind == 'image':
			args.ttkind = 'image'
		elif args.datakind == 'feat':
			args.ttkind = 'feat'

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
		return

	def UpdateParams(self, **kwargs):
		''' Updates the parameters given in kwargs, and updates params.txt'''
		self.paramsDict = vars(self.params)
		if kwargs is not None:
			for key, value in kwargs.items():
				self.paramsDict[key] = value
		self.WriteParams()

		return

	def LoadData(self, datapath=None, L=None, class_select=None, datakind=None, training_data=True):
		''' 
		Loads dataset using the function in the Cdata class.
		Acts differently in case it is the first time or not that the data is loaded
		'''

		# Default values
		if 	   datapath == None:     datapath = self.params.datapath
		if 			  L == None:            L = self.params.L
		# if class_select == None: class_select = self.params.class_select # I think this one is wrong, because class_select==None has an explicit meaning
		if 	   datakind == None:     datakind = self.params.datakind
		if training_data== None:training_data = self.params.training_data

		# Initialize or Load Data Structure
		if self.data is None:
			self.data = hd.Cdata(datapath, L, class_select, datakind, training_data=training_data)
		else:
			self.data.Load(datapath, L, class_select ,datakind, training_data=training_data)

		# Reset parameters	
		self.params.datapath = self.data.datapath
		self.params.L        = self.data.L
		self.params.class_select = self.data.class_select
		self.params.datakind = self.data.kind

		return

	def CreateTrainTestSets(self, ttkind=None, testSplit=None, random_state=12345):
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


		self.tt=htts.CTrainTestSet(self.data.X, self.data.y, ttkind=ttkind, testSplit=testSplit)
		self.params.ttkind=self.tt.ttkind

		return


	def Train(self):

		# Save classes
		np.save(self.params.outpath+'/classes.npy', self.tt.lb.classes_)

		# Callbacks
		checkpointer    = keras.callbacks.ModelCheckpoint(filepath=self.params.outpath+'/bestweights.hdf5', monitor='val_loss', verbose=0, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy
		logger          = keras.callbacks.callbacks.CSVLogger(self.params.outpath+'/epochs.log', separator=' ', append=False)
		callbacks=[checkpointer, logger]
		if self.params.earlyStopping>0:
			earlyStopping   = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', patience=self.params.earlyStopping, restore_best_weights=True)
			callbacks.append(earlyStopping)

		self.aug = None if (self.params.aug == False) else ImageDataGenerator(
                        rotation_range=90,
                        vertical_flip=True,
                        horizontal_flip=True,
                        shear_range=10
                )

		self.trainParams=hm.CreateParams(
									layers = self.params.layers, 
									lr = self.params.lr,
        							bs = self.params.bs,
        							optimizer = self.params.opt,
        							totEpochs = self.params.totEpochs,
        							dropout = self.params.dropout,
        							callbacks = callbacks,
        							aug = self.aug,
        							model = self.model,
        							model_image = self.params.model_image,
        							model_feat  = self.params.model_feat,
        							load_weights = self.params.load_weights,
        							initial_epoch=self.params.initial_epoch
        							)

		# train the neural network
		start=time.time()

		if self.params.ttkind == 'mixed':
			self.history, self.model = hm.MixedModel([self.tt.trainXimage,self.tt.trainXfeat], self.tt.trainY, [self.tt.testXimage,self.tt.testXfeat], self.tt.testY, self.trainParams)
		else:
			self.history, self.model = hm.PlainModel(self.tt.trainX, self.tt.trainY, self.tt.testX, self.tt.testY, self.trainParams)

		trainingTime=time.time()-start
		print('Training took',trainingTime/60,'minutes')

		print('Saving the last model. These are not the best weights, they are the last ones. For the best weights use the callback output (bestweights.hdf5)]')
		self.SaveModel()

		return


	def Report(self):

		predictions = self.Predict()
		clrep=classification_report(self.tt.testY.argmax(axis=1), predictions.argmax(axis=1), target_names=self.tt.lb.classes_)
		print(clrep)

		return


	def LoadModel(self, modelfile=None):

		if modelfile is not None:
			self.params.load = modelfile
		self.model=keras.models.load_model(self.params.load)

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
		print("Training-time: {:} seconds".format(trainingTime), file=self.fsummary)
		self.fsummary.close()


	def SaveModel(self):

		self.model.save(self.params.outpath+'/'+self.params.saveModelName, overwrite=True, include_optimizer=True)
		return











print('\nRunning',sys.argv[0],sys.argv[1:])

if __name__=='__main__':
	sim=Ctrain(initMode='args')
	sim.LoadData()
	sim.CreateOutDir()
	sim.CreateTrainTestSets()
	sim.Train()
	sim.Report()








