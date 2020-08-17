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

		parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
		# I/O
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
		parser.add_argument('-model_image', choices=['mlp','conv2','smallvgg'], default=None, help='For mixed data models, tells what model to use for the image branch. For image models, it is the whole model')
		parser.add_argument('-model_feat', choices=['mlp'], default=None, help='For mixed data models, tells what model to use for the feature branch. For feat models, it is the whole model.')
		parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers for MLP")
		parser.add_argument('-dropout', type=float, default=None, help="This is a dropout parameter which is passed to the model wrapper but is currently not used (August 2020) because dropouts are currently hardcoded.")
		# Data
		parser.add_argument('-L', type=int, default=128, help="Images are resized to a square of LxL pixels")
		parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the test set")
		parser.add_argument('-class_select', nargs='*', default=None, help='List of classes to be looked at (put the class names one by one, separated by spaces). If None, all available classes are studied.')
		parser.add_argument('-datakind', choices=['mixed','feat','image'], default=None, help="Which data to load: features, images, or both")
		parser.add_argument('-ttkind', choices=['mixed','feat','image'], default=None, help="Which data to use in the test and training sets: features, images, or both")
		parser.add_argument('-training_data', choices=['True','False'], default='True', help="This is to cope with the different directory structures that I was given. Sometimes the class folder has an extra folder inside, called training_data. For the moment, this happens in the training images they gave me, but not with the validation images.")
		# Training time
		parser.add_argument('-totEpochs', type=int, default=5, help="Total number of epochs for the training")
		parser.add_argument('-initial_epoch', type=int, default=0, help='Initial epoch of the training')
		parser.add_argument('-earlyStopping', type=int, default=100, help='If >0, we do early stopping, and this number is the patience (how many epochs without improving)')

		# parser.add_argument('-override_lr', action='store_true', help='If true, when loading a previously trained model it discards its LR in favor of args.lr')
		# parser.add_argument('-augtype', default='standard', help='Augmentation type')
		# parser.add_argument('-augparameter', type=float, default=0, help='Augmentation parameter')
		# parser.add_argument('-cpu', default=False, help='performs training only on cpus')
		# parser.add_argument('-gpu', default=False, help='performs training only on gpus')
		# args=parser.parse_args()
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
		if args.model_image is None and args.model_feat is None:
			args.model_image = 'conv2'
			args.datakind == 'image'
			args.ttkind = 'image'
			if self.verbose:
				print('No model was specified by the user, so we analyze images with {}'.format(args.model_image))
		
		elif args.model_image is None:
			args.datakind == 'feat'
			args.ttkind = 'feat'

		elif args.model_feat is None:
			args.datakind == 'image'
			args.ttkind = 'image'

		else:
			args.datakind == 'mixed'
			args.ttkind = 'mixed'

		print('kind:',args.ttkind)


		if args.ttkind != 'image' and args.aug==True: 
			print('User asked for data augmentation, but we set it to False, because we only to it for `image` models')
			args.aug=False


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

	def LoadData(self, datapaths=None, L=None, class_select=-1, datakind=None, training_data=True):
		''' 
		Loads dataset using the function in the Cdata class.
		Acts differently in case it is the first time or not that the data is loaded

		The flag `training_data` is there because of how the taxonomists created the data directories. In the folders that I use for training there is an extra subfolder called `training_data`. This subfolder is absent, for example, in the validation directories.
		'''

		# Default values
		if 	   datapaths == None:    datapaths = self.params.datapaths
		if 			  L == None:             L = self.params.L
		if class_select == -1: 	  class_select = self.params.class_select # class_select==None has the explicit meaning of selecting all the classes
		if 	   datakind == None:      datakind = self.params.datakind
		if training_data== None: training_data = self.params.training_data

		# Initialize or Load Data Structure
		if self.data is None:
			self.data = hd.Cdata(datapaths, L, class_select, datakind, training_data=training_data)
		else:
			self.data.Load(datapaths, L, class_select ,datakind, training_data=training_data)

		# Reset parameters	
		self.params.datapaths = self.data.datapath
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


	def Train(self, train=True):

		# Save classes
		if train:
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
									layers 		= self.params.layers, 
									lr 			= self.params.lr,
									bs 			= self.params.bs,
									optimizer 	= self.params.opt,
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

		# train the neural network
		start=time.time()


		# If train==False the wrapper will only load the model. The other branch is for mixed vs non-mixed models
		if train==False:
			trX, trY, teX, teY = (None, None, None, None)
		elif (self.params.ttkind == 'mixed'):
			trX, trY, teX, teY = ([self.tt.trainXimage,self.tt.trainXfeat], self.tt.trainY, [self.tt.testXimage,self.tt.testXfeat], self.tt.testY)
		else:
			trX, trY, teX, teY = (self.tt.trainX, self.tt.trainY, self.tt.testX, self.tt.testY)

		wrapper = hm.CModelWrapper(trX, trY, teX, teY, self.trainParams)
		self.model, self.history = wrapper.model, wrapper.history


		if train:
			trainingTime=time.time()-start
			print('Training took',trainingTime/60,'minutes')

			print('Saving the last model. These are not the best weights, they are the last ones. For the best weights use the callback output (bestweights.hdf5)]')
			self.SaveModel()

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
	sim.Report()
	sim.Finalize()








