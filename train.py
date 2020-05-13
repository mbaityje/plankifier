#!/usr/bin/env python3
# 
# Script that trains a model
# 
# Options: 
# - kind of model
# - kind of data input type
# 

# DA FARE
# 
# - modularizzare il tutto
# - implement logging
# - condizioni iniziali
# 


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
from src import helper_models as hm, helper_data as hd
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pdb



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
		print(initMode)
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
		parser.add_argument('-aug', action='store_true', help="Perform data augmentation.")
		parser.add_argument('-model', choices=['mlp','conv2','smallvgg'], default='mlp', help='The model. MLP gives decent results, conv2 is the best, smallvgg overfits (*validation* accuracy oscillates).')
		parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers for MLP")
		# Data
		parser.add_argument('-L', type=int, default=128, help="Images are resized to a square of LxL pixels")
		parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
		parser.add_argument('-class_select', nargs='*', default=None, help='List of classes to be looked at (put the class names one by one, separated by spaces). If None, all available classes are studied.')
		parser.add_argument('-datakind', choices=['mixed','feat','image'], default='mixed', help="Which data to load: features, images (not implemented), or both")
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

		self.params.modelkind=args.datakind

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

		return

	def CreateOutDir(self):
		''' Create a unique output directory, and put inside it a file with the simulation parameters '''
		outDir = self.params.outpath+'/'+self.params.model+'_mix/'+datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")+'/'
		pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)
		fsummary=open(outDir+'params.txt','w')
		print(self.params, file=fsummary); 
		fsummary.flush()
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
			self.data.Load(datakind)

		# Reset parameters	
		self.params.datapath = self.data.datapath
		self.params.L        = self.data.L
		self.params.class_select = self.data.class_select
		self.params.datakind = self.data.kind

		return

	def CreateTrainTestSets(self, ttkind='mixed', random_state=12345):
		''' 
		Create Training and Test Sets: 
		- Decides what X is according to the model kind
		- Splits data in train and test sets
		- Transforms the labels y in an nclass-vector
		'''

		# If the data is mixed, we allow to only get images or features. If it is not mixed, this cannot be done.
		if self.data.kind == 'image' or ttkind=='image':
			ttkind='image'
			X=np.array([self.data.Ximage.values[i] for i in range(len(self.data.Ximage.index))])
		elif self.data.kind == 'feat' or ttkind=='feat':
			ttkind='feat'
			X=self.data.Xfeat
		elif ttkind=='mixed':
			X=self.data.X
		else:
			raise NotImplementerError('CreateTrainTestSets() not implemented ttkind')

		
		self.tts=CTrainTestSet(self.data.X, self.data.y, self.params.testSplit, random_state=random_state)

		if ttkind == 'mixed':
			(self.trainX, self.testX, trainY, testY) = train_test_split(X, self.data.y, test_size=self.params.testSplit, random_state=random_state)
			self.numFeat   = len(self.data.Xfeat.columns)
		elif ttkind == 'feat':
			(self.trainX, self.testX, trainY, testY) = train_test_split(X, self.data.y, test_size=self.params.testSplit, random_state=random_state)
			self.numFeat   = len(self.data.Xfeat.columns)
		elif ttkind == 'image':
			(self.trainX, self.testX, trainY, testY) = train_test_split(X, self.data.y, test_size=self.params.testSplit, random_state=random_state)
		else:
			raise NotImplementerError('Unknown kind '+ttkind+' in CreateTrainTestSets')

		# This is where we will act if we decide to train with HYBRID LABELS
		lb = LabelBinarizer()
		# print('trainY:',trainY.tolist())
		# print('testY:',testY.tolist())
		self.trainY = lb.fit_transform(trainY.tolist())
		self.testY = lb.transform(testY.tolist())
		# print('trainY:',self.trainY)
		# print('testY:',self.testY)


		# pdb.set_trace()

		self.trainSize = len(self.trainY)
		self.testSize  = len(self.testY)

		return


	def SetModel(self, kind=None):
		''' Set the Model and Optimizer - still needs a lot of work '''

		if kind==None: kind=self.params.modelkind # Change this

		if kind=='mixed':
			ni = 32 # Number of output nodes for the image branch
			nf = 16 # Number of output nodes for the image branch
			model_feat = hm.MultiLayerPerceptron.Build2Layer(input_shape=(self.numFeat,)             , classes=None, layers=self.params.layers)
			model_imag = hm.MultiLayerPerceptron.Build2Layer(input_shape=np.shape(sim.data.Ximage.values[0]), classes=None, layers=self.params.layers)
			combinedInput = concatenate([model_feat.output, model_imag.output]) # Combine the two
			model_join = Dense(64, activation="relu")(combinedInput)
			model_join = Dense(len(self.data.classes), activation="softmax")(model_join)			
			self.model = Model(inputs=[model_feat.input, model_imag.input], outputs=model_join)

		elif kind=='feat':
			self.model = hm.MultiLayerPerceptron.Build2Layer(input_shape=(self.numFeat,), classes=len(self.data.classes), layers=self.params.layers)

		elif kind=='image':
			print('Shape:', np.shape(sim.data.Ximage.values[0]))
			self.model = hm.MultiLayerPerceptron.Build2Layer(input_shape=np.shape(self.data.Ximage.values[0]), classes=len(self.data.classes), layers=self.params.layers)

		# compile the model using SGD as our optimizer and categorical
		# cross-entropy loss (you'll want to use binary_crossentropy
		# for 2-class classification)
		if self.params.opt=='adam':
			self.opt = keras.optimizers.Adam(learning_rate=self.params.lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
		elif self.params.opt=='sgd':
			self.opt = keras.optimizers.SGD(lr=self.params.lr, nesterov=True)
		else:
			raise NotImplementedError('Optimizer {} is not implemented'.format(self.arg.opt))
		self.model.compile(loss="categorical_crossentropy", optimizer=self.opt, metrics=["accuracy"])

		return

	def Train(self):

		# checkpoints
		checkpointer    = keras.callbacks.ModelCheckpoint(filepath=self.params.outpath+'/bestweights.hdf5', monitor='val_loss', verbose=0, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy
		# coitointerrotto = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', patience=params.totEpochs, restore_best_weights=True)
		logger          = keras.callbacks.callbacks.CSVLogger(self.params.outpath+'epochs.log', separator=' ', append=False)
		callbacks=[checkpointer, logger]

		if self.params.aug:
			raise NotImplementedError('Data Aumentation not implemented')

		### TRAIN ###

		# train the neural network
		start=time.time()

		if self.params.modelkind=='mixed':
			trainXim= np.array([self.trainX.npimage.to_numpy()[i] for i in range(len(self.trainX.index))])
			testXim = np.array([self.testX.npimage.to_numpy ()[i] for i in range(len(self.testX.index))])
			history = self.model.fit(
				[self.trainX.drop(columns='npimage').to_numpy(), trainXim], self.trainY, batch_size=self.params.bs, 
				validation_data=([self.testX.drop(columns='npimage').to_numpy(),testXim], self.testY), 
				epochs=self.params.totEpochs, 
				callbacks=callbacks,
				initial_epoch = self.params.initial_epoch)
		elif self.params.modelkind=='feat':
			history = self.model.fit(self.trainX, self.trainY, validation_data=(self.testX, self.testY), epochs=self.params.totEpochs, batch_size=self.params.bs)
			# history = self.model.fit(
			# 	self.trainX.drop(columns='npimage', errors='ignore').to_numpy(), self.trainY, batch_size=self.params.bs, 
			# 	validation_data=(self.testX.drop(columns='npimage', errors='ignore').to_numpy(), self.testY), 
			# 	epochs=self.params.totEpochs, 
			# 	callbacks=callbacks,
			# 	initial_epoch = self.params.initial_epoch)
		elif self.params.modelkind=='image':
			history = self.model.fit(
				self.trainX, self.trainY, batch_size=self.params.bs, 
				validation_data=(self.testX, self.testY), 
				epochs=self.params.totEpochs, 
				callbacks=callbacks,
				initial_epoch = self.params.initial_epoch)


		trainingTime=time.time()-start
		print('Training took',trainingTime/60,'minutes')

		return

	def Report(self):
		return

	def Predict(self):
		return

	def LoadModel(self):
		return

	def SaveModel(self):
		return



class CTrainTestSet:
	def __init__(self, X, y, test_size=0.2, random_state=12345):
		''' X and y are dataframes with features and labels'''
		self.X=X
		self.y=y

		(self.trainX, self.testX, trainY, testY) = \
							train_test_split(self.X,     self.y, \
											test_size=test_size, \
											random_state=random_state)

		return

	def SelectCols(self, cols):
		''' 
		Keeps only the columns cols. 
		cols is a list with the columns names
		'''
		self.X=self.X[cols]
		return

	def DropCols(self, cols):
		''' 
		Gets rid of the columns cols. 
		cols is a list with the columns names
		'''
		self.X = self.X.drop(columns=cols, errors='ignore')
	
		return

	def MergeLabels(self):
		''' Merges labels to create aggregated classes '''
		raise NotImplementedError






print('\nRunning',sys.argv[0],sys.argv[1:])

if __name__=='__main__':
	sim=Ctrain()
	sim.SetParameters('args')
	print(sim.params)
	print('class_select:',sim.params.class_select)
	sim.data = hd.Cdata(sim.params.datapath, sim.params.L, sim.params.class_select, sim.params.datakind)



	sim.CreateOutDir()

	sim.CreateTrainTestSets()
	sim.SetModel()
	sim.Train()
	sim.Report()








