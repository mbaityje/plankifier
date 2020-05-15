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
# - preprocess features (mean zero, std one), images as float32
# - implement logging
# - condizioni iniziali
# - data Augmentation
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

		if args.datakind == 'image':
			args.modelkind = 'image'
			args.ttkind = 'image'
		elif args.datakind == 'feat':
			args.modelkind = 'feat'
			args.ttkind = 'feat'

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
		self.tt=CTrainTestSet(self.data.X, self.data.y, ttkind=ttkind, split=True)
		self.params.ttkind=self.tt.ttkind

		return


	def Train(self):


		checkpointer    = keras.callbacks.ModelCheckpoint(filepath=sim.params.outpath+'/bestweights.hdf5', monitor='val_loss', verbose=0, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy
		coitointerrotto = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
		logger          = keras.callbacks.callbacks.CSVLogger(sim.params.outpath+'epochs.log', separator=' ', append=False)

		trainParams=hm.CreateParams(
									layers=sim.params.layers, 
									lr=sim.params.lr,
        							bs=sim.params.bs,
        							totEpochs=sim.params.totEpochs,
        							callbacks= [checkpointer, logger, coitointerrotto]
        							)
		# train the neural network
		start=time.time()

		if self.params.modelkind == 'mixed':
			history, model = hm.MixedModel([self.tt.trainXimage,self.tt.trainXfeat], self.tt.trainY, [self.tt.testXimage,self.tt.testXfeat], self.tt.testY, trainParams)
		else:
			history, model = hm.MLP(self.tt.trainX, self.tt.trainY, self.tt.testX, self.tt.testY, trainParams)

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
	def __init__(self, X, y, ttkind='mixed', split=True):
		''' X and y are dataframes with features and labels'''
		self.ttkind=ttkind

		# Take care of the labels
		self.y=y
		self.VectorizeLabels()


		# Now the features

		if ttkind == 'image':
			self.X=self.ImageNumpyFromMixedDataframe(X)
		elif ttkind == 'feat':
			X=self.DropCols(X, ['npimage','rescaled'])
			self.X=np.array([X.to_numpy()[i] for i in range(len(X.index))])
		else:
			# This checks if there are images, but it also implicitly checks if there are features.
			# In fact, if there are only images, X is a series and has no attribute columns (I am aware this should be coded better). 
			if 'npimage' not in X.columns:
				raise RuntimeError('Error: you asked for mixed Train-Test, but the dataset you gave me does not contain images.')
			self.X=X #Note that with ttkind=mixed, X stays a dataframe
	
		if split:
			self.Split()

		return

	def VectorizeLabels(self):
		''' 
		Transform labels in one-hot encoded vectors 
		This is where we will act if we decide to train with HYBRID LABELS
		'''

		self.lb = LabelBinarizer()

		self.y = self.lb.fit_transform(self.y.tolist())
		return

	def UnvectorizeLabels(self, y):
		''' Recovers the original labels from the vectorized ones '''
		return self.lb.inverse_transform(y) 


	def ImageNumpyFromMixedDataframe(self, X=None):
		''' Returns a numpy array of the shape (nexamples, L, L, channels)'''
		if X is None:
			X=self.X

		# The column containing npimage
		im_col = [i for i,col in enumerate(X.columns) if col == 'npimage'][0] 
		
		return np.array([X.to_numpy()[i, im_col] for i in range( len(X.index) )])


	def Split(self, test_size=0.2, random_state=12345):
		''' handles differently the mixed case, because in that case  X is a dataframe'''
		self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

		if self.ttkind == 'mixed':
			# Images
			self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)
			self.testXimage = self.ImageNumpyFromMixedDataframe(self.testX)
			#Features
			Xf=self.DropCols(self.trainX, ['npimage','rescaled'])
			self.trainXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
			Xf=self.DropCols(self.testX, ['npimage','rescaled'])
			self.testXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])

		return


	def Rescale(self, cols=None):
		raise NotImplementedError

	def SelectCols(self, X, cols):
		''' 
		Keeps only the columns cols from the dataframe X. 
		cols is a list with the columns names
		'''

		if isinstance(X, pd.DataFrame): # Make sure it is not a series
			if set(cols).issubset(set(X.columns)): # Check that columns we want to select exist
				return X[cols]
			else:
				print('self.X.columns: {}'.format(self.X.columns))
				print('requested cols: {}'.format(cols))
				raise IndexError('You are trying to select columns that are not present in the dataframe')
		else:
			assert(len(cols)==1) # If it's a series there should be only one column
			assert(self.X.name==cols[0])# And that column should coincide with the series name
			return

	def DropCols(self, X, cols):
		''' 
		Gets rid of the columns cols from the dataframe X. 
		cols is a list with the columns names
		'''
		return X.drop(columns=cols, errors='ignore')

	def MergeLabels(self):
		''' Merges labels to create aggregated classes '''
		raise NotImplementedError



class Cmodel:

	def __init__():
		return

	@staticmethod
	def MLP(trainX, trainY, testX, testY, params):

		model = hm.MultiLayerPerceptron.Build2Layer(input_shape=(self.numFeat,), classes=len(self.data.classes), layers=self.params.layers)

		return out, model


print('\nRunning',sys.argv[0],sys.argv[1:])

if __name__=='__main__':
	sim=Ctrain()
	sim.SetParameters('args')
	sim.data = hd.Cdata(sim.params.datapath, sim.params.L, sim.params.class_select, sim.params.datakind)
	sim.CreateOutDir()
	sim.CreateTrainTestSets()

	sim.Train()
	sim.Report()








