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
# -scremare lista di argomenti
# -pisciare flatten_image e immettere shape quadrata nel MLP
# -


###########
# IMPORTS #
###########

import os, sys, pathlib, glob, time, datetime, argparse, numpy as np, pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_models as hm, helper_data as hd
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

###############
# TRAIN CLASS #
###############

class Ctrain:
	def __init__(self):
		self.data=None
		self.trainSize=None
		self.testSize=None
		self.RecordInitTime()
		self.ReadArgs()
		self.CreateOutDir()
		self.data = hd.Cdata(self.args.datapath, self.args.L, self.args.class_select, self.args.kind)
		self.data.Preprocess()

		return

	def ReadArgs(self):
		parser = argparse.ArgumentParser(description='Train a model on zooplankton images')

		parser.add_argument('-datapath', default='./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/', help="Print many messages on screen.")
		parser.add_argument('-outpath', default='./out/', help="Print many messages on screen.")

		parser.add_argument('-verbose', action='store_true', help="Print many messages on screen.")
		parser.add_argument('-plot', action='store_true', help="Plot loss and accuracy during training once the run is over.")

		parser.add_argument('-opt', choices=['sgd','adam'], default='sgd', help="Choice of the minimization algorithm (sgd,adam)")
		parser.add_argument('-bs', type=int, default=32, help="Batch size")
		parser.add_argument('-lr', type=float, default=0.00005, help="Learning Rate")
		parser.add_argument('-aug', action='store_true', help="Perform data augmentation.")
		parser.add_argument('-model', choices=['mlp','conv2','smallvgg'], default='mlp', help='The model. MLP gives decent results, conv2 is the best, smallvgg overfits (*validation* accuracy oscillates).')

		parser.add_argument('-L', type=int, default=128, help="Images are resized to a square of LxL pixels")
		parser.add_argument('-depth', type=int, default=3, help="Number of channels")
		parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
		parser.add_argument('-class_select', nargs='*', default=None, help='List of classes to be looked at (put the class names one by one, separated by spaces). If None, all available classes are studied.')
		parser.add_argument('-kind', choices=['mixed','feat'], default='mixed', help="Which data to load: features, images (not implemented), or both")

		parser.add_argument('-totEpochs', type=int, default=5, help="Total number of epochs for the training")
		parser.add_argument('-initial_epoch', type=int, default=0, help='Initial epoch of the training')

		# parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers for MLP")
		# parser.add_argument('-load', default=None, help='Path to a previously trained model that should be loaded.')
		# parser.add_argument('-override_lr', action='store_true', help='If true, when loading a previously trained model it discards its LR in favor of args.lr')
		# parser.add_argument('-augtype', default='standard', help='Augmentation type')
		# parser.add_argument('-augparameter', type=float, default=0, help='Augmentation parameter')
		# parser.add_argument('-cpu', default=False, help='performs training only on cpus')
		# parser.add_argument('-gpu', default=False, help='performs training only on gpus')
		# args=parser.parse_args()
		args=parser.parse_args()


		# Add a trailing / to the paths, just for safety
		args.datapath = args.datapath+'/'
		args.outpath  = args.outpath +'/'
		self.args=args
		
		self.ArgsCheck()

		return

	def ArgsCheck(self):
		''' Consistency checks for command line arguments '''
		if self.args.L<8:
			raise ValueError('Linear size of the images <8 pixels is too small to be wanted. Abort.')
		if self.args.aug==True and self.args.model in ['mlp']:
			print('We don\'t do data augmentation with the MLP')
			args.aug=False

		# flatten_image = True if args.model in ['mlp'] else False
		if self.args.initial_epoch>=self.args.totEpochs:
			print('The initial epoch is already equalr or larger than the target number of epochs, so there is no need to do anything. Exiting...')
			raise SystemExit

		if self.args.verbose:
			ngpu=len(keras.backend.tensorflow_backend._get_available_gpus())
			print('We have {} GPUs'.format(ngpu))

		return

	def RecordInitTime(self):
		self.initTime = datetime.datetime.now()
		return

	def CreateOutDir(self):
		''' Create a unique output directory, and put inside it a file with the simulation parameters '''
		outDir = self.args.outpath+'/'+self.args.model+'_mix/'+self.initTime.strftime("%Y-%m-%d_%Hh%Mm%Ss")+'/'
		pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)
		fsummary=open(outDir+'args.txt','w')
		print(self.args, file=fsummary); 
		fsummary.flush()
		return

	def LoadData(self, kind):
		''' 
		Loads dataset using the function in the Cdata class
		'''
		self.data.Load(kind)
		return

	def CreateTrainTestSets(self, kind=None, random_state=12345):

		if kind==None: kind=self.args.kind

		if kind == 'mixed':
			(self.trainX, self.testX, self.trainY, self.testY) = train_test_split(self.data.X, self.data.y, test_size=self.args.testSplit, random_state=random_state)
		elif kind == 'feat':
			(self.trainX, self.testX, self.trainY, self.testY) = train_test_split(self.data.X.drop(columns=['npimage'], errors='ignore'), self.data.y, test_size=self.args.testSplit, random_state=random_state)
		elif kind == 'image':
			(self.trainX, self.testX, self.trainY, self.testY) = train_test_split(self.data.X.npimage, self.data.y, test_size=self.args.testSplit, random_state=random_state)
		else:
			raise NotImplementerError('Unknown kind '+kind+' in CreateTrainTestSets')

		self.trainSize=len(self.trainY)
		self.testSize =len(self.testY)


	def Train(self, kind):
		return

	def Predict(self):
		return




##########################
# COMMAND LINE ARGUMENTS #
##########################





########
# INIT #
########



#######
# RUN #
#######
print('\nRunning',sys.argv[0],sys.argv[1:])

if __name__=='__main__':
	sim=Ctrain()
	sim.CreateTrainTestSets()








