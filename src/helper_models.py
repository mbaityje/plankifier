# import the necessary packages

# keras related libraries
import keras
from keras.models import Sequential, Model,load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import concatenate,Reshape
from keras import backend as K
from keras import metrics as metrics
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.densenet import DenseNet201, DenseNet121
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import efficientnet.keras as efn

# tensorflow related libraries
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import top_k_categorical_accuracy

#Keras tuner related libraries
import kerastuner as kt
from kerastuner import HyperModel
from kerastuner.tuners import BayesianOptimization #,RandomSearch,Hyperband,

#SKlearn related libraries
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier
# from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix

# Other libraries
import numpy as np
import pandas as pd
import pickle
from numpy import dstack
import joblib
from pathlib import Path
import os
import matplotlib.pyplot as plt, seaborn as sns
from collections import Counter

# from matplotlib import pyplot
# import keras_metrics as km

np.set_printoptions(threshold=np.inf)

    
def CreateParams(layers= None, lr =None, bs=None, optimizer=None,classifier=None,totEpochs= None, dropout=None, callbacks= None, 
				 initial_epoch=0, aug=None, modelfile=None, model_feat=None, model_image=None, load_weights=None, 
				 # override_lr=False, 
				 train=True, 
				 # numclasses=None
				 ):
	''' Creates an empty dictionary with all possible entries'''

	params={
		'layers': layers,
        'lr': lr,
        'bs': bs,
        'optimizer': optimizer,
        'classifier': classifier,
        'totEpochs': totEpochs,
        'dropout': dropout,
        'callbacks': callbacks,
        'initial_epoch': initial_epoch,
        'aug': aug,
        'modelfile': modelfile, # Name of the file where a model is stored, that we want to load
        'model_feat': model_feat, # For mixed models, what the feature branch gets
        'model_image': model_image, # For mixed models, what the image branch gets
		'load_weights': load_weights, # If you want to load weights from file, put the filename (with path) here
		# 'override_lr': override_lr, # Whether to load model from file
		'train': train, # Whether to train the model (e.g. maybe you only want to load it)
		# 'numclasses': numclasses, # If no labels are given, we must give the number of classes through this variable
		}

	return params


class CModelWrapper:
	'''
	A wrapper class for models
	'''
	def __init__(self, trainX, trainY, testX, testY,class_weights, params, verbose=False, numclasses=None):

		self.history, self.model = None, None

		(self.trainX, self.trainY, self.testX, self.testY,self.class_weights, self.params, self.verbose) = (trainX, trainY, testX, testY,class_weights, params, verbose)

		self.numclasses = numclasses
#		if trainY is not None:
#			assert(len(trainY[0]) == self.numclasses)

		self.SetArchitecture()   # Defines self.model
		self.InitModelWeights()

		if params['train'] == False: # If we are not interested in training, we are only loading the model
			return
        
		self.SetOptimizer()
		self.Compile()
		self.Train() # Trains if params['train'] is set to True
		return


	def InferModelKind(self):
		''' 
		Decide whether the model in the wrapper is image, feat, or mixed.
		Sets the following variables:

		modelkind: tells us which kind of model we have (feat, image or mixed)
		modelname: tells us which specific models we are useing (e.g. MLP, smallvgg, etc...)

		'''

		if (self.params['model_feat'] is None) and (self.params['model_image'] is None):
			print('Either model_feat ({}) or model_image ({}) should be defined'.format(self.params['model_feat'],self.params['model_image']))
			self.modelkind = None
			raise ValueError

		elif self.params['model_image'] is None:
			self.modelkind = 'feat'
			self.modelname = self.params['model_feat']
			assert(len(np.shape(self.trainX[0])) == 1)

		elif self.params['model_feat'] is None:
			self.modelkind = 'image'
			self.modelname = self.params['model_image']
			assert(len(np.shape(self.trainX[0])) == 3)

		else:
			self.modelkind = 'mixed'
			self.modelname = (self.params['model_image'], self.params['model_feat'])

		if self.verbose:
			print('InferModelKind(): setting model kind to {}'.format(self.modelkind))

		return

	def SetModel(self):

		if self.modelkind == 'feat':
			self.SetModelFeat()

		elif self.modelkind == 'image':
			self.SetModelImage()

		elif self.modelkind == 'mixed':
			self.SetModelMixed()

		else:
			raise ValueError('SetModel: unrecognized modelkind {}'.format(self.modelkind))

            
	def SetModelImage(self):
		'''
		Currently available image models: MLP, conv2, smallvgg
		'''

		if self.modelname == 'mlp':
			self.model = MultiLayerPerceptron.Build2Layer(input_shape=self.trainX[0].shape, classes=self.numclasses, layers=self.params['layers'])
		elif self.modelname == 'conv2':
			self.model = Conv2Layer.Build(input_shape=self.trainX[0].shape, classes=self.numclasses, last_activation='softmax')
		elif self.modelname == 'mobile':
			self.model = MobileNet.Build(input_shape=self.trainX[0].shape, classes=self.numclasses)  
		elif self.modelname == 'eff0':
			self.model = EfficientNetB0Model.Build(input_shape=self.trainX[0].shape, classes=self.numclasses) 
		elif self.modelname == 'eff1':
			self.model = EfficientNetB1Model.Build(input_shape=self.trainX[0].shape, classes=self.numclasses)
		elif self.modelname == 'eff2':
			self.model = EfficientNetB2Model.Build(input_shape=self.trainX[0].shape, classes=self.numclasses)
		elif self.modelname == 'eff3':
			self.model = EfficientNetB3Model.Build(input_shape=self.trainX[0].shape, classes=self.numclasses)
		elif self.modelname == 'eff4':
			self.model = EfficientNetB4Model.Build(input_shape=self.trainX[0].shape, classes=self.numclasses)
		elif self.modelname == 'eff5':
			self.model = EfficientNetB5Model.Build(input_shape=self.trainX[0].shape, classes=self.numclasses)
		elif self.modelname == 'eff6':
			self.model = EfficientNetB6Model.Build(input_shape=self.trainX[0].shape, classes=self.numclasses)
		elif self.modelname == 'eff7':
			self.model = EfficientNetB7Model.Build(input_shape=self.trainX[0].shape, classes=self.numclasses) 
		elif self.modelname == 'res50':
			self.model = ResNet50.Build(input_shape=self.trainX[0].shape, classes=self.numclasses)   
		elif self.modelname == 'incepv3':
			self.model = InceptionV3.Build(input_shape=self.trainX[0].shape, classes=self.numclasses) 
		elif self.modelname == 'dense121':
			self.model = DenseNet121.Build(input_shape=self.trainX[0].shape, classes=self.numclasses) 
		elif self.modelname == 'smallvgg':
			self.model = SmallVGGNet.Build(input_shape=self.trainX[0].shape, classes=self.numclasses)
		else:
			raise NotImplementedError('SetModelImage() - chosen model {} is not implemented'.format(self.modelname))

	def SetModelFeat(self):
		'''
		Currently available feature models: MLP
		'''

		if self.modelname == 'mlp':
			self.model = MultiLayerPerceptron.Build2Layer(input_shape=self.trainX[0].shape, classes=self.numclasses, layers=self.params['layers'])

		else:
			raise NotImplementedError('SetModelImage() - chosen model {} is not implemented'.format(self.modelname))

	def SetModelMixed(self):
		'''
		Set Model for image+features input data
		'''

		shape_of_image = self.trainX[0][0].shape
		shape_of_feat  = self.trainX[1][0].shape

		# Currently there is only one option for mixed models, so no branching required
		self.model = MixedModel.Build(	input_shape	= [shape_of_image, shape_of_feat], 
										classes 	= self.numclasses,
										modelnames	= self.modelname, # This is a tuple with 2 model names, one per branch
										layers		= [self.params['layers'], self.params['layers']] # For the moment we're assigning the same layers to each mlp. These layers are only used if the models are MLP
										)


	def SetArchitecture(self):
		'''
		Set Model Architecture in the self.model attribute
		'''

		# Either load the model...
		if self.params['modelfile'] is not None:
			self.model=keras.models.load_model(self.params['modelfile'])
			self.modelkind = 'undetermined' # At some point we should infer the model kind from the loaded architecture

			'''
			We can decide to override the features of the loaded model. In the following example,
			I do it for the learning rate. This is commented out because I am not using it, and because 
			probably it should be done in a specific method devoted to the learning rate (which will be
			likely developed when the LR schedule is implemented)

			print('LR of the loaded model:', K.get_value(model.optimizer.lr))
			if params['override_lr']==True:
				K.set_value(model.optimizer.lr, params['lr'])
				print('Setting the LR to', params['lr'])


			Currently, we are overriding the optimizer state and compilation. 
			We likely want to change this.
			'''
		
		# ...or start a model from scratch
		else:
			self.InferModelKind()
			self.SetModel()

		return


	def InitModelWeights(self):
		'''
		Weight initialization. This function is only partly implemented, since custom initializations are not done.
		'''

		if (self.params['load_weights'] is None):
			print('WARNING: At the current state, we are taking the default weight initialization, whatever it is. This must change in order to have better control.')
		else:
			print('Loading weights from ',self.params['load_weights'])
			self.model.load_weights(self.params['load_weights'])


	def SetOptimizer(self):

		# Set Optimizer
		if self.params['optimizer'] == 'sgd':
			self.optimizer = keras.optimizers.SGD(lr=self.params['lr'], nesterov=True)

		elif self.params['optimizer'] == 'adam':
			self.optimizer = keras.optimizers.Adam(learning_rate=self.params['lr'], beta_1=0.9, beta_2=0.999, amsgrad=False)



	def Compile(self):
	# Set Classifier
		if self.params['classifier'] == 'binary' or 'versusall':
			self.model.compile(loss="binary_crossentropy", optimizer=self.optimizer, 
                               metrics=["accuracy"])
		elif self.params['classifier'] == 'multi':
			self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, 
                               metrics=["accuracy"])
		return

    
	def Train(self):
		'''
		Trains the model if params['train'] is set to True, and logs the history in self.history
		'''
		if self.params['train']:            
			if self.params['aug'] is None:
				self.history = self.model.fit(
									self.trainX, self.trainY, 
									validation_data=(self.testX, self.testY), 
									class_weight=self.class_weights,
									epochs=self.params['totEpochs'], 
									batch_size=self.params['bs'], 
									callbacks=self.params['callbacks'],
									initial_epoch = self.params['initial_epoch'])
			else:
				assert(self.modelkind!='feat', self.modelkind!='mixed', "We only augment with image data")
				self.history = self.model.fit(
									self.params['aug'].flow(self.trainX, self.trainY, batch_size=self.params['bs']), 
									validation_data=(self.testX, self.testY),
									class_weight=self.class_weights,
									epochs=self.params['totEpochs'], 
									callbacks=self.params['callbacks'],
									initial_epoch = self.params['initial_epoch'],
									steps_per_epoch=len(self.trainX)//self.params['bs']
									)
		return


class MultiLayerPerceptron:
	@staticmethod
	def Build2Layer(input_shape, classes, layers=[64,32], activation="sigmoid", last_activation="softmax"):
		model = Sequential()
		if len(input_shape)==1:
			model.add(Dense(layers[0], input_shape=input_shape, activation=activation))
		else:
			model.add( Flatten(input_shape=input_shape ) )
			model.add(Dense(layers[0], activation=activation))
		model.add(Dense(layers[1], activation=activation))

		model.add(Dense(classes, activation=last_activation))
		return model

class Conv2Layer:
	@staticmethod
	def Build(input_shape, classes, last_activation='softmax'):
		model = Sequential()
		chanDim = -1

		# Beware, kernel_size is hard coded for the moment, so it might not work if images are small
		model.add(Conv2D(64, kernel_size=24, activation='relu', input_shape=input_shape))
		model.add(Conv2D(32, kernel_size=12, activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())

		model.add(Dense(classes, activation=last_activation))

		return model

class SmallVGGNet:
	@staticmethod
	def Build(input_shape, classes, last_activation='softmax'):
		model = Sequential()
		chanDim = -1 		# initialize the model along with the input shape to be "channels last" and the channels dimension itself

		# CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 3 => POOL layer set
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation(last_activation))

		# return the constructed network architecture
		return model


class MixedModel:
	@staticmethod
	def Build(input_shape, classes, modelnames=['conv2','mlp'], nout=[32,32], ncombine=32, layers=[[128,64],[128,64]]):
		'''
		Builds model that takes features on one side, and images on the other.
		Some things are hard-coded because I didn't really use mixed models for the moment.

		We assume that mixed variables are in the format: [images, features]

		Features     Images
			|			|
			|			|
			\          /
			 \        /
			  \      /
			   \    /
				\  /
				 \/
				 ||
				 ||
			   Output			   

		Input:
		- input_shape:	[shape_of_feat, shape_of_image]
		- classes: 		number of classes
		- nout:			[#output nodes of feature branch, #output nodes of image branch]
		- ncombine:		size of the hidden layer that combines the two branches
		- models:  		[model of feature branch, model of image branch]
		- layers:  		[model of feature branch, model of image branch]

		Output: 
		- a kickass model
		'''

		# Make sure that mixed variables are in the format: [images, features]
		assert(len(input_shape[0]) == 3, "MixedModel: first dimension should be images, but has shape {}".format(input_shape[0]))
		assert(len(input_shape[1]) == 1, "MixedModel: second dimension should be features, but has shape {}".format(input_shape[1]))

		#
		# Image branch
		#
		if modelnames[0] == 'mlp':
			model_image = MultiLayerPerceptron.Build2Layer(
				input_shape=input_shape[0], classes=nout[0], last_activation = 'sigmoid', layers=layers[0])
		elif modelnames[0] == 'conv2':
			model_image = Conv2Layer.Build(
				input_shape=input_shape[0], classes=nout[0], last_activation = 'sigmoid')
		elif modelnames[0] == 'mobile':
			model_image = MobileNetModel.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'eff0':
			model_image = EfficientNetB0Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'eff1':
			model_image = EfficientNetB1Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'eff2':
			model_image = EfficientNetB2Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'eff3':
			model_image = EfficientNetB3Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'eff4':
			model_image = EfficientNetB4Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'eff5':
			model_image = EfficientNetB5Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'eff6':
			model_image = EfficientNetB6Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'eff7':
			model_image = EfficientNetB7Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'res50':
			model_image = ResNet50Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'incepv3':
			model_image = InceptionV3Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'dense121':
			model_image = DenseNet121Model.Build(
				input_shape=input_shape[0], classes=nout[0])
		elif modelnames[0] == 'smallvgg':
			model_image = SmallVGGNetModel.Build(
				input_shape=input_shape[0], classes=nout[0], last_activation = 'sigmoid')
		else: 		
			raise NotImplementedError('MixedModel -- Not implemented model image')

		#
		# Feature branch
		#
		if modelnames[1] == 'mlp':
			model_feat = MultiLayerPerceptron.Build2Layer(
				input_shape=input_shape[1] , classes=nout[1], last_activation = 'sigmoid', layers=layers[1])
		else: 
			raise NotImplementedError('MixedModel -- Not implemented model feat')


		#
		# Combine branches
		#
		combinedInput = concatenate([model_image.output, model_feat.output]) # Combine the two
		model_join = Dense(ncombine, activation="relu")(combinedInput)
		model_join = Dense(classes, activation="softmax")(model_join)				
		model = Model(inputs=[model_image.input, model_feat.input], outputs=model_join)

		return model

class LeNet: # This is from old code - was not tested here
	@staticmethod
	def Build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
    

class MobileNetModel:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=MobileNet(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model    

class EfficientNetB0Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=efn.EfficientNetB0(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model  


class EfficientNetB1Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=efn.EfficientNetB1(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model 
    

class EfficientNetB2Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=efn.EfficientNetB2(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model 
    

class EfficientNetB3Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=efn.EfficientNetB3(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model 
    

class EfficientNetB4Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=efn.EfficientNetB4(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model 
    

class EfficientNetB5Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=efn.EfficientNetB5(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model 
    
class EfficientNetB6Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=efn.EfficientNetB6(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model 


class EfficientNetB7Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=efn.EfficientNetB7(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model  
    
class ResNet50Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=ResNet50(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model  
    
class InceptionV3Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=InceptionV3(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model  
    
class DenseNet121Model:
	@staticmethod
	def Build(input_shape, classes):
        
		# initialize the model
		base_model=DenseNet121(input_shape=input_shape,weights='imagenet',include_top=False,input_tensor=Input(shape=(128, 128, 3)))
        
		# Make the base layer untrainable
		for layer in base_model.layers:
			layer.trainable = False
        
		# Add custom layers
		x=base_model.output
		x=GlobalAveragePooling2D()(x)
		x = Dropout(rate = 0.4, name='dropout1')(x)
		x = BatchNormalization()(x)
		x = Dense(512, activation='relu', bias_initializer='zeros')(x)
		x = Dropout(rate = 0.3, name='dropout2')(x)
		x = BatchNormalization()(x)  
		preds = Dense(classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
		model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
		# return the constructed network architecture
		return model     
    
    

class Explore_MLP(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss

    def build(self, hp):
              
        model = Sequential()
        model.add(Dense(hp.Int('Dense_1', 32, 256, step=16, default=32),input_shape=self.input_shape, activation=hp.Choice('Activation_1',['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear'])))
        model.add(Dropout(hp.Float('Dropout_1', 0.0, 0.7, step=0.1, default=0.2)))
        for i in range(hp.Int('n_layers',1,3)):
            model.add(Dense(hp.Int(f"Dense_{i+2}",16,96,step=16, default=16),activation=hp.Choice(f"Activation_{i+2}",['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']), bias_initializer='zeros'))
            model.add(Dropout(hp.Float(f"Dropout_{i+2}", 0.0, 0.7, step=0.1, default=0.2)))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4, 1e-5])),loss=self.loss,metrics=['accuracy'])
        return model

# class Explore_Conv(HyperModel):
#     def __init__(self, input_shape, num_classes,loss):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.loss = loss

#     def build(self, hp):  
#         model = Sequential()
#         model.add(Conv2D(filters=hp.Int('Conv_filter_1', 32, 128, step=32, default=64),
#                          kernel_size=hp.Int('Kernel_size_1', 8, 32, step=8),activation='relu',
#                          input_shape=self.input_shape,
#                          bias_regularizer=regularizers.l2(hp.Float('Bias_regularizer_1',0.0, 0.1, step=0.05))))
#         for i in range(hp.Int('n_layers',1,3)):
#             model.add(Conv2D(hp.Int(f"Conv_filter_{i+2}",16,64,step=16),
#                              kernel_size=hp.Int('Kernel_size_'+ str(i+2), 10, 30,step=4),
#                              activation='relu',
#                              kernel_regularizer=regularizers.l2(hp.Float('Kernel_regularizer_'+ str(i+2),0.0, 0.2, step=0.05)), 
#                              bias_regularizer=regularizers.l2(hp.Float('Bias_regularizer_'+ str(i+2),0.0, 0.2, step=0.05))))
#             model.add(Activation('relu'))
#         model.add(BatchNormalization(axis=-1)) #chanDim
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         model.add(Dense(self.num_classes, activation='softmax'))
#         model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4, 1e-5])),loss=self.loss,metrics=['accuracy'])
#         return model


class Explore_Conv(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss

    def build(self, hp):  
        model = Sequential()
        model.add(Conv2D(filters=hp.Int('Conv_filter_1', 32, 128, step=32, default=64),
                         kernel_size=hp.Int('Kernel_size_1', 8, 32,step=8),activation='relu',
                         input_shape=self.input_shape,
                         bias_regularizer=regularizers.l2(hp.Float('Bias_regularizer_1',0.0, 0.1, step=0.05))))
        
        model.add(Conv2D(hp.Int('Conv_filter_2',16,64,step=16),
                         kernel_size=hp.Int('Kernel_size_2', 10, 30,step=4),activation='relu',
                         kernel_regularizer=regularizers.l2(hp.Float('Kernel_regularizer_2',0.0,0.1,step=0.05)), 
                         bias_regularizer=regularizers.l2(hp.Float('Bias_regularizer_2',0.0,0.1,step=0.05))))
        
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=-1)) #chanDim
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4, 1e-5])),loss=self.loss,metrics=['accuracy'])
        return model
    
    
class Explore_Mobile(HyperModel):
    
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=MobileNet(input_shape=self.input_shape,
                             weights='imagenet',include_top=False,
                            input_tensor=Input(shape=(128, 128, 3))) 
        
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax', 
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model

class Explore_EfficientNetB0(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=efn.EfficientNetB0(input_shape=self.input_shape,
                                      weights='imagenet',include_top=False,
                                      input_tensor=Input(shape=(128, 128, 3))) 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax', 
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model

class Explore_EfficientNetB1(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=efn.EfficientNetB1(input_shape=self.input_shape,
                                      weights='imagenet',include_top=False,
                                      input_tensor=Input(shape=(128, 128, 3))) 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax', 
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model
    
class Explore_EfficientNetB2(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=efn.EfficientNetB2(input_shape=self.input_shape,
                                      weights='imagenet',include_top=False,
                                      input_tensor=Input(shape=(128, 128, 3))) 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax', 
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model
    
class Explore_EfficientNetB3(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=efn.EfficientNetB3(input_shape=self.input_shape,
                                      weights='imagenet',include_top=False,
                                      input_tensor=Input(shape=(128, 128, 3))) 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax', 
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model
    
    
class Explore_EfficientNetB4(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=efn.EfficientNetB4(input_shape=self.input_shape,
                                      weights='imagenet',include_top=False,
                                      input_tensor=Input(shape=(128, 128, 3))) 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax', 
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model
    
class Explore_EfficientNetB5(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=efn.EfficientNetB5(input_shape=self.input_shape,
                                      weights='imagenet',include_top=False,
                                      input_tensor=Input(shape=(128, 128, 3))) 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax', 
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model 
    
class Explore_EfficientNetB6(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=efn.EfficientNetB6(input_shape=self.input_shape,
                                      weights='imagenet',include_top=False,
                                      input_tensor=Input(shape=(128, 128, 3))) 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax', 
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model
    
    
class Explore_EfficientNetB7(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=efn.EfficientNetB7(input_shape=self.input_shape,
                                      weights='imagenet',include_top=False,
                                      input_tensor=Input(shape=(128, 128, 3)))
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax',
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model

class Explore_ResNet50(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=ResNet50(input_shape=self.input_shape,
                            weights='imagenet',include_top=False,
                            input_tensor=Input(shape=(128, 128, 3))) #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax',
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model

class Explore_InceptionV3(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=InceptionV3(input_shape=self.input_shape,
                               weights='imagenet',include_top=False,
                               input_tensor=Input(shape=(128, 128, 3))) 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.7, step=0.1, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.7, step=0.1, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax',
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model

class Explore_DenseNet(HyperModel):
    def __init__(self, input_shape, num_classes,loss):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss = loss
        
    def build(self, hp):
        base_model=DenseNet121(input_shape=self.input_shape,
                               weights='imagenet',include_top=False,
                               input_tensor=Input(shape=(128, 128, 3))) 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dropout(hp.Float('dropout1', 0.0, 0.8, step=0.2, default=0.5))(x)
        x = BatchNormalization()(x)
        x = Dense(hp.Int('hidden_size', 50, 1000, step=100, default=50),
                  activation='relu', bias_initializer='zeros')(x)
        x = Dropout(hp.Float('dropout2', 0.0, 0.8, step=0.2, default=0.2))(x)
        x = BatchNormalization()(x)
        preds = Dense(self.num_classes, activation='softmax', 
                      kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        model=Model(inputs=base_model.input,outputs=preds) 
        #now a model has been created based on our architecture
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4, 1e-5])),
                      loss=self.loss,metrics=['accuracy'])
        return model
    
    
################################### RELATED TO HYPER-PARAMETER TUNING ##################################
    
def Bayesian_optimization_search(model,X_train,y_train,X_test,y_test,X_val,y_val,
                                 bayesian_epoch,max_trials,executions_per_trial,
                                 directory,project_name,aug,outpath,epochs,
                                 model_to_train,classes,finetune,loss,ttkind,Mixed,
                                 finetune_epochs,class_weight,valid_set,init_name):
    
    tuner_Bayesian = BayesianOptimization(model,objective='val_accuracy',
                                          max_trials=max_trials,
                                          executions_per_trial=executions_per_trial,
                                          directory=directory,
                                          project_name=project_name)
    if aug==1:
        datagen =ImageDataGenerator(rotation_range=90,vertical_flip=True,
                                    horizontal_flip=True,zoom_range=0.2,shear_range=10)
        train_generator = datagen.flow(X_train,y_train,batch_size=32)
        validation_generator = datagen.flow(X_test,y_test,batch_size=32)
        tuner_Bayesian.search(train_generator, epochs=bayesian_epoch, 
                              validation_data=validation_generator,verbose = 2,class_weight = class_weight)
    else:
        tuner_Bayesian.search(X_train,y_train, epochs=bayesian_epoch, 
                              validation_data=(X_test, y_test),verbose = 2,class_weight = class_weight)

    selected_hps=tuner_Bayesian.get_best_hyperparameters()[0]
    
    # Build the model with the optimal hyperparameters and train it on the data
    best_model = tuner_Bayesian.hypermodel.build(selected_hps)
    
    # Train the best model
    if Mixed ==1:
        bestmodelpath=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Mixed/Singles/'+ model_to_train
    elif model_to_train == 'mlp':
        bestmodelpath=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Feature/'+ model_to_train
    else:
        bestmodelpath=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Singles/'+ model_to_train
    
    weightnames=bestmodelpath+'/bestweights.hdf5'
    
    if os.path.isfile(weightnames)!=1:
        # finetune is set to zer0
        compile_and_train(best_model,outpath,model_to_train,0.0001,
                          X_train,y_train,X_test,y_test,X_val,y_val,
                          epochs,aug,loss,0,classes,ttkind, 
                          Mixed,class_weight,valid_set,init_name) # here learning rate is not used for training
    
    weightnames2=bestmodelpath+'/bestweights_finetune.hdf5'
    if os.path.isfile(weightnames2)!=1:
        if finetune==1:
            # finetune is set to one
            # Fine tune the best model
            compile_and_train(best_model,outpath,model_to_train,1e-06,
                              X_train,y_train,X_test,y_test,X_val,y_val,
                              finetune_epochs,aug,loss,1,classes,ttkind,
                              Mixed,class_weight,valid_set,init_name) # here learning rate is used 
    elif finetune==2:
            # finetune is set to two
            # Fine tune the best model
            compile_and_train(best_model,outpath,model_to_train,1e-06,
                              X_train,y_train,X_test,y_test,X_val,y_val,
                              finetune_epochs,aug,loss,2,classes,ttkind,
                              Mixed,class_weight,valid_set,init_name) # here learning rate is used 
            
def load_best_model(outpath,foldername,finetune, Mixed,init_name):
    
    if Mixed ==1:
        bestmodelpath=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Mixed/Singles/'+ foldername
    elif foldername == 'mlp':
        bestmodelpath=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Feature/'+ foldername
    else:
        bestmodelpath=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Singles/'+ foldername
          
    if finetune==0:
        modelnames=bestmodelpath+'/keras_model.h5'
        weightnames=bestmodelpath+'/bestweights.hdf5'

    elif finetune==1:
        modelnames=bestmodelpath+'/keras_model_finetune.h5'
        weightnames=bestmodelpath+'/bestweights_finetune.hdf5'

    elif finetune==2:
        modelnames=bestmodelpath+'/keras_model_finetune2.h5'
        weightnames=bestmodelpath+'/bestweights_finetune2.hdf5'
        
    model = keras.models.load_model(modelnames)
    model.load_weights(weightnames)
    
    return model   
            
def compile_and_train(model,outpath,foldername,learning_rate,
                      X_train,y_train,X_test,y_test,X_val,y_val,
                      epochs,aug,loss,finetune,classes,ttkind,
                      Mixed,class_weight,valid_set,init_name):
    
#     print('X_train {}'.format(X_train))
#     print('LENGTH of X_train{}'.format(len(X_train)))
#     print('Shape of X_train{}'.format(X_train.shape))
        
    if Mixed ==1:
        bestmodelpath=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Mixed/Singles/'+ foldername
    elif foldername == 'mlp':
        bestmodelpath=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Feature/'+ foldername
    else:
        bestmodelpath=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Singles/'+ foldername
            
    Path(bestmodelpath).mkdir(parents=True, exist_ok=True)
    np.save(bestmodelpath+'/classes.npy', classes)

    if finetune ==1 and ttkind!='mixed': # Because for mixed we should not change the model parameters
        
        # load the not tuned model
        model=load_best_model(outpath,foldername,finetune=0,Mixed=Mixed,init_name=init_name)
        
#         for layer in model.layers[-15:]:
#             layer.trainable = True
            
        for layer in model.layers:
            layer.trainable = True
        model.compile(loss=loss, 
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                  metrics=["accuracy"])
        callbacks=get_callbacks(bestmodelpath=bestmodelpath,
                                patience=epochs/2,finetune=finetune)

    elif finetune ==2 and ttkind!='mixed': # Because for mixed we should not change the model parameters
        
        # load the not tuned model
        model=load_best_model(outpath,foldername,finetune=1,Mixed=Mixed,init_name=init_name)
        
#         for layer in model.layers[-15:]:
#             layer.trainable = True
            
        for layer in model.layers:
            layer.trainable = True
        model.compile(loss=loss, 
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                  metrics=["accuracy"])
        callbacks=get_callbacks(bestmodelpath=bestmodelpath,
                                patience=epochs/2,finetune=finetune)
        
        
    else:
        model.compile(loss=loss, optimizer='Adam',metrics=["accuracy"])
        callbacks=get_callbacks(bestmodelpath=bestmodelpath,
                                patience=epochs/3,finetune=finetune)

    if aug==1:
        datagen =ImageDataGenerator(rotation_range=90,vertical_flip=True,
                                    horizontal_flip=True,zoom_range=0.2,shear_range=10)
        train_generator = datagen.flow(X_train,y_train,batch_size=32)
        history = model.fit(train_generator,epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks,verbose=2,
                            class_weight=class_weight)   
    elif aug==0:
        history = model.fit(X_train,y_train,epochs=epochs,
                            validation_data=(X_test, y_test),
                            batch_size=256, callbacks=callbacks,verbose = 2,
                            class_weight=class_weight) 

    if finetune==0:
        model.save(bestmodelpath+'/keras_model.h5', overwrite=True, 
                   include_optimizer=True)
        
        plot_acc_loss(history,bestmodelpath,finetune)
        
        with open(bestmodelpath+'/history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        
    elif finetune==1:
        model.save(bestmodelpath+'/keras_model_finetune.h5', overwrite=True, 
                   include_optimizer=True)
        
        plot_acc_loss(history,bestmodelpath,finetune)
        
        with open(bestmodelpath+'/history_finetune', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
    elif finetune==2:
        model.save(bestmodelpath+'/keras_model_finetune2.h5', overwrite=True, 
                   include_optimizer=True)
        
        plot_acc_loss(history,bestmodelpath,finetune)
        
        with open(bestmodelpath+'/history_finetune2', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
    
    print_performance(model,'Report_test',
                      X_test=X_test,
                      y_test=y_test,
                      outpath=outpath,
                      model_to_train=foldername,
                      classes=classes,
                      finetune=finetune,
                      Mixed=Mixed,
                      init_name=init_name)
    
    if valid_set=='yes':
        print_performance(model,'Report_valid',
                          X_test=X_val,
                          y_test=y_val,
                          outpath=outpath,
                          model_to_train=foldername,
                          classes=classes,
                          finetune=finetune,
                          Mixed=Mixed,
                          init_name=init_name)
        
        
def print_performance(model,ReportName,X_test,y_test,outpath,
                      model_to_train,classes,finetune,Mixed,init_name):
    
    y_test_max=y_test.argmax(axis=1)  # The class that the classifier would bet on
    y_test_label=np.array([classes[y_test_max[i]] for i in range(len(y_test_max))],dtype=object)
    
    model_loss, accuracy = model.evaluate(X_test,y_test,verbose = 0)

    probs,predictions_names=get_predictions_names(model,classes,X_test,Mixed)

    # Print and save classification report
    clf_report=classification_report(y_test_label, predictions_names,digits=4)
    conf_matrix=confusion_matrix(y_test_label, predictions_names)
    f1=f1_score(y_test_label, predictions_names, average='macro')

    if Mixed ==1:
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Mixed/Singles/'+ model_to_train +'/'
    elif model_to_train == 'mlp':
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Feature/'+ model_to_train +'/'
    else:
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Singles/'+ model_to_train +'/'
        
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    
    Final_predictions=[probs,y_test,predictions_names,y_test_label]
    
    if finetune ==0:
        f = open(pathname+ReportName+'.txt', 'w')
        f.write('\nModel Name: \n\n{}\n\n Accuracy\n\n{}\n\nTest Loss\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nModel Summary \n\n{}\n'.format(model_to_train,accuracy,model_loss,f1,clf_report, conf_matrix,short_model_summary))
        f.close()
        
        with open(pathname+'/Predictions_'+ReportName+'.pickle', 'wb') as cw:
            pickle.dump(Final_predictions,cw) 

    elif finetune ==1 and Mixed!=1:
        f = open(pathname+ReportName+'_Finetuned.txt', 'w')
        f.write('\nModel Name: \n\n{}\n\n Accuracy\n\n{}\n\nTest Loss\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nModel Summary \n\n{}\n'.format(model_to_train,accuracy,model_loss,f1,clf_report, conf_matrix,short_model_summary))
        f.close()
        
        with open(pathname+'/Predictions_'+ReportName+'_Finetuned.pickle', 'wb') as cw:
            pickle.dump(Final_predictions,cw) 
   
    elif finetune ==2 and Mixed!=1:
        f = open(pathname+ReportName+'_Finetuned2.txt', 'w')
        f.write('\nModel Name: \n\n{}\n\n Accuracy\n\n{}\n\nTest Loss\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nModel Summary \n\n{}\n'.format(model_to_train,accuracy,model_loss,f1,clf_report, conf_matrix,short_model_summary))
        f.close()
        
        with open(pathname+'/Predictions_'+ReportName+'_Finetuned2.pickle', 'wb') as cw:
            pickle.dump(Final_predictions,cw) 


def get_callbacks(bestmodelpath,patience,finetune):
    
    if finetune==0:
        checkpointer    = keras.callbacks.ModelCheckpoint(filepath=bestmodelpath+'/bestweights.hdf5', 
                                                      monitor='val_loss', verbose=0, save_best_only=True)
        logger          = keras.callbacks.CSVLogger(bestmodelpath+'/epochs.log', separator=' ', append=False)
        earlyStopping   = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, 
                                                    restore_best_weights=True)
        
    elif finetune==1:
        checkpointer    = keras.callbacks.ModelCheckpoint(filepath=bestmodelpath+'/bestweights_finetune.hdf5', 
                                                      monitor='val_loss', verbose=0, save_best_only=True) 
        logger          = keras.callbacks.CSVLogger(bestmodelpath+'/epochs_finetune.log', separator=' ', append=False)
        earlyStopping   = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, 
                                                    restore_best_weights=True)
        
    elif finetune==2:
        checkpointer    = keras.callbacks.ModelCheckpoint(filepath=bestmodelpath+'/bestweights_finetune2.hdf5', 
                                                      monitor='val_loss', verbose=0, save_best_only=True) 
        logger          = keras.callbacks.CSVLogger(bestmodelpath+'/epochs_finetune2.log', separator=' ', append=False)
        earlyStopping   = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, 
                                                    restore_best_weights=True)
        
    callbacks=[checkpointer, logger,earlyStopping]
    
    return callbacks
    


            
def get_and_train_best_models(X_train,y_train,X_test,y_test,X_val,y_val,outpath,
                              model_to_train,epochs,aug,classes,bayesian_epoch,
                              max_trials,executions_per_trial,loss,finetune,
                              ttkind,Mixed,finetune_epochs,class_weight,valid_set,init_name):

    if model_to_train=='mlp':
        model = Explore_MLP(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)   
    elif model_to_train=='conv2':
        model = Explore_Conv(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='mobile':
        model = Explore_Mobile(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='eff0':
        model = Explore_EfficientNetB0(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='eff1':
        model = Explore_EfficientNetB1(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='eff2':
        model = Explore_EfficientNetB2(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='eff3':
        model = Explore_EfficientNetB3(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='eff4':
        model = Explore_EfficientNetB4(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='eff5':
        model = Explore_EfficientNetB5(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='eff6':
        model = Explore_EfficientNetB6(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='eff7':
        model = Explore_EfficientNetB7(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='res50':
        model = Explore_ResNet50(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='incepv3':
        model = Explore_InceptionV3(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    elif model_to_train=='dense121':
        model = Explore_DenseNet(input_shape=X_train[0].shape, num_classes=len(classes),loss=loss)
    else:
        print('Check if the model name is right else the requested model is not implemented')
        raise ValueError

    Bayesian_optimization_search(model,
                                 X_train,y_train,
                                 X_test,y_test,
                                 X_val,y_val,
                                 bayesian_epoch=bayesian_epoch,
                                 max_trials=max_trials,
                                 executions_per_trial=executions_per_trial,
                                 directory=outpath+'BayesianSearchTrials/',
                                 project_name=model_to_train,
                                 aug=aug,outpath=outpath,
                                 epochs=epochs,model_to_train=model_to_train,
                                 classes=classes, finetune=finetune,loss=loss,
                                 ttkind=ttkind,Mixed=Mixed,
                                 finetune_epochs=finetune_epochs,
                                 class_weight=class_weight,
                                 valid_set=valid_set,
                                 init_name=init_name)


####################### Average Ensemble #######################################

    
def combine_models(models_image,outpath,finetune,Mixed,init_name):
    members = list()
    
    for i in range(len(models_image)):
        members.append(load_best_model(outpath,foldername=models_image[i],
                                       finetune=finetune,Mixed=Mixed,init_name=init_name))
    return members

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, X_test,classes):
	# make predictions
	yhats = [model.predict(X_test) for model in members]
	yhats = np.array(yhats)
	# sum across ensemble members
	probs = np.sum(yhats, axis=0)
	# argmax across classes
	result = probs.argmax(axis=1)
	result_confidences = probs.max(axis=1)
	result_names=np.array([classes[result[i]] for i in range(result.shape[0])],dtype=object)
	return probs,result_names

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, X_test, y_test_label,classes,models_image):
	# select a subset of members
	subset = members[:n_members]
	members_name=models_image[:n_members]
	# make prediction
	yhat,yhat_names = ensemble_predictions(subset, X_test,classes)
	# calculate accuracy
	return accuracy_score(y_test_label, yhat_names),members_name

def print_ens_performance_metrics(pathname,y_test_label,y_test,probs,
                                  predictions_names,ReportName,foldername):
    
    # Print and save classification report
    clf_report=classification_report(y_test_label, predictions_names,digits=4)
    conf_matrix=confusion_matrix(y_test_label, predictions_names)
    
    acc= accuracy_score(y_test_label, predictions_names)
    f1=f1_score(y_test_label, predictions_names, average='macro')
    
    top_1_acc_1 = top_k_categorical_accuracy(y_test,probs, k=1)
    top_1_acc_1=top_1_acc_1.numpy()
    top_1_acc=sum(top_1_acc_1)/len(top_1_acc_1)
    
    top_2_acc_1 = top_k_categorical_accuracy(y_test,probs, k=2)
    top_2_acc_1=top_2_acc_1.numpy()
    top_2_acc=sum(top_2_acc_1)/len(top_2_acc_1)
    
    top_3_acc_1 = top_k_categorical_accuracy(y_test,probs, k=3)
    top_3_acc_1=top_3_acc_1.numpy()
    top_3_acc=sum(top_3_acc_1)/len(top_3_acc_1)
    
    f = open(pathname+'/'+ ReportName +'.txt', 'w')
    f.write('\n\nModel Name: \n\n{}\n\nAcc\n\n{}\n\nTop_1_Acc\n\n{}\n\nTop_2_Acc\n\n{}\n\nTop_3_Acc\n\n{}\n\n F1 score\n\n{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(foldername,acc,top_1_acc,top_2_acc,top_3_acc,f1,clf_report, conf_matrix))
    f.close()

def Avg_predictions_and_print(members,X_test,y_test,classes,pathname,ReportName,foldername):
    
    predictions=y_test.argmax(axis=1)  # The class that the classifier would bet on
    y_test_label=np.array([classes[predictions[i]] for i in range(len(predictions))],dtype=object)
    
    probs,predictions_names=ensemble_predictions(members,X_test,classes)
    Final_predictions=[probs,predictions_names]
    with open(pathname+'/Probs_and_predictions_'+ReportName+'.pickle', 'wb') as cw:
        pickle.dump(Final_predictions,cw) 
        
    np.save(pathname+'/classes.npy', classes)

    ## Print the performances on text
    print_ens_performance_metrics(pathname,y_test_label,y_test,probs,
                                  predictions_names,ReportName,foldername)
    
def avg_ensemble(X_test,y_test,X_val,y_val,
                 classes,models_image,outpath,
                 finetune,Mixed,valid_set,for_mixed,init_name):
    
    members=combine_models(models_image,outpath,finetune=finetune, Mixed=Mixed,init_name=init_name)
    Models_for_avg_ensemble = '_'.join(map(str, for_mixed))
    
    if finetune==0 and Mixed==0:
        foldername='Ens_of_'+Models_for_avg_ensemble
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Average_Ensemble/' + foldername
        Path(pathname).mkdir(parents=True, exist_ok=True)
    
    elif finetune==1 and Mixed==0:
        foldername='Ens_of_'+Models_for_avg_ensemble
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Finetuned_Average_Ensemble/' + foldername
        Path(pathname).mkdir(parents=True, exist_ok=True)

    elif finetune==2 and Mixed==0:
        foldername='Ens_of_'+Models_for_avg_ensemble
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Finetuned2_Average_Ensemble/' + foldername
        Path(pathname).mkdir(parents=True, exist_ok=True)
        
        
    elif finetune ==0 and Mixed==1:        
        foldername='Ens_of_'+Models_for_avg_ensemble
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Mixed/Average_Ensemble/'+foldername
        Path(pathname).mkdir(parents=True, exist_ok=True)

    Avg_predictions_and_print(members,X_test,y_test,classes,pathname,'Report_test',foldername)
    
    if valid_set=='yes':
        Avg_predictions_and_print(members,X_val,y_val,classes,pathname,'Report_val',foldername)
        

####################### STACKED Ensemble #######################################
        
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, X_test):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(X_test, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX  

def stack_predictions(stacked_model,stackedXtest,y_test,classes,pathname,foldername,ReportName):
    
    predictions_names = stacked_model.predict(stackedXtest)
    probs= stacked_model.predict_proba(stackedXtest)
    
    predictions=y_test.argmax(axis=1)  # The class that the classifier would bet on
    y_test_label=np.array([classes[predictions[i]] for i in range(len(predictions))],dtype=object)
    
    Final_predictions=[probs,y_test, predictions_names,y_test_label]
    
    with open(pathname+'/Predictions'+ReportName+'.pickle', 'wb') as cw:
        pickle.dump(Final_predictions,cw)    
   
    print_ens_performance_metrics(pathname,y_test_label,y_test,probs,
                              predictions_names,ReportName,foldername)

def fit_stacked_ensemble_and_save(stackedXtrain,stackedXtest,stackedXval,
                                  model,foldername,finetune,Mixed,
                                  y_train_label,y_test_label,y_val_label,
                                  outpath,classes,valid_set,init_name,
                                  y_test,y_val):
    
    # fit standalone model on trained stack
    stacked_model = make_pipeline(StandardScaler(),model(max_iter=5000, tol=1e-5))
    stacked_model.fit(stackedXtrain, y_train_label)

    # save the model to disk
    if finetune==0 and Mixed==0:
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Stacking_Ensemble/' + 'Ens_of_'+ foldername
    elif finetune==1 and Mixed==0:
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Finetuned_Stacking_Ensemble/' + 'Ens_of_'+ foldername
        
    elif finetune==2 and Mixed==0:
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Image/Finetuned2_Stacking_Ensemble/' + 'Ens_of_'+ foldername
        
    elif finetune ==0 and Mixed==1:
        pathname=outpath+'BestModelsFromBayesianSearch/'+ init_name +'/Mixed/Stacking_Ensemble/'+'Ens_of_'+ foldername

    Path(pathname).mkdir(parents=True, exist_ok=True)
    filename=pathname +'/model.sav'
    joblib.dump(stacked_model, filename)
    np.save(pathname+'/classes.npy', classes)
        
    stack_predictions(stacked_model,stackedXtest,y_test,classes,pathname,foldername,'test')
    
    if valid_set=='yes':
        stack_predictions(stacked_model,stackedXval,y_val,classes,pathname,foldername,'val')

def stacking_ensemble(X_train,y_train,
                      X_test,y_test,
                      X_val,y_val,
                      classes,models_image,outpath,
                      finetune,valid_set,for_mixed,
                      init_name,Mixed):

    members=combine_models(models_image,outpath,finetune=finetune, Mixed=Mixed,init_name=init_name)
    Models_for_stacking = '_'.join(map(str, for_mixed)) 

    # On train Data
    predictions_train=y_train.argmax(axis=1)  # The class that the classifier would bet on
    y_train_label=np.array([classes[predictions_train[i]] for i in range(len(predictions_train))],dtype=object)
    stackedXtrain = stacked_dataset(members, X_train)
    
    # On test data
    predictions_test=y_test.argmax(axis=1)  # The class that the classifier would bet on
    y_test_label=np.array([classes[predictions_test[i]] for i in range(len(predictions_test))],dtype=object)
    stackedX_test = stacked_dataset(members, X_test)
    
    if valid_set=='yes':
        predictions_val=y_val.argmax(axis=1)  # The class that the classifier would bet on
        y_val_label=np.array([classes[predictions_val[i]] for i in range(len(predictions_val))],dtype=object)
        stackedX_val = stacked_dataset(members, X_val)
        
        fit_stacked_ensemble_and_save(stackedXtrain,stackedX_test,stackedX_val,
                                      LogisticRegression,Models_for_stacking,
                                      finetune,Mixed,
                                      y_train_label,y_test_label,y_val_label,
                                      outpath,classes,valid_set,init_name,y_test,y_val)
    else:
        stackedX_val,y_val_label=[],[]
        fit_stacked_ensemble_and_save(stackedXtrain,stackedX_test,stackedX_val,
                                      LogisticRegression,Models_for_stacking,
                                      finetune,Mixed,
                                      y_train_label,y_test_label,y_val_label,
                                      outpath,classes,valid_set,init_name,y_test,y_val)

        
####################### For Validation counts and selected models #######################################

def combine_models_selected(models_image,outpath,finetune,Mixed,init_name):
    members = list()
    for i in range(len(init_name)):
        members.append(load_best_model(outpath,foldername=models_image[i],
                                          finetune=finetune,Mixed=Mixed,init_name=init_name[i]))
    return members

def combine_models_across_iteration(model_image,outpath,finetune,Mixed,init_name):
    members = list()
    
    for i in range(len(init_name)):
        members.append(load_best_model(outpath,foldername=model_image,
                                       finetune=finetune,Mixed=Mixed,init_name=init_name[i]))
    return members


def avg_ensemble_selected_on_unlabelled(im_names,X_test,classes,models_image,outpath,
                          finetune,Mixed,for_mixed,init_name,path_to_save):
    
    members=combine_models_selected(models_image,outpath,finetune,Mixed,init_name)
    Models_for_avg_ensemble = '_'.join(map(str, for_mixed))
    foldername='Ens_of_'+Models_for_avg_ensemble

    if finetune==0 and Mixed==0:
        pathname=path_to_save+'/Image/Average_Ensemble/' + foldername
    elif finetune==1 and Mixed==0:
        pathname=path_to_save+'/Image/Finetuned_Average_Ensemble/' + foldername
    elif finetune ==0 and Mixed==1:        
        pathname=path_to_save+'/Mixed/Average_Ensemble/'+foldername
        
    Path(pathname).mkdir(parents=True, exist_ok=True)
    probs,predictions_names=ensemble_predictions(members,X_test,classes)
    Final_predictions=[im_names,probs,predictions_names]
    
    To_write = [i + '------------------' + j for i, j in zip(im_names, predictions_names)]
    np.savetxt(pathname+'/Predictions_avg_ens.txt', To_write, fmt='%s')
    
    
    with open(pathname+'/Filenames_Probs_and_Predictions.pickle', 'wb') as cw:
        pickle.dump(Final_predictions,cw)   
    
    return Final_predictions


def stacking_ensemble_selected_on_unlabelled(im_names,X_test,classes,models_image,outpath,
                      finetune,Mixed,init_name,stack_path,for_mixed,path_to_save):
    
    members=combine_models_selected(models_image,outpath,finetune,Mixed,init_name)
    
    Models_for_avg_ensemble = '_'.join(map(str, for_mixed))
    
    filename=stack_path +'/model.sav'
    stacked_model = joblib.load(open(filename, 'rb'))
    
    stackedXtest = stacked_dataset(members, X_test)
    predictions_names = stacked_model.predict(stackedXtest)
    probs= stacked_model.predict_proba(stackedXtest)
    
    foldername='Ens_of_'+Models_for_avg_ensemble
    
    if finetune==0 and Mixed==0:
        pathname=path_to_save+'/Image/Stacking_Ensemble/' + foldername
    elif finetune==1 and Mixed==0:
        pathname=path_to_save+'/Image/Finetuned_Stacking_Ensemble/' + foldername
    elif finetune ==0 and Mixed==1:        
        pathname=path_to_save+'/Mixed/Stacking_Ensemble/'+foldername
    
    Path(pathname).mkdir(parents=True, exist_ok=True)

    Final_predictions=[im_names,probs,predictions_names]
    
    To_write = [i + '------------------' + j for i, j in zip(im_names, predictions_names)]
    np.savetxt(pathname+'/Predictions_stack_ens.txt', To_write, fmt='%s')
    
    with open(pathname+'/Filenames_Probs_and_Predictions.pickle', 'wb') as cw:
        pickle.dump(Final_predictions,cw)   
    
    return Final_predictions

def stacking_ensemble_selected_on_labelled_test(X_test,y_test,classes,models_image,outpath,
                      finetune,Mixed,init_name,stack_path,for_mixed,ReportName):
    
    members=combine_models_selected(models_image,outpath,
                                    finetune,Mixed,init_name)
    
    Models_for_avg_ensemble = '_'.join(map(str, for_mixed))
    
    filename=stack_path +'/model.sav'
    stacked_model = joblib.load(open(filename, 'rb'))
    
    stackedXtest = stacked_dataset(members, X_test)
    predictions_names = stacked_model.predict(stackedXtest)
    probs= stacked_model.predict_proba(stackedXtest)
    
    foldername='Ens_of_'+Models_for_avg_ensemble

    if finetune==0 and Mixed==0:
        pathname=outpath+'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/'\
        'Image/Stacking_Ensemble/' + foldername    
    elif finetune==1:
        pathname=outpath+'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/'\
        'Image/Finetuned_Stacking_Ensemble/' + foldername
    elif finetune ==0 and Mixed==1:        
        pathname=outpath+'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/'\
        'Mixed/Stacking_Ensemble/'+foldername
    
    Path(pathname).mkdir(parents=True, exist_ok=True)

    Final_predictions=[probs,predictions_names]

    with open(pathname+'/Probs_and_predictions_'+ReportName+'.pickle', 'wb') as cw:
        pickle.dump(Final_predictions,cw) 
        
    predictions=y_test.argmax(axis=1)  # The class that the classifier would bet on
    y_test_label=np.array([classes[predictions[i]] for i in range(len(predictions))],dtype=object)

    print_ens_performance_metrics(pathname,y_test_label,y_test,probs,
                              predictions_names,ReportName,foldername)
    
    return Final_predictions



        
####################### Across Initial conditions or selected models #################################

def avg_ensemble_iter_or_selected_models(X_test,y_test,X_val,y_val,classes,models_image,
                                      outpath,finetune,Mixed,valid_set,for_mixed,init_name,sel,ReportName):
    
    if sel=='yes':
        members=combine_models_selected(models_image,outpath,finetune, Mixed,init_name)
        Models_for_avg_ensemble = '_'.join(map(str, for_mixed))
    else:
        members=combine_models_across_iteration(models_image,outpath,finetune, Mixed,init_name)
        Models_for_avg_ensemble = for_mixed #Adapted here to across iterations
    
    foldername='Ens_of_'+Models_for_avg_ensemble

    if finetune==0 and Mixed==0:
        pathname=outpath+'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/'\
        'Image/Average_Ensemble/' + foldername    
    elif finetune==1 and Mixed==0:
        pathname=outpath+'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/'\
        'Image/Finetuned_Average_Ensemble/' + foldername
    elif finetune ==0 and Mixed==1:
        pathname=outpath+'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/'\
        'Mixed/Average_Ensemble/'+foldername
    
    Path(pathname).mkdir(parents=True, exist_ok=True)
    
    Report_on_test='Report_test'+ReportName
    Report_on_valid='Report_valid'+ReportName
    Avg_predictions_and_print(members,X_test,y_test,classes,pathname,Report_on_test,foldername)
    
    if valid_set=='yes':
        Avg_predictions_and_print(members,X_val,y_val,classes,pathname,Report_on_valid,foldername)

def stacking_ensemble_iter_or_selected_models(X_train,y_train,X_test,y_test,X_val,y_val,classes,models_image,
                                              outpath,finetune,Mixed,valid_set,for_mixed,init_name,sel):
    
    if sel=='yes':
        members=combine_models_selected(models_image,outpath,finetune, Mixed,init_name)
        Models_for_stacking = '_'.join(map(str, for_mixed))
    else:
        members=combine_models_across_iteration(models_image,outpath,finetune, Mixed,init_name)
        Models_for_stacking = for_mixed #Adapted here to across iterations
        
    # On train Data
    predictions_train=y_train.argmax(axis=1)  # The class that the classifier would bet on
    y_train_label=np.array([classes[predictions_train[i]] for i in range(len(predictions_train))],dtype=object)
    stackedXtrain = stacked_dataset(members, X_train)
    
    # On test data
    predictions_test=y_test.argmax(axis=1)  # The class that the classifier would bet on
    y_test_label=np.array([classes[predictions_test[i]] for i in range(len(predictions_test))],dtype=object)
    stackedX_test = stacked_dataset(members, X_test)
    
    if valid_set=='yes':
        predictions_val=y_val.argmax(axis=1)  # The class that the classifier would bet on
        y_val_label=np.array([classes[predictions_val[i]] for i in range(len(predictions_val))],dtype=object)
        stackedX_val = stacked_dataset(members, X_val)
        
        fit_stacked_ensemble_and_save_iter(stackedXtrain,stackedX_test,stackedX_val,
                                      LogisticRegression,Models_for_stacking,finetune,Mixed,
                                      y_train_label,y_test_label,y_val_label,
                                      outpath,classes,valid_set,y_test,y_val)
    else:
        stackedX_val,y_val_label=[],[]
        fit_stacked_ensemble_and_save_iter(stackedXtrain,stackedX_test,stackedX_val,
                                      LogisticRegression,Models_for_stacking,
                                      finetune,Mixed,
                                      y_train_label,y_test_label,y_val_label,
                                      outpath,classes,valid_set,y_test,y_val)
        
def fit_stacked_ensemble_and_save_iter(stackedXtrain,stackedXtest,stackedXval,
                                       model,foldername,finetune,Mixed,
                                       y_train_label,y_test_label,y_val_label,
                                       outpath,classes,valid_set,y_test,y_val):
    
    # fit standalone model on trained stack
    stacked_model = make_pipeline(StandardScaler(),model(max_iter=5000, tol=1e-5))
    stacked_model.fit(stackedXtrain, y_train_label)

    # save the model to disk
    if finetune==0 and Mixed==0:
        pathname=outpath+'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/'\
        'Image/Stacking_Ensemble/' + 'Ens_of_'+ foldername
    elif finetune==1:
        pathname=outpath+'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/'\
        'Image/Finetuned_Stacking_Ensemble/' + 'Ens_of_'+ foldername
    elif finetune ==0 and Mixed==1:
        pathname=outpath+'BestModelsFromBayesianSearch/For_each_model_across_Iterations_and_selected_models/'\
        'Mixed/Stacking_Ensemble/'+'Ens_of_'+ foldername
    Path(pathname).mkdir(parents=True, exist_ok=True)
    filename=pathname +'/model.sav'
    joblib.dump(stacked_model, filename)
    np.save(pathname+'/classes.npy', classes)

    stack_predictions(stacked_model,stackedXtest,y_test,y_test_label,pathname,foldername,'test')
    
    if valid_set=='yes':
        stack_predictions(stacked_model,stackedXval,y_val,y_val_label,pathname,foldername,'val')
  
        
####################### MIXED Models #######################################
            
def define_Mixed_model(members,num_classes,loss):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(100, activation='relu')(merge)
	output = Dense(num_classes, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
    
	# compile
# 	model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
	return model

def define_Mixed_model_from_scratch(members,num_classes,loss):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
		for layer in model.layers[-10:]:
			layer.trainable = True
			# rename to avoid 'unique layer name' issue
			layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(100, activation='relu')(merge)
	output = Dense(num_classes, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
    
	# compile
# 	model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
	return model

def Select_Mixed_Model(outpath,foldername,FeaturePath,finetune,Mixed,init_name):
    members = list()
    members.append(load_best_model(outpath,foldername,finetune,0,init_name)) # load from non-mixed model, i.e. Mixed=0
    members.append(load_best_model(outpath,FeaturePath,finetune,0,init_name)) # load from non-mixed model, i.e. Mixed=0
    return members

def Select_Mixed_Model_from_scratch(X_train,classes,ImageName):
    members = list()
    
    if ImageName == 'conv2' or ImageName == 'conv':
        members.append(Conv2Layer.Build(input_shape=X_train[0][0].shape, 
                                        classes=len(classes), last_activation='softmax'))
    elif ImageName == 'mobile':
        members.append(MobileNetModel.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'eff0':
        members.append(EfficientNetB0Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'eff1':
        members.append(EfficientNetB1Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'eff2':
        members.append(EfficientNetB2Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'eff3':
        members.append(EfficientNetB3Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'eff4':
        members.append(EfficientNetB4Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'eff5':
        members.append(EfficientNetB5Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'eff6':
        members.append(EfficientNetB6Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'eff7':
        members.append(EfficientNetB7Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'res50':
        members.append(ResNet50Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))    
    elif ImageName == 'incepv3':
        members.append(InceptionV3Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))
    elif ImageName == 'dense121':
        members.append(DenseNet121Model.Build(input_shape=X_train[0][0].shape, classes=len(classes)))    
    elif ImageName == 'smallvgg':
        members.append(SmallVGGNetModel.Build(input_shape=X_train[0][0].shape, 
                                              classes=len(classes), last_activation = 'softmax'))  
    members.append(MultiLayerPerceptron.Build2Layer(input_shape=X_train[1][0].shape, 
                                                    classes=len(classes),last_activation='softmax',layers=[64,32]))        
    return members


def get_predictions_names(model,classes,X_test,Mixed):
    
    probs = model.predict(X_test)
    predictions=probs.argmax(axis=1)  # The class that the classifier would bet on
    confidences=probs.max(axis=1)     # equivalent to: [probs[i][predictions[i]] for i in range(len(probs))] 
    
    if Mixed==1:
        predictions_names=np.array([classes[predictions[i]] for i in range(len(X_test[0]))],dtype=object)
    else:
        predictions_names=np.array([classes[predictions[i]] for i in range(len(X_test))],dtype=object)

#     print('CLASSES {}'.format(classes))
#     print('LENGTH of X_test{}'.format(len(X_test[0])))
#     print('Shape of X TEST{}'.format(X_test[0].shape))
#     print('PROBS{}'.format(probs.shape))
#     print('PREDICTIONS{}'.format(predictions))
#     print('PREDICTIONS_NAMES {}'.format(predictions_names))

    return probs,predictions_names  


def plot_acc_loss(history,bestmodelpath,finetune):
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Feature accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if finetune==1:
        plt.savefig(bestmodelpath+'/Accuracy_finetuned.png')
        plt.close() 
    elif finetune==2:
        plt.savefig(bestmodelpath+'/Accuracy_finetuned2.png')
        plt.close() 
    else:
        plt.savefig(bestmodelpath+'/Accuracy.png')
        plt.close() 
        
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Feature accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.xscale('log')
    if finetune==1:
        plt.savefig(bestmodelpath+'/LogX_Accuracy_finetuned.png')
        plt.close() 
    elif finetune==2:
        plt.savefig(bestmodelpath+'/LogX_Accuracy_finetuned2.png')
        plt.close() 
    else:
        plt.savefig(bestmodelpath+'/LogX_Accuracy.png')
        plt.close() 
        
    # summarize history for loss
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Feature loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
    if finetune==1:
        plt.savefig(bestmodelpath+'/Loss_finetuned.png')
        plt.close() 
    elif finetune==2:
        plt.savefig(bestmodelpath+'/Loss_finetuned2.png')
        plt.close() 
    else:
        plt.savefig(bestmodelpath+'/Loss.png')
        plt.close() 

    # summarize history for loss
    plt.figure(3)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Feature loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.xscale('log')
#     plt.show()
    if finetune==1:
        plt.savefig(bestmodelpath+'/LogX_Loss_finetuned.png')
        plt.close() 
    elif finetune==2:
        plt.savefig(bestmodelpath+'/LogX_Loss_finetuned2.png')
        plt.close() 
    else:
        plt.savefig(bestmodelpath+'/LogX_Loss.png')
        plt.close()   


def get_testdirs_and_labels(testpaths,classifier,class_select):
    testdirs=[]
    Class_labels=[]   
    individual_labels_original = []
    individual_names_original  = []
    Class_labels_original = []
    
    for idata in range(len(testpaths)):
        for path in os.listdir(testpaths[idata]):
            full_path = os.path.join(testpaths[idata], path)
            testdirs.append(full_path)
            Class_labels.append(path)
            
    Class_labels_original=Class_labels

    if classifier=='multi':
        if class_select is None:
            Class_labels=Class_labels
            testdirs=testdirs
        else:
            Class_labels = [classes for classes in Class_labels if classes in class_select]
            Indices = [i for i, x in enumerate(Class_labels) if any(thing in x for thing in class_select)]
            testdirs = [i for j, i in enumerate(testdirs) if j in Indices]
    
    elif classifier=='binary':
        if class_select is None:
            Class_labels=Class_labels
            testdirs=testdirs
        else:
            Class_labels = [classes for classes in Class_labels if classes in class_select]
            Indices = [i for i, x in enumerate(Class_labels) if any(thing in x for thing in class_select)]
            testdirs = [i for j, i in enumerate(testdirs) if j in Indices]
            
    elif classifier=='versusall':
        
        negative_class_name='Not_' + '_'.join(class_select) 
        positive_class_name='_'.join(class_select)
        testdirs=testdirs
        Class_labels=[negative_class_name if x not in class_select else x for x in Class_labels]
        Class_labels=[positive_class_name if x in class_select else x for x in Class_labels]
        
#     for itd,td in enumerate(testdirs):
#         im_names_here_original = np.array(glob.glob(td+'/*.jpeg'),dtype=object) 
#         individual_names_original.extend( im_names_here_original)
#         individual_labels_original.extend([Class_labels_original[itd] for i in range(len(im_names_here_original))])
    
#     individual_names_original = np.array(individual_names_original)
#     individual_labels_original = np.array(individual_labels_original)
    
    return Class_labels,Class_labels_original,testdirs#,individual_labels_original,individual_names_original

def get_hist_of_val_counts(predictions,tax_path,save_fig_path,logy):
    # Create a dataframe with the classification counts
    histo_cla = Counter(predictions[1])
    df_cla=pd.DataFrame(histo_cla.values(), index=histo_cla.keys(), columns=['cla'])
    
    #
    # Data from taxonomists
    #
    df_tax=pd.read_csv(tax_path+'counts_cleaned.csv', sep=';',encoding="ISO-8859-1")                                                                             
    useless_cols = ['name_folder', 'ID','timestamp','year', 'month', 'day', 'hour', 'name_counter', 'total_ROI', 'comments']
    classes = list(set(df_tax.columns)-set(useless_cols))
    df_tax = pd.DataFrame(df_tax[classes].sum().values, index=df_tax[classes].sum().index, columns=['tax'])
    print('According to the taxonomists, there are {} images'.format(df_tax.sum().values.item()))

    #
    # Concatenate and plot the two histograms
    #

    df=pd.concat([df_cla, df_tax], axis=1, ignore_index=False, sort=True)
    ax=df[['cla','tax',]].sort_values(by='tax', ascending=False).plot.bar(logy=logy)
    ax.set_ylabel("Taxon count")
    plt.tight_layout()
    plt.show()
    plt.savefig(save_fig_path+'Validation_counts.png')

    #
    # Make a nicer plot that excludes classes that are not present in either tax or cla
    #
    ax=df[(df.tax>0) & (df.cla>0)].sort_values(by='tax', ascending=False).plot.bar(logy=logy)
    ax.set_ylabel("Taxon count")
    plt.tight_layout()
    plt.show()
    plt.savefig(save_fig_path+'Validation_counts_clean.png')
    
    
    Sorted=df[['cla','tax']].sort_values(by='tax', ascending=False)

    Ratios=Sorted['cla']/Sorted['tax']
    
    return Sorted,Ratios
