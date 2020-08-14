# import the necessary packages
import keras
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import concatenate
from keras import backend as K
from keras import metrics as metrics
import numpy as np

def CreateParams(layers= None, lr =None, bs=None, optimizer='sgd', totEpochs= None, dropout=None, callbacks= None, 
				 initial_epoch=0, aug=None, modelfile=None, model_feat='mlp', model_image='mlp', load_weights=None, 
				 override_lr=False, train=True, numclasses=None):
	''' Creates an empty dictionary with all possible entries'''

	params={
		'layers': layers,
        'lr': lr,
        'bs': bs,
        'optimizer': optimizer,
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
		'numclasses': numclasses, # If no labels are given, we must give the number of classes through this variable
		}

	return params




class CModelWrapper:
	'''
	A wrapper class for models
	'''
	def __init__(self, trainX, trainY, testX, testY, params, verbose=False):

		(self.trainX, self.trainY, self.testX, self.testY, self.params, self.verbose) = (trainX, trainY, testX, testY, params, verbose)

		self.numclasses = len(trainY[0]) if (params['numclasses'] is None) else params['numclasses']

		self.SetArchitecture()   # Defines self.model
		self.InitModelWeights()
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

		if (self.params['load_weights'] is None) and (self.params['modelfile']==None):
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

		self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])
		
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
									epochs=self.params['totEpochs'], 
									batch_size=self.params['bs'], 
									callbacks=self.params['callbacks'],
									initial_epoch = self.params['initial_epoch'])
			else:
				assert(self.modelkind!='feat', self.modelkind!='mixed', "We only augment with image data")
				self.history = self.model.fit_generator(
									self.params['aug'].flow(self.trainX, self.trainY, batch_size=self.params['bs']), 
									validation_data=(self.testX, self.testY), 
									epochs=self.params['totEpochs'], 
									callbacks=self.params['callbacks'],
									initial_epoch = self.params['initial_epoch'],
									steps_per_epoch=len(self.trainX)//self.params['bs']
									)

		else: #params['train']==False here
			self.history=None

		return


# def PlainModel(trainX, trainY, testX, testY, params):
# 	'''
# 	A wrapper for models that use feature-only or image-only data
# 	'''

# 	numclasses = len(trainY[0]) if (params['numclasses'] is None) else params['numclasses']

# 	#
# 	# Define model architecture
# 	#
# 	if params['model'] is None:

# 		# See whether input is images or features
# 		if len(np.shape(trainX[0]))==3:
# 			modelkind = params['model_image']
# 		elif len(np.shape(trainX[0]))==1:
# 			modelkind = params['model_feat']
# 		else:
# 			raise RuntimeError('PlainModel(): The shape of the input is neither 1D (feat) nor 3D (image)')


# 		# Define model
# 		if modelkind == 'mlp':
# 			model = MultiLayerPerceptron.Build2Layer(input_shape=trainX[0].shape, classes=numclasses, layers=params['layers'])
# 		elif modelkind == 'conv2':
# 			model = Conv2Layer.Build(input_shape=trainX[0].shape, classes=numclasses, last_activation='softmax')
# 		elif modelkind == 'smallvgg':
# 			model = SmallVGGNet.Build(input_shape=trainX[0].shape, classes=numclasses)
# 		else:
# 			raise NotImplementedError('PlainModel() - chosen model is not implemented')

# 		# Initialize weights
# 		if params['load_weights'] is None:
# 			print('WARNING: At the current state, we are taking the default weight initialization, whatever it is. This must change.')
# 		else:
# 			print('Loading weights from ',params['load_weights'])
# 			model.load_weights(params['load_weights'])
		
# 		# Set Optimizer
# 		if params['optimizer'] == 'sgd':
# 			optimizer=keras.optimizers.SGD(lr=params['lr'], nesterov=True)
# 		elif params['optimizer'] == 'adam':
# 			optimizer = keras.optimizers.Adam(learning_rate=params['lr'], beta_1=0.9, beta_2=0.999, amsgrad=False)

# 		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


# 	else:
# 		model = params['model']
# 		if params['load_weights'] is not None:
# 			model.load_weights(params['load_weights'])
# 		# print('LR of the loaded model:', K.get_value(model.optimizer.lr))
# 		# if params['override_lr']==True:
# 		# 	K.set_value(model.optimizer.lr, params['lr'])
# 		# 	print('Setting the LR to', params['lr'])

# 	if params['train']:

# 		if params['aug'] is None:

# 			history = model.fit(
# 								trainX, trainY, 
# 								validation_data=(testX, testY), 
# 								epochs=params['totEpochs'], 
# 								batch_size=params['bs'], 
# 								callbacks=params['callbacks'],
# 								initial_epoch = params['initial_epoch'])
# 		else:
# 			history = model.fit_generator(
# 								params['aug'].flow(trainX, trainY, batch_size=params['bs']), 
# 								validation_data=(testX, testY), 
# 								epochs=params['totEpochs'], 
# 								callbacks=params['callbacks'],
# 								initial_epoch = params['initial_epoch'],
# 								steps_per_epoch=len(trainX)//params['bs']
# 								)

# 	else: #params['train']==False here
# 		history=None

# 	return history, model



def MixedModel(trainX, trainY, testX, testY, params):
	'''
	A wrapper for models that use mixed data
	'''

	nclasses=len(trainY[0])

	# We assume that trainX and testX are in the format: [images, features]
	trainXi, trainXf = trainX[0], trainX[1]
	testXi , testXf  =  testX[0],  testX[1]
	assert(len(trainXi.shape)>len(trainXf.shape))
	assert(len( testXi.shape)>len( testXf.shape))

	# Number of output nodes of the image and feature branches (will become a user parameter)
	nout_f = params['layers'][1]
	nout_i = params['layers'][1]


	## First branches - features and images separate
	# Set model for first branch on features
	if params['model_feat'] == 'mlp':
		model_feat = MultiLayerPerceptron.Build2Layer(
			input_shape=trainXf[0].shape , classes=nout_f, last_activation = 'sigmoid', layers=params['layers'])
	else: 
		raise NotImplementedError

	# Set model for first branch on images
	if params['model_image'] == 'mlp':
		model_image = MultiLayerPerceptron.Build2Layer(
			input_shape=trainXi[0].shape, classes=nout_i, last_activation = 'sigmoid', layers=params['layers'])
	elif params['model_image'] == 'conv2':
		model_image = Conv2Layer.Build(
			input_shape=trainXi[0].shape, classes=nout_i, last_activation = 'sigmoid')
	elif params['model_image'] == 'smallvgg':
		model_image = SmallVGGNet.Build(
			input_shape=trainXi[0].shape, classes=nout_i, last_activation = 'sigmoid')
	else: 		
		raise NotImplementedError

	## Second branch - join features and images
	combinedInput = concatenate([model_image.output, model_feat.output]) # Combine the two
	model_join = Dense(64, activation="relu")(combinedInput)
	model_join = Dense(nclasses, activation="softmax")(model_join)				
	model = Model(inputs=[model_image.input, model_feat.input], outputs=model_join)

	if params['optimizer'] == 'sgd':
		optimizer=keras.optimizers.SGD(lr=params['lr'], nesterov=True)
	elif params['optimizer'] == 'adam':
		optimizer = keras.optimizers.Adam(learning_rate=params['lr'], beta_1=0.9, beta_2=0.999, amsgrad=False)

	model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

	if params['aug'] is None:
	
		history = model.fit(
							[trainXi,trainXf], trainY, 
							validation_data=([testXi,testXf], testY), 
							epochs=params['totEpochs'], 
							batch_size=params['bs'], 
							callbacks=params['callbacks'],
							initial_epoch = params['initial_epoch']
							)

	else: #here, params['aug'] is set

		history = model.fit_generator(
							params['aug'].flow([trainXi,trainXf], trainY, batch_size=params['bs']), 
							validation_data=([testXi,testXf], testY), 
							epochs=params['totEpochs'], 
							callbacks=params['callbacks'],
							initial_epoch = params['initial_epoch'],
							steps_per_epoch=len(trainXi)//params['bs']
							)

	return history, model


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
		elif modelnames[0] == 'smallvgg':
			model_image = SmallVGGNet.Build(
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

