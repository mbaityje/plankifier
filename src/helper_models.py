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

def CreateParams(layers= None, lr =None, bs=None, optimizer='sgd', totEpochs= None, dropout=None, callbacks= None, initial_epoch=0, aug=None, model=None, model_feat='mlp', model_image='mlp', load_weights=None, override_lr=False, train=True, numclasses=None):
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
        'model': model, # For mixed models, what the image branch gets
        'model_feat': model_feat, # For mixed models, what the feature branch gets
        'model_image': model_image, # For mixed models, what the image branch gets
		'load_weights': load_weights, # If you want to load weights from file, put the filename (with path) here
		'override_lr': override_lr, # Whether to load model from file
		'train': train, # Whether to train the model (e.g. maybe you only want to load it)
		'numclasses': numclasses, # If no labels are given, we must give the number of classes through this variable
		}

	return params



def PlainModel(trainX, trainY, testX, testY, params):
	'''
	A wrapper for models that use feature-only or image-only data
	'''

	numclasses = len(trainY[0]) if (params['numclasses'] is None) else params['numclasses']

	#
	# Define model architecture
	#
	if params['model'] is None:

		print('np.shape(trainX[0]):',np.shape(trainX[0]))
        if len(np.shape(trainX[0]))==3:
		    modelkind = params['model_image']
        elif len(np.shape(trainX[0]))==1:
            modelkind = params['model_feat']
        else:
            raise RuntimeError('PlainModel(): The shape of the input is neither 1D (feat) nor 3D (image)')


		# Define model
		if modelkind == 'mlp':
			model = MultiLayerPerceptron.Build2Layer(input_shape=trainX[0].shape, classes=numclasses, layers=params['layers'])
		elif modelkind == 'conv2':
			model = Conv2Layer.Build(input_shape=trainX[0].shape, classes=numclasses, last_activation='softmax')
		elif modelkind == 'smallvgg':
			model = SmallVGGNet.Build(input_shape=trainX[0].shape, classes=numclasses)
		else:
			raise NotImplementedError('PlainModel() - chosen model is not implemented')

		# Initialize weights
		if params['load_weights'] is None:
			print('At the current state, we are taking the default weight initialization, whatever it is. This must change.')
		else:
			model.load_weights(params['load_weights'])
		
		# Set Optimizer
		if params['optimizer'] == 'sgd':
			optimizer=keras.optimizers.SGD(lr=params['lr'], nesterov=True)
		elif params['optimizer'] == 'adam':
			optimizer = keras.optimizers.Adam(learning_rate=params['lr'], beta_1=0.9, beta_2=0.999, amsgrad=False)

		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


	else:
		model = params['model']
		if params['load_weights'] is not None:
			model.load_weights(params['load_weights'])
		# print('LR of the loaded model:', K.get_value(model.optimizer.lr))
		# if params['override_lr']==True:
		# 	K.set_value(model.optimizer.lr, params['lr'])
		# 	print('Setting the LR to', params['lr'])

	if params['train']:

		if params['aug'] is None:

			history = model.fit(
								trainX, trainY, 
								validation_data=(testX, testY), 
								epochs=params['totEpochs'], 
								batch_size=params['bs'], 
								callbacks=params['callbacks'],
								initial_epoch = params['initial_epoch'])
		else:
			history = model.fit_generator(
								params['aug'].flow(trainX, trainY, batch_size=params['bs']), 
								validation_data=(testX, testY), 
								epochs=params['totEpochs'], 
								callbacks=params['callbacks'],
								initial_epoch = params['initial_epoch'],
								steps_per_epoch=len(trainX)//params['bs']
								)

	else: #params['train']==False here
		history=None

	return history, model



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

