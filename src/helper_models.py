# import the necessary packages
import keras
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers import concatenate
from keras import backend as K


def CreateParams(layers= None, lr =None, bs=None, totEpochs= None, callbacks= None, initial_epoch=0, aug=None, model='mlp', model_feat='mlp', model_image='mlp'):
	''' Creates an empty dictionary with all possible entries'''

	params={
		'layers': layers,
        'lr': lr,
        'bs': bs,
        'totEpochs': totEpochs,
        'callbacks': callbacks,
        'initial_epoch': initial_epoch,
        'aug': aug,
        'model': model, # For mixed models, what the image branch gets
        'model_feat': model_feat, # For mixed models, what the feature branch gets
        'model_image': model_image # For mixed models, what the image branch gets
		}

	return params

def PlainModel(trainX, trainY, testX, testY, params):
	'''
	A wrapper for models that use feature-only or image-only data
	'''


	if params['model'] == 'mlp':
		model = MultiLayerPerceptron.Build2Layer(input_shape=trainX[0].shape, classes=len(trainY[0]), layers=params['layers'])
	elif params['model'] == 'conv2':
		model = Conv2Layer.Build(input_shape=trainX[0].shape, classes=len(trainY[0]), last_activation='softmax')
	else:
		raise NotImplementedError('PlainModel() - chosen model is not implemented')

	optimizer=keras.optimizers.SGD(lr=params['lr'], nesterov=True)
    
	model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

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

	## First branches - features and images separate
	# Set model for first branch on features
	if params['model_feat'] == 'mlp':
		model_feat = MultiLayerPerceptron.Build2Layer(input_shape=trainXf[0].shape , classes=None, layers=params['layers'])
	else: 
		raise NotImplementedError

	# Set model for first branch on images
	if params['model_image'] == 'mlp':
		model_image= MultiLayerPerceptron.Build2Layer(input_shape=trainXi[0].shape , classes=None, layers=params['layers'])
	elif params['model_image'] == 'conv2':
		model_image= Conv2Layer.Build(input_shape=trainXi[0].shape, classes=params['layers'][1], last_activation = 'sigmoid')
	else: 		
		raise NotImplementedError


	## Second branch - join features and images
	combinedInput = concatenate([model_image.output, model_feat.output]) # Combine the two
	model_join = Dense(64, activation="relu")(combinedInput)
	model_join = Dense(nclasses, activation="softmax")(model_join)				
	model = Model(inputs=[model_image.input, model_feat.input], outputs=model_join)

	optimizer=keras.optimizers.SGD(lr=params['lr'], nesterov=True)

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

	else:

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
	def Build2Layer(input_shape, classes, layers=[64,32]):
		model = Sequential()
		if len(input_shape)==1:
			model.add(Dense(layers[0], input_shape=input_shape, activation="sigmoid"))
		else:
			model.add( Flatten(input_shape=input_shape ) )
			model.add(Dense(layers[0], activation="sigmoid"))
		model.add(Dense(layers[1], activation="sigmoid"))
		if classes != None:
			model.add(Dense(classes, activation="softmax"))
		return model

class Conv2Layer:
	@staticmethod
	def Build(input_shape, classes, last_activation='softmax'):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		# inputShape = (height, width, depth) # This should be the normal situation -  we use channels last
		# chanDim = -1

		# # if we are using "channels first", update the input shape
		# # and channels dimension
		# if K.image_data_format() == "channels_first":
		# 	inputShape = (depth, height, width)
		# 	chanDim = 1

		# Beware, kernel_size is hard coded for the moment, so it might not work if images are small
		model.add(Conv2D(64, kernel_size=24, activation='relu', input_shape=input_shape))
		model.add(Conv2D(32, kernel_size=12, activation='relu'))
		model.add(BatchNormalization(axis=-1))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.25))
		model.add(Flatten())

		model.add(Dense(classes, activation=last_activation))

		return model


class SmallVGGNet:
	@staticmethod
	def Build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
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
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

