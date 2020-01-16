# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallTomNet:
	@staticmethod
	def build(width, height, depth, classes):
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


		model.add(Conv2D(64, kernel_size=24, activation='relu', input_shape=(height,width,depth))) # This layer gives a warning, which can be ignored, as per qlzh727's comment in https://github.com/tensorflow/tensorflow/issues/30263
		model.add(Conv2D(32, kernel_size=12, activation='relu'))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(classes, activation='softmax'))

		# return the constructed network architecture
		return model

	@staticmethod
	def buildDropout(width, height, depth, classes):
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


		model.add(Conv2D(64, kernel_size=24, activation='relu', input_shape=(height,width,depth))) # This layer gives a warning, which can be ignored, as per qlzh727's comment in https://github.com/tensorflow/tensorflow/issues/30263
		model.add(Conv2D(32, kernel_size=12, activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(classes, activation='softmax'))

		# return the constructed network architecture
		return model

	@staticmethod
	def buildLinearRegression(classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		model.add(Flatten())
		model.add(Dense(classes, activation='softmax'))

		# return the constructed network architecture
		return model