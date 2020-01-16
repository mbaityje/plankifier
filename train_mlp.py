#!/usr/bin/env python3
# 
# Train on zooplankton data with a simple multi-layer perceptron model.
# 
# Launch as:
# 	python train_mlp.py -xtype='flat' -lr=0.1 -totEpochs=200 -width=64 -height=64 -plot
#
# 
#########################################################################

import os, subprocess, time, argparse, numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from src.models.smallvggnet import SmallVGGNet
from src.models.smallTom import SmallTomNet
# import cv2 # See here for installation on Debian: https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
parser.add_argument('-verbose', action='store_true', help="Print many messages on screen.")
parser.add_argument('-plot', action='store_true', help="Plot loss and accuracy during training once the run is over.")
parser.add_argument('-totEpochs', type=int, default=5, help="Total number of epochs for the training")
parser.add_argument('-xtype', choices=['flat'], default='flat', help="Whether the model accepts 2D images (e.g. convnet) or flattened data (e.g. MLP)")
parser.add_argument('-bs', type=int, default=16, help="Batch size")
parser.add_argument('-lr', type=float, default=0.001, help="Learning Rate")
parser.add_argument('-height', type=int, default=64, help="Image height")
parser.add_argument('-width', type=int, default=64, help="Image width")
parser.add_argument('-depth', type=int, default=3, help="Number of channels")
parser.add_argument('-valSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
args=parser.parse_args()

if args.verbose:
	ngpu=len(keras.backend.tensorflow_backend._get_available_gpus())
	print('We have {} GPUs'.format(ngpu))
# np.random.seed(12345678)

#########
# PATHS #
#########
rootDir = './'
dataDir = rootDir+'/data/zooplankton_trainingset/'
outDir  = rootDir+'/out'

######################
# DERIVED PARAMETERS #
######################
all_image_fnames   = [os.listdir(dataDir+'/'+directory) for directory in os.listdir(dataDir)]
num_classes        = len(all_image_fnames)
num_images         = np.sum([len(f) for f in all_image_fnames])
train_images       = int((1-args.valSplit)*num_images)
val_images         = int(args.valSplit*num_images)
n_processed_images = args.totEpochs*num_images
tot_steps          = n_processed_images//args.bs # number of minimization steps (i.e. times the gradient is calculated)
if args.verbose:
	print('There are {} images in the dataset.'.format(num_images))
	print(train_images,'go in the training dataset')
	print(val_images,'go in the validation dataset')
	print('num_classes:',num_classes)
	print('tot epochs:',args.totEpochs)
	print('n_processed_images:',n_processed_images)
	print('tot_steps:',tot_steps)

########
# DATA #
########
data,labels = [], np.array([])
classes = {'name': os.listdir(dataDir)}
classes['num']    = len(classes['name'])
classes['num_ex'] =  np.zeros(classes['num'], dtype=int)
for ic in range(classes['num']):
	c=classes['name'][ic]
	classPath=dataDir+c+'/'
	if args.verbose: print('class:',c)
	classImages = os.listdir(classPath)
	classes['num_ex'][ic] = len(classImages) # number of examples per class
	for imageName in classImages:
		imagePath = classPath+imageName

		image = Image.open(imagePath).resize((args.width,args.height))
		npimage = np.array(image.copy() )

		if args.xtype == 'flat':
			npimage = npimage.flatten()
		data.append(npimage)
		image.close()
	labels=np.concatenate(( labels, np.full(classes['num_ex'][ic], ic) ), axis=0)
classes['tot_ex'] =  classes['num_ex'].sum()
data = np.array(data, dtype="float") / 255.0 # scale the raw pixel intensities to the range [0, 1]
labels = np.array(labels)
#shuffle data (in a second moment the shuffling will need to take into account class imbalance)
p=np.random.permutation(classes['tot_ex'])
data=data[p]
labels=labels[p]



#Split train and test
(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=args.valSplit, random_state=42)
if args.verbose:
	print('We expect the training   examples ({}) to be {}'.format(train_generator.samples,train_images))
	print('We expect the validation examples ({}) to be {}'.format(validation_generator.samples,val_images))


# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


model = Sequential()
model.add(Dense(1024, input_shape=(len(data[0]),), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))


# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = keras.optimizers.SGD(lr=args.lr)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# checkpoints
filepath = outDir+'/weights_epoch{epoch:03d}.hdf5' # make sure that the callback filepath exists, since it won't create directories
checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy


# train the neural network
start=time.time()
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=args.totEpochs, batch_size=args.bs,    callbacks=[checkpointer])
end=time.time()
print('Training took',(end-start)/60,'minutes')

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=classes['name']))

# Plot loss and accuracy during training
if args.plot:
	plt.figure(1)
	plt.title('Model loss during training')
	plt.plot(np.arange(1,args.totEpochs+1),history.history['loss'], label='train')
	plt.plot(np.arange(1,args.totEpochs+1),history.history['val_loss'], label='test')
	plt.xlabel('epoch')
	plt.xlabel('loss')
	plt.legend()
	plt.figure(2)
	plt.title('Model accuracy during training')
	plt.ylim((0,1))
	plt.plot(np.arange(1,args.totEpochs+1),history.history['accuracy'], label='train')
	plt.plot(np.arange(1,args.totEpochs+1),history.history['val_accuracy'], label='test')
	plt.plot(np.arange(1,args.totEpochs+1),np.ones(args.totEpochs)/num_classes, label='random', color='black', linestyle='-.')
	plt.xlabel('epoch')
	plt.xlabel('loss')
	plt.legend()
	plt.show()
