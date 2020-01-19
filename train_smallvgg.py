#!/usr/bin/env python3
# 
# Train on zooplankton data with a simple multi-layer perceptron model.
#
# Optimizer: SGD with no weight decay.
# 
# Launch as:
# 	python train_smallvgg.py -lr=0.1 -totEpochs=100 -width=128 -height=128 -plot
#
# 
#########################################################################

import os, pathlib, time, datetime, argparse, numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt, seaborn as sns
from src.models.smallvggnet import SmallVGGNet
from src.helper_data import ResizeWithProportions
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
parser.add_argument('-datapath', default='./data/zooplankton_trainingset/', help="Print many messages on screen.")
parser.add_argument('-outpath', default='./out/', help="Print many messages on screen.")
parser.add_argument('-verbose', action='store_true', help="Print many messages on screen.")
parser.add_argument('-plot', action='store_true', help="Plot loss and accuracy during training once the run is over.")
parser.add_argument('-totEpochs', type=int, default=5, help="Total number of epochs for the training")
parser.add_argument('-xtype', choices=['image2D'], default='image2D', help="Whether the model accepts 2D images (e.g. convnet) or flattened data (e.g. MLP)")
parser.add_argument('-opt', choices=['sgd','adam'], default='sgd', help="Choice of the minimization algorithm (sgd,adam)")
parser.add_argument('-bs', type=int, default=16, help="Batch size")
parser.add_argument('-lr', type=float, default=0.00005, help="Learning Rate")
parser.add_argument('-height', type=int, default=128, help="Image height")
parser.add_argument('-width', type=int, default=128, help="Image width")
parser.add_argument('-depth', type=int, default=3, help="Number of channels")
parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
parser.add_argument('-aug', action='store_true', help="Perform data augmentation.")
parser.add_argument('-resize', choices=['keep_proportions','acazzo'], default='acazzo', help='The way images are resized')
args=parser.parse_args()

if args.width!=args.height:
	raise NotImplementedError('Height and width of the image must be the same for the moment.')

if args.verbose:
	ngpu=len(keras.backend.tensorflow_backend._get_available_gpus())
	print('We have {} GPUs'.format(ngpu))
np.random.seed(12345)

# Create a unique output directory
now = datetime.datetime.now()
dt_string = now.strftime("%Y-%m-%d_%Hh%Mm%Ss")
outDir = args.outpath+'/smallvgg/'+dt_string+'/'
pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)
fsummary=open(outDir+'args.txt','w')
print(args, file=fsummary); fsummary.flush()


########
# DATA #
########
data,labels = [], np.array([])
classes = {'name': [ name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name)) ]}
classes['num']    = len(classes['name'])
classes['num_ex'] =  np.zeros(classes['num'], dtype=int)
for ic in range(classes['num']):
	c=classes['name'][ic]
	classPath=args.datapath+c+'/'
	if args.verbose: print('class:',c)
	classImages = os.listdir(classPath)
	classes['num_ex'][ic] = len(classImages) # number of examples per class
	for imageName in classImages:
		imagePath = classPath+imageName
		image = Image.open(imagePath)

		if args.resize == 'acazzo':
			image = image.resize((args.width,args.height))
		else:
			# Set image's largest dimension to target size, and fill the rest with black pixels
			image,rescaled = ResizeWithProportions(image, args.width) # width and height are assumed to be the same (assertion at the beginning)

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
(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=args.testSplit, random_state=42)
train_size=len(trainX)
test_size=len(testX)
if args.verbose:
	print('We expect the training   examples ({}) to be {}'.format(train_size, train_images))
	print('We expect the validation examples ({}) to be {}'.format(test_size , test_images))


# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
if args.aug:
	# aug = ImageDataGenerator(
	# 	rotation_range=180,      width_shift_range=0.1,
	# 	height_shift_range=0.1, shear_range=0.2, 
	# 	zoom_range=0.2, 		horizontal_flip=True, 
	# 	brightness_range=(0.8,1.2),
	# 	fill_mode="constant", cval=0)
	aug = ImageDataGenerator(
		rotation_range=180,      width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2,
		zoom_range=0.2, 		horizontal_flip=True
		)


# initialize our VGG-like Convolutional Neural Network
model = SmallVGGNet.build(width=args.width, height=args.height, depth=args.depth, classes=len(lb.classes_))


# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
if args.opt=='adam':
	opt = keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
elif args.opt=='sgd':
	opt = keras.optimizers.SGD(lr=args.lr, nesterov=True)
else:
	raise NotImplementedError('Optimizer {} is not implemented'.format(arg.opt))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# checkpoints
filepath = outDir+'/weights_epoch{epoch:03d}.hdf5' # make sure that the callback filepath exists, since it won't create directories
checkpointer    = keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy
coitointerrotto = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', patience=args.totEpochs, restore_best_weights=True)
callbacks=[checkpointer,coitointerrotto]

### TRAIN ###

# train the neural network
start=time.time()
if args.aug:
	history = model.fit_generator(
		aug.flow(trainX, trainY, batch_size=args.bs), 
		validation_data=(testX, testY), 
		steps_per_epoch=len(trainX)//args.bs,	
		epochs=args.totEpochs, 
		callbacks=callbacks)
else:
	history = model.fit(
		trainX, trainY, batch_size=args.bs, 
		validation_data=(testX, testY), 
		epochs=args.totEpochs, 
		callbacks=[checkpointer, coitointerrotto])
trainingTime=time.time()-start
print('Training took',trainingTime/60,'minutes')


### evaluate the network
print("[INFO] evaluating network...")
if args.aug:
	predictions = model.predict(testX)
else:
	predictions = model.predict(testX, batch_size=args.bs)
clrep=classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classes['name'])
print(clrep)



# Identify the easiest prediction and the worse mistake
i_maxconf_right=-1; i_maxconf_wrong=-1
maxconf_right  = 0; maxconf_wrong  = 0
for i in range(test_size):
	# Spot easiestly classified image (largest confidence, and correct classification)
	if testY.argmax(axis=1)[i]==predictions.argmax(axis=1)[i]: # correct classification
		if predictions[i][predictions[i].argmax()]>maxconf_right: # if the confidence on this prediction is larger than the largest seen until now
			i_maxconf_right = i
			maxconf_right   = predictions[i][predictions[i].argmax()]
	# Spot biggest mistake (largest confidence, and incorrect classification)
	else: # wrong classification
		if predictions[i][predictions[i].argmax()]>maxconf_wrong:
			i_maxconf_wrong=i
			maxconf_wrong=predictions[i][predictions[i].argmax()]


# Confidences of right and wrong predictions
confidences = predictions.max(axis=1) # confidence of each prediction made by the classifier
whether = np.array([1 if testY.argmax(axis=1)[i]==predictions.argmax(axis=1)[i] else 0 for i in range(len(predictions))]) #0 if wrong, 1 if right
confidences_right = confidences[np.where(testY.argmax(axis=1)==predictions.argmax(axis=1))[0]]
confidences_wrong = confidences[np.where(testY.argmax(axis=1)!=predictions.argmax(axis=1))[0]]


# Abstention accuracy
thresholds = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.98,0.99,0.995,0.997,0.999,0.9995,0.9999,0.99995,0.99999], dtype=np.float)
accs,nconfident = np.ndarray(len(thresholds),dtype=np.float), np.ndarray(len(thresholds), dtype=np.int)
for i,thres in enumerate(thresholds):
	confident     = np.where(confidences>thres)[0]
	nconfident[i] = len(confident)
	accs      [i] = whether[confident].sum()/nconfident[i] if nconfident[i]>0 else np.nan


##########
# OUTPUT #
##########

### TEXT ###

# Add training time to summary
print("Training-time: {:} seconds".format(trainingTime), file=fsummary)
fsummary.close()

# Save classification report
print(clrep, file=open(outDir+'/classification_report.txt','w'))

# Table with abstention data
print('threshold accuracy nconfident', file=open(outDir+'/abstention.txt','w'))
fabst=open(outDir+'/abstention.txt','a')
for i in range(len(thresholds)):
	print('{}\t{}\t{}'.format(thresholds[i],accs[i],nconfident[i]), file=fabst)
fabst.close()

### IMAGES ###

# Image of the easiest prediction
plt.figure(0)
npimage=testX[i_maxconf_right].reshape((args.width,args.height,args.depth))
npimage=np.rint(npimage*256).astype(np.uint8)
image=Image.fromarray(npimage)
plt.title('Prediction: {}, Truth: {}\nConfidence:{:.2f}'.format(classes['name'][ predictions[i_maxconf_right].argmax() ], 
																classes['name'][ testY      [i_maxconf_right].argmax() ],
																confidences[i_maxconf_right]) )
plt.imshow(image)
plt.savefig(outDir+'/easiest-prediction.png')

# Image of the worse prediction (i.e. the classifier was really sure about the prediction but it was wrong)
plt.figure(1)
npimage=testX[i_maxconf_wrong].reshape((args.width,args.height,args.depth))
npimage=np.rint(npimage*256).astype(np.uint8)
image=Image.fromarray(npimage)
plt.title('Prediction: {}, Truth: {}\nConfidence:{:.2f}'.format(classes['name'][ predictions[i_maxconf_wrong].argmax() ], 
																classes['name'][ testY      [i_maxconf_wrong].argmax() ],
																confidences[i_maxconf_wrong]) )
plt.imshow(image)
plt.savefig(outDir+'/worse-prediction.png')


# Plot loss during training
plt.figure(2)
plt.title('Model loss during training')
simulated_epochs=len(history.history['loss']) #If we did early stopping it is less than args.totEpochs
plt.plot(np.arange(1,simulated_epochs+1),history.history['loss'], label='train')
plt.plot(np.arange(1,simulated_epochs+1),history.history['val_loss'], label='test')
plt.xlabel('epoch')
plt.xlabel('loss')
plt.legend()
plt.savefig(outDir+'/loss.png')

# Plot accuracy during training
plt.figure(3)
plt.title('Model accuracy during training')
plt.ylim((0,1))
plt.plot(np.arange(1,simulated_epochs+1),history.history['accuracy'], label='train')
plt.plot(np.arange(1,simulated_epochs+1),history.history['val_accuracy'], label='test')
plt.plot(np.arange(1,simulated_epochs+1),np.ones(simulated_epochs)/classes['num'], label='random', color='black', linestyle='-.')
plt.xlabel('epoch')
plt.xlabel('loss')
plt.grid(axis='y')
plt.legend()
plt.savefig(outDir+'/accuracy.png')

# Scatter plot and density of correct and incorrect predictions (useful for active and semi-supervised learning)
plt.figure(4)
plt.title('Correct/incorrect predictions and their confidence')
sns.distplot(confidences_right, bins=20, label='Density of correct predictions', color='green')
sns.distplot(confidences_wrong, bins=20, label='Density of wrong   predictions', color='red')
plt.plot(confidences, whether, 'o', label='data (correct:1, wrong:0)', color='black', markersize=1)
plt.xlabel('confidence')
plt.xlim((0,1))
plt.ylim(bottom=-0.2)
plt.legend()
plt.savefig(outDir+'/confidence.png')


# Plot Abstention
plt.figure(5)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=.7)
ax1=plt.subplot(2, 1, 1)
ax1.set_ylim((0,1))
plt.title('Abstention')
plt.ylabel('Accuracy after abstention')
plt.xlabel('Threshold')
plt.plot(thresholds, accs, color='darkred')
plt.grid(axis='y')
ax2=plt.subplot(2, 1, 2)
ax2.set_ylim((0.1,test_size*1.5))
ax2.set_yscale('log')
plt.ylabel('Remaining data after abstention')
plt.xlabel('Threshold')
plt.plot(thresholds, nconfident, color='darkred')
plt.grid(axis='y')
plt.savefig(outDir+'/abstention.png')

if args.plot:
	plt.show()



