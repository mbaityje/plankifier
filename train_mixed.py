#!/usr/bin/env python3
# 
# This program trains a model on a combination of images and other features.
# It is not optimized and it is not meant to be used. It is instead a baseline
# program showing how to implement this kind of models. 
# The reason for this is that at the moment I do not have access to the feature
# data, so I could only write a program that will speed up my work once I will have it.
#
# In practice, the only extra information this program uses with respect to a normal
# MLP on the images is whether the image has been rescaled or not. Already this is
# enough to reach better accuracies on a model whose hyperparameters were chosen without
# thinking. The teaching is that the rescaling has a negative effect on the predictability.
#
# Currently, the same MLP is used to deal with the images and with the features.
# The outputs from the two kinds of data are then combined and fed into another MLP.
#
# Another issue is data augmentation. For data augmentation one option is to create
# a custom data augmentation generator 
# (such as https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/)
# 
# 
# Optimizer: SGD with no weight decay and Nesterov momentum.
# 
# Launch as:
# 	python train_mixed.py -totEpochs=100 -width=128 -height=128 -model=mlp -resize=keep_proportions -bs=16 -lr=0.0001 -opt=sgd -datapath='./data/zooplankton_trainingset_15oct/' -plot
# 
#########################################################################

import os, sys, pathlib, time, datetime, argparse, numpy as np, pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, concatenate
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_models, helper_data
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
parser.add_argument('-opt', choices=['sgd','adam'], default='sgd', help="Choice of the minimization algorithm (sgd,adam)")
parser.add_argument('-bs', type=int, default=32, help="Batch size")
parser.add_argument('-lr', type=float, default=0.00005, help="Learning Rate")
parser.add_argument('-height', type=int, default=128, help="Image height")
parser.add_argument('-width', type=int, default=128, help="Image width")
parser.add_argument('-depth', type=int, default=3, help="Number of channels")
parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
parser.add_argument('-aug', action='store_true', help="Perform data augmentation.")
parser.add_argument('-resize', choices=['keep_proportions','acazzo'], default='keep_proportions', help='The way images are resized')
parser.add_argument('-model', choices=['mlp','conv2','smallvgg'], default='mlp', help='The model. MLP gives decent results, conv2 is the best, smallvgg overfits (*validation* accuracy oscillates).')
parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers for MLP")
parser.add_argument('-load', default=None, help='Path to a previously trained model that should be loaded.')
parser.add_argument('-override_lr', action='store_true', help='If true, when loading a previously trained model it discards its LR in favor of args.lr')
parser.add_argument('-initial_epoch', type=int, default=0, help='Initial epoch of the training')
args=parser.parse_args()

print('\nRunning',sys.argv[0],sys.argv[1:])

# Check command line arguments
if args.width!=args.height:
	raise NotImplementedError('Height and width of the image must be the same for the moment.')
if args.aug==True and args.model in ['mlp']:
	print('We don\'t do data augmentation with the MLP')
	args.aug=False
if args.model != 'mlp':
	raise NotImplementedError('In this version we only implemented MLP')
flatten_image = True if args.model in ['mlp'] else False
if args.initial_epoch>=args.totEpochs:
	print('The initial epoch is already equalr or larger than the target number of epochs, so there is no need to do anything. Exiting...')
	raise SystemExit

if args.verbose:
	ngpu=len(keras.backend.tensorflow_backend._get_available_gpus())
	print('We have {} GPUs'.format(ngpu))
np.random.seed(12345)




# Create a unique output directory
now = datetime.datetime.now()
dt_string = now.strftime("%Y-%m-%d_%Hh%Mm%Ss")
outDir = args.outpath+'/'+args.model+'_mix/'+dt_string+'/'
pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)
fsummary=open(outDir+'args.txt','w')
print(args, file=fsummary); fsummary.flush()


########
# DATA #
########
def data_loader(args, seed=None):
	if not seed==None: np.random.seed(seed)

	# images 
	df = pd.DataFrame(columns=['name', 'npimage', 'rescaled', 'label'])

	# Basic stuff on classes
	classes = {'name': [ name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name)) ]}
	classes['num']    = len(classes['name'])
	classes['num_ex'] =  np.zeros(classes['num'], dtype=int)

	# Loop for data loading
	for ic in range(classes['num']):
		c=classes['name'][ic]
		classPath=args.datapath+c+'/'
		if args.verbose: print('class:',c)
		classImages = os.listdir(classPath)
		classes['num_ex'][ic] = len(classImages) # number of examples per class
		print('ic:',ic)
		sys.stdout.flush()
		for imageName in classImages:
			imagePath = classPath+imageName
			image = Image.open(imagePath)
			if args.resize == 'acazzo':
				image = image.resize((args.width,args.height))
				rescaled=1
			elif args.resize=='keep_proportions':
				# Set image's largest dimension to target size, and fill the rest with black pixels
				image,rescaled = helper_data.ResizeWithProportions(image, args.width) # width and height are assumed to be the same (assertion at the beginning)
			else:
				raise NotImplementedError('Unknown resize option in command line arguments: {}'.format(args.resize))
			npimage = np.array(image.copy() , dtype=np.float32)
			if flatten_image: 
				npimage = npimage.flatten()
			df=df.append({'name':imagePath, 'npimage':npimage, 'rescaled':rescaled, 'label':ic}, ignore_index=True)
			image.close()
	classes['tot_ex'] =  classes['num_ex'].sum()
	df.npimage = df.npimage / 255.0 # scale the raw pixel intensities to the range [0, 1]
	np.save(outDir+'classes.npy', classes)

	return df, classes

df, classes=data_loader(args)


#Split train and test
(trainX, testX, trainY, testY) = train_test_split(df.drop(['label'], axis=1, inplace=False) ,	df.label, test_size=args.testSplit, random_state=42)
train_size=len(trainX)
test_size=len(testX)

# Images train and test set
trainImage = trainX.npimage.values.tolist()
testImage  = testX.npimage.values.tolist()

# Features train and test set
trainFeat  = trainX.drop(['npimage','name'], axis=1, inplace=False).values.tolist()
testFeat   = testX.drop(['npimage','name'], axis=1, inplace=False).values.tolist()


# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY.values.tolist())
testY = lb.transform(testY.values.tolist())


if args.aug:
	raise NotImplementedError('Data Augmentation not implemented yet for mixed data.')

# initialize our VGG-like Convolutional Neural Network
if args.load!=None:
	raise NotImplementedError('Model loading not implemented yet for mixed data.')



# MODEL DEFINITION
def create_mlp(dim):
	# define our MLP network
	model = Sequential()
	model.add(Dense(32, input_dim=dim, activation="relu"))
	model.add(Dense(16, activation="relu"))
	return model

model_feat = create_mlp(len(trainFeat[0])) # Model for the features
model_imag = create_mlp(len(trainImage[0])) # Model for the images
combinedInput = concatenate([model_feat.output, model_imag.output]) # Combine the two
# our final FC layer head will have two dense layers
x = Dense(64, activation="relu")(combinedInput)
x = Dense(classes['num'], activation="softmax")(x)
 
# our final model will accept categorical/numerical data on model_feat
# input and images on the model_imag input, outputting a single value
model = Model(inputs=[model_feat.input, model_imag.input], outputs=x)


# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
if args.opt=='adam':
	opt = keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
elif args.opt=='sgd':
	opt = keras.optimizers.SGD(lr=args.lr, nesterov=True)
else:
	raise NotImplementedError('Optimizer {} is not implemented'.format(arg.opt))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



# checkpoints
checkpointer    = keras.callbacks.ModelCheckpoint(filepath=outDir+'/bestweights.hdf5', monitor='val_loss', verbose=0, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy
# coitointerrotto = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', patience=args.totEpochs, restore_best_weights=True)
logger          = keras.callbacks.callbacks.CSVLogger(outDir+'epochs.log', separator=' ', append=False)
callbacks=[checkpointer, logger]

### TRAIN ###

# train the neural network
start=time.time()
if args.aug:
	raise NotImplementedError('Data Aumentation not implemented')
else:
	history = model.fit(
		[trainFeat,trainImage], trainY, batch_size=args.bs, 
		validation_data=([testFeat,testImage], testY), 
		epochs=args.totEpochs, 
		callbacks=callbacks,
		initial_epoch = args.initial_epoch)
trainingTime=time.time()-start
print('Training took',trainingTime/60,'minutes')


### evaluate the network
print("[INFO] evaluating network...")
if args.aug:
	raise NotImplementedError('Data Augmentation not implemented')
else:
	predictions = model.predict([testFeat,testImage], batch_size=args.bs)
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
accs,nconfident = np.ndarray(len(thresholds), dtype=np.float), np.ndarray(len(thresholds), dtype=np.int)
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
with open(outDir+'/classification_report.txt','w') as frep:
	print(clrep, file=frep)
	# For each class, write down what it was confused with
	print('\nLet us see with which other taxa each class gets confused.', file=frep)
	for ic,c in enumerate(classes['name']):
		print("{:18}: ".format( classes['name'][ic]), end=' ', file=frep)
		ic_examples = np.where(testY.argmax(axis=1)==ic)[0] # examples in the test set with label ic
		ic_predictions = predictions[ic_examples].argmax(axis=1)
		histo = np.histogram(ic_predictions, bins=np.arange(classes['num']+1))[0]/len(ic_examples)
		ranks = np.argsort(histo)[::-1]
		# ic_classes = [classes['name'][ranks[i]] for i in range(classes['num'])]
		for m in range(5): # Print only first few mistaken classes
			print("{:18}({:.2f})".format( classes['name'][ranks[m]],histo[ranks[m]]), end=', ', file=frep)
		print('...', file=frep)

# Table with abstention data
print('threshold accuracy nconfident', file=open(outDir+'/abstention.txt','w'))
fabst=open(outDir+'/abstention.txt','a')
for i in range(len(thresholds)):
	print('{}\t{}\t{}'.format(thresholds[i],accs[i],nconfident[i]), file=fabst)
fabst.close()


### IMAGES ###

def plot_npimage(npimage, ifig=0, width=64, height=64, depth=3, title='Yet another image', filename=None):
	plt.figure(ifig)
	npimage=npimage.reshape((args.width,args.height,args.depth))	
	npimage=np.rint(npimage*256).astype(np.uint8)
	image=Image.fromarray(npimage)
	plt.title(title)
	plt.imshow(image)
	if filename!=None:
		plt.savefig(filename)

# Image of the easiest prediction
plot_npimage(testImage[i_maxconf_right], 0, args.width, args.height, args.depth, 
	title='Prediction: {}, Truth: {}\nConfidence:{:.2f}'.format(classes['name'][ predictions[i_maxconf_right].argmax() ], 
																classes['name'][ testY      [i_maxconf_right].argmax() ],
																confidences[i_maxconf_right]),
	filename=outDir+'/easiest-prediction.png')

# Image of the worst prediction
plot_npimage(testImage[i_maxconf_wrong], 1, args.width, args.height, args.depth, 
	title='Prediction: {}, Truth: {}\nConfidence:{:.2f}'.format(classes['name'][ predictions[i_maxconf_wrong].argmax() ], 
																classes['name'][ testY      [i_maxconf_wrong].argmax() ],
																confidences[i_maxconf_wrong]),
	filename=outDir+'/worst-prediction.png')


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



