#!/usr/bin/env python3
# 
# Train on zooplankton data with a simple multi-layer perceptron model.
#
# Optimizer: SGD with no weight decay, no momentum.
# 
# Launch as:
# 	python train_features_mlp.py -lr=0.1 -totEpochs=100 -layers 256 128 -plot
#
# 
#########################################################################

import keras, os, pathlib, time, datetime, argparse, numpy as np, pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
parser.add_argument('-outpath', default='./out/', help="Print many messages on screen.")
parser.add_argument('-verbose', action='store_true', help="Print many messages on screen.")
parser.add_argument('-plot', action='store_true', help="Plot loss and accuracy during training once the run is over.")
parser.add_argument('-totEpochs', type=int, default=5, help="Total number of epochs for the training")
parser.add_argument('-bs', type=int, default=16, help="Batch size")
parser.add_argument('-lr', type=float, default=0.1, help="Learning Rate")
parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers")
parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
args=parser.parse_args()

if args.verbose:
	ngpu=len(keras.backend.tensorflow_backend._get_available_gpus())
	print('We have {} GPUs'.format(ngpu))
np.random.seed(12345)

# Create a unique output directory
now = datetime.datetime.now()
dt_string = now.strftime("%Y-%m-%d_%Hh%Mm%Ss")
outDir = args.outpath+'/MLP_feat/'+dt_string+'/'
pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)
fsummary=open(outDir+'args.txt','w')
print(args, file=fsummary); fsummary.flush()

########
# DATA #
########
filename='./data/feature-data-dummy/0P5X_euk.txt'
target='genus'
X=pd.read_csv(filename,sep='\t', low_memory=False)
X.dropna(axis=0,subset=[target], inplace=True) # Remove rows without target
# Separate data from labels
y = X[target]                          
X.drop([target], axis=1, inplace=True) 
# Remove empty columns
empty_cols=[col for col in X.columns if X[col].count()==0]
if len(empty_cols)>0:
    X.drop(empty_cols,axis=1,inplace=True)
# Remove other columns that shouldn't be used
# Columns that are good for the dataset
size_cols=['maj_axis_len','min_axis_len','area','aspect_ratio','eccentricity','estimated_volume','file_size','image_height','image_width','orientation','solidity',]
color_cols=sorted(X.loc[:, X.columns.str.contains('intensity')])
# Columns that should absolutely be removed
leakage_cols=['class','species','family','empire','kingdom','order','phylum']
# Other columns
time_cols=['acquisition_time']
useless_cols=sorted(X.loc[:, X.columns.str.contains('modif')])+['_id','extension','filename','group_id','tags','upload_id']
boh_cols=['multiple_species','partially_cropped'] # These columns are mostly nan, but I suspect that these nan should be False
# Check that we are not forgetting any columns
keep_cols=size_cols+color_cols
remove_cols=leakage_cols+useless_cols+time_cols+boh_cols
remaining_cols=[col for col in X.columns if col not in remove_cols+keep_cols]
if len(remaining_cols)!=0:
    print('There are still some columns that you didn\'t take into account!')
    print(remaining_cols)
    raise Exception
if len(remove_cols)>0:
	X.drop(remove_cols,axis=1,inplace=True)
else:
	raise Warning('It is very suspicious that remove_cols is empty')

# Columns with categorical values
categoric_cols = [col for col in X.columns if X[col].dtype=='object']
int_cols = [col for col in X.columns if X[col].dtype=='int']
float_cols = [col for col in X.columns if X[col].dtype=='float']
# Check that we accounted for all data types
if len(int_cols+float_cols+categoric_cols)!=len(X.columns):
    print('There are still some types that you didn\'t take into account!')
    print(set([X[col].dtype for col in X.columns]))
#In principle there are no relevant categoric columns. If there are, we throw a warning and remove it
if len(categoric_cols)>0: 
	X.drop(categoric_cols,axis=1,inplace=True)
	raise Warning('We found columns with categorical values, that we did not expect: {}. We remove them and keep going.'.format(categoric_cols))





#Split train and test
X=X.values # If I work with numpy arrays I can use verbatim the code of the other models
y=y.values
num_classes=len(X[0])
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=args.testSplit, random_state=42)
train_size=len(trainX)
test_size=len(testX)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


model = Sequential()
model.add(Dense(args.layers[0], input_shape=(len(trainX[0]),), activation="sigmoid"))
model.add(Dense(args.layers[1], activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))


# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = keras.optimizers.SGD(lr=args.lr, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the neural network
start=time.time()
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=args.totEpochs, batch_size=args.bs)
trainingTime=time.time()-start
print('Training took',trainingTime/60,'minutes')

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
clrep=classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
print(clrep)



# Identify the easiest prediction and the worse mistake
i_maxconf_right=-1; i_maxconf_wrong=-1
maxconf_right  = 0; maxconf_wrong  = 0
for i in range(test_size):
	# Spot easiestly classified example (largest confidence, and correct classification)
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
np.savetxt(outDir+'/abstention.txt',np.stack((thresholds,accs,nconfident),axis=1),header='threshold accuracy nconfident')

### IMAGES ###

# Plot loss during training
plt.figure(2)
plt.title('Model loss during training')
plt.plot(np.arange(1,args.totEpochs+1),history.history['loss'], label='train')
plt.plot(np.arange(1,args.totEpochs+1),history.history['val_loss'], label='test')
plt.xlabel('epoch')
plt.xlabel('loss')
plt.legend()
plt.savefig(outDir+'/loss.png')

# Plot accuracy during training
plt.figure(3)
plt.title('Model accuracy during training')
plt.ylim((0,1))
plt.plot(np.arange(1,args.totEpochs+1),history.history['accuracy'], label='train')
plt.plot(np.arange(1,args.totEpochs+1),history.history['val_accuracy'], label='test')
plt.plot(np.arange(1,args.totEpochs+1),np.ones(args.totEpochs)/num_classes, label='random', color='black', linestyle='-.')
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
ax1.set_xlim((0,1))
plt.title('Abstention')
plt.ylabel('Accuracy after abstention')
plt.xlabel('Threshold')
plt.plot(thresholds, accs, color='darkred')
plt.grid(axis='y')
ax2=plt.subplot(2, 1, 2)
ax2.set_ylim((0.1,test_size*1.5))
axx.set_xlim((0,1))
ax2.set_yscale('log')
plt.ylabel('Remaining data after abstention')
plt.xlabel('Threshold')
plt.plot(thresholds, nconfident, color='darkred')
plt.grid(axis='y')
plt.savefig(outDir+'/abstention.png')

if args.plot:
	plt.show()



