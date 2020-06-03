# Imports
'''
Quick and dirty program that loads a model and generates prediction on custom data.

Next upgrades:
	- Put all data in a single pandas dataframe

Launch as:
	python predict.py -modelpath='./out/conv2/2020-02-06_17h56m55s/' -weightsname='bestweights.hdf5' -testdir='../Q-AQUASCOPE/pictures/annotation_classifier/tommy_for_classifier/tommy_validation/images/' -target='daphnia'

'''


import os, keras, argparse, re, glob, pathlib, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_data as hd, helper_models as hm
# from PIL import Image
import tensorflow as tf


parser = argparse.ArgumentParser(description='Load a model and use it to make predictions on images')
parser.add_argument('-modelname', default='keras_model.h5', help='name of the model to be loaded')
parser.add_argument('-modelpath', default='./util-files/trained-conv2/', help='directory of the model to be loaded')
parser.add_argument('-weightsname', default='bestweights.hdf5', help='name of the model to be loaded. If None, choose latest created hdf5 file in the directory')
parser.add_argument('-testdir', default='data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/', help='directory of the test data')
parser.add_argument('-preddir', default='./', help='directory where you want the output to be')
parser.add_argument('-predname', default='predict.txt', help='name of the file with the model predictions')
parser.add_argument('-fullname', action='store_true', help='Output contains full image path instead of only the name')
parser.add_argument('-verbose', action='store_true', help='Print lots of useless tensorflow information')
parser.add_argument('-PRfilter', default=None, type=float, help='Give a threshold value, a>1. Screen output is filtered with the value of the participation ratio (PR) and only includes predictions with PR<a.')
parser.add_argument('-notxt', action='store_true', help='Avoid writing output to file (only have screen output)')
args=parser.parse_args()

if not args.verbose:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if not os.path.isdir(args.testdir):
	print('You told me to read the images from a directory that does not exist:\n{}'.format(args.testdir))
	raise IOError

# Read parameters of the model that is loaded
params, classes = hd.ReadArgsTxt(args.modelpath)



sys.exit('fine prova')


#If PR filter is unset, just set it to the maximum value that it can assume (i.e. the number of classes)
if args.PRfilter is None: 	args.PRfilter=len(classes)
else:
	if (args.PRfilter<1):
		print('args.PRfilter should be >=1, so we cannot accept ',args.PRfilter)
		raise ValueError


# Read names of images that we want to predict on.
im_names=np.array(glob.glob(args.testdir+'/*.jpeg'),dtype=object)
print('\nWe will predict the class of {} images'.format(len(im_names)))

# Check that the paths contained images
if len(im_names)<1:
	print('We found no .jpeg images')
	print('Folder which should contain the images:',args.testdir)
	print('Content of the folder:',os.listdir(args.testdir))
	raise IOError()
else:
	print('There are {} images in {}'.format(len(im_names), args.testdir))



# If no model name is given, select the newest one contained in the directory
if args.weightsname is None:
	from stat import S_ISREG, ST_CTIME, ST_MODE
	# Choose model. We load the latest created .hdf5 file, since later is better
	entries = [modelpath+'/'+entry for entry in os.listdir(modelpath) if '.hdf5' in entry]
	entries = ((os.stat(path), path) for path in entries)
	entries = ((stat[ST_CTIME], path) for stat, path in entries if S_ISREG(stat[ST_MODE]))
	modelfile=sorted(entries)[-1][1]
	print('Loading last generated model'.format(modelfile))
else:
	modelfile = args.modelpath+args.weightsname
	print('modelfile: {}'.format(modelfile))



# Load images
npimages=hd.LoadImageList(im_names, L, show=False)

# Initialize and load model
# model=keras.models.load_model(modelfile)
params = {
		'model': 'conv2', 
		'train': False, 
		'load': modelfile,
		'numclasses': len(classes),
		'lr': 0, # This number should not be relevant, since we don't train
		}
history, model = hm.PlainModel(npimages, None, None, None, params)



# Print prediction
probs=model.predict(npimages)
predictions=probs.argmax(axis=1)  # The class that the classifier would bet on
confidences=probs.max(axis=1)     # equivalent to: [probs[i][predictions[i]] for i in range(len(probs))] 
predictions_names=np.array([classes[predictions[i]] for i in range(len(npimages))],dtype=object)

predictions2=probs.argsort(axis=1)[:,-2] # The second most likely class
confidences2=np.array([probs[i][predictions2[i]] for i in range(len(probs))], dtype=float) 
predictions2_names=np.array([classes[predictions2[i]] for i in range(len(npimages))],dtype=object)


# Participation ratio
def PR(vec):
	'''
	if vec is:
	- fully localized:   PR=1
	- fully delocalized: PR=N
	'''
	num=np.square(np.sum(vec))
	den=np.square(vec).sum()
	return num/den

pr=np.array([PR(prob) for prob in probs], dtype=float)





##########
# OUTPUT #
##########
if not args.fullname:
	im_names=np.array([os.path.basename(im) for im in im_names],dtype=object)


header='Name Prediction Confidence Prediction(2ndGuess) Confidence(2ndGuess) participation-ratio'
if not args.notxt:
	np.savetxt(args.preddir+'/'+args.predname, np.c_[im_names, predictions_names, confidences, predictions2_names, confidences2, pr], fmt='%s %s %.3f %s %.3f %.3f', header=header)

print(header)
for i in range(len(npimages)):
	
	if pr[i]<=args.PRfilter:
		print('{}\t{:20s}\t{:.3f}\t{:20s}\t{:.3f}\t{:.3f}'.format(im_names[i], classes[predictions[i]], confidences[i], classes[predictions2[i]], confidences2[i], pr[i] ) )



