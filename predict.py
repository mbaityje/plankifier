# Imports
'''
Quick and dirty program that loads a model and generates prediction on custom data.

Next upgrades:
	- Put all data in a single pandas dataframe

Launch as:
	python predict.py -modelpath='./out/conv2/2020-02-06_17h56m55s/' -modelname='bestweights.hdf5' -testdir='../Q-AQUASCOPE/pictures/annotation_classifier/tommy_for_classifier/tommy_validation/images/' -target='daphnia'

'''


import os, keras, argparse, re, glob, pathlib, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_models, helper_data
from PIL import Image
import tensorflow as tf


parser = argparse.ArgumentParser(description='Load a model and use it to make predictions on images')
parser.add_argument('-modelpath', default='./util-files/trained-conv2/', help='directory of the model to be loaded')
parser.add_argument('-modelname', default='bestweights.hdf5', help='name of the model to be loaded. If None, choose latest created hdf5 file in the directory')
parser.add_argument('-testdir', default='	../Q-AQUASCOPE/pictures/annotation_classifier/tommy_for_classifier/tommy_validation/images/', help='directory of the test data')
parser.add_argument('-fullname', action='store_true', help='Output contains full image path instead of only the name')
parser.add_argument('-verbose', action='store_true', help='Print lots of useless tensorflow information')
args=parser.parse_args()

if not args.verbose:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if not os.path.isdir(args.testdir):
	print('You told me to read the images from a directory that does not exist:\n{}'.format(args.testdir))
	raise IOError


def CheckArgs(modelpath):
	'''
	Looks with what arguments the model was trained, and makes a series of consistency checks.
	'''
	argsname=modelpath+'args.txt'

	# Read Arguments
	with open(argsname,'r') as fargs:
		args=fargs.read()
		print('---------- Arguments for generation of the model ----------')
		print(args)
		print('-----------------------------------------------------------')
		layers=[None,None]
		for s in re.split('[\,,\),\(]',args):
		    if 'height' in s:
		        height=np.int64(re.search('(\d+)',s).group(1))
		    if 'width' in s:
		        width=np.int64(re.search('(\d+)',s).group(1))
		    if 'depth' in s:
		        depth=np.int64(re.search('(\d+)',s).group(1))
		    if 'model' in s:
		        modelname=re.search('=\'(.+)\'$',s).group(1)
		    if 'resize' in s:
		        resize=re.search('=\'(.+)\'$',s).group(1)
		    if 'layers' in s: #first layer
		        layers[0]=np.int64(re.search('=\[(.+)$',s).group(1))
		    if re.match('^ \d+',s): #second layer
		        layers[1]=np.int64(re.match('^ (\d+)',s).group(1))
		    if 'datapath' in s:
		        datapath=re.search('=\'(.+)\'$',s).group(1)
		    if 'outpath' in s:
		        outpath=re.search('=\'(.+)\'$',s).group(1)

	print('height: ', height)
	print('width: ', width)
	print('depth: ', depth)
	print('model name: ', modelname)
	print('resize: ', resize)
	print('layers: ', layers)
	print('datapath: ', datapath)
	print('outpath: ', outpath)
	print('-----------------------------------------------------------')

	# Extract classes from npy file in output directory
	classes_dict=np.load(modelpath+'classes.npy',allow_pickle=True).item()

	def OutputClassPaths():
		print('\nclasses_dict full filename: ',modelpath+'classes.npy')
		print('classes_dict[\'name\']:',classes_dict['name'])
		print('\nclasses are the subdirectories in the following folder: ', datapath)
		print('classes:',classes)
		print('')

	try:
		# Extract classes from data directory (if datapath is accessible)
		classes = [ name for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name)) ]
	except FileNotFoundError:
		pass
	else:
		# Compare the two
		if len(classes_dict['name']) != len(classes):
			OutputClassPaths()
			raise IndexError('Number of classes in data directory is not the same as in classes.npy')
		if not np.all(set(classes_dict['name'])==set(classes)):
			OutputClassPaths()
			raise ValueError('Some of the classes in data directory are not the same as in classes.npy')


	return height, width, depth, modelname, resize, layers, datapath, outpath, classes_dict

height, width, depth, modelname, resize, layers, datapath, outpath, classes_dict = CheckArgs(args.modelpath)


# Read names of images that we want to predict on.
im_names=glob.glob(args.testdir+'/*.jpeg')
print('\nWe will predict the class of {} images'.format(len(im_names)))

# Check that the paths contained images
if len(im_names)<1:
	print('We found no .jpeg images')
	print('Folder which should contain the images:',args.testdir)
	print('Content of the folder:',os.listdir(args.testdir))
	raise IOError()
else:
	print('There are {} images in {}'.format(len(im_names), args.testdir))

if args.modelname == None:
	from stat import S_ISREG, ST_CTIME, ST_MODE
	# Choose model. We load the latest created .hdf5 file, since later is better
	entries = [modelpath+'/'+entry for entry in os.listdir(modelpath) if '.hdf5' in entry]
	entries = ((os.stat(path), path) for path in entries)
	entries = ((stat[ST_CTIME], path) for stat, path in entries if S_ISREG(stat[ST_MODE]))
	modelfile=sorted(entries)[-1][1]
	print('Loading last generated model'.format(modelfile))
else:
	modelfile = args.modelpath+args.modelname
	print('Loading {}'.format(modelfile))

# Initialize and load model
model=keras.models.load_model(modelfile)

def load_images(im_names, width, height, depth, modelname, resize):
    ''' 
    Function that loads a list of images given in im_names, and returns 
    them in a numpy format that can be used by the classifier
    '''
    npimages=np.ndarray((len(im_names),width,height,depth)) if modelname != 'mlp' else np.ndarray((len(im_names),width*height*depth))

    for i,im_name in enumerate(im_names):
        image = Image.open(im_name)
        if resize == 'acazzo':
            image = image.resize((width,height))
        else:
            # Set image's largest dimension to target size, and fill the rest with black pixels
            if width!=height:
            	print('load_images(): width={}, height={}'.format(width,height))
            	raise NotImplementedError('We are only treating square, not rectangular images')
            image,rescaled = helper_data.ResizeWithProportions(image, width) # width and height are assumed to be the same
            npimage=np.array(image.copy())
        if model == 'mlp':
            npimage = npimage.flatten()
        npimages[i]=npimage
        if len(im_names)==1:
            image.show()
        image.close()    
    return npimages/255.0


# Load images
npimages=load_images(im_names, width, height, depth, modelname, resize)


# Print prediction
probs=model.predict(npimages)
predictions=probs.argmax(axis=1)  # The class that the classifier wuold bet on
confidences=probs.max(axis=1)     # equivalent to: [probs[i][predictions[i]] for i in range(len(probs))] 

predictions2=probs.argsort(axis=1)[:,-2] # The second most likely class
confidences2=[probs[i][predictions2[i]] for i in range(len(probs))] 


print('Name Prediction Confidence Prediction(2ndGuess) Confidence(2ndGuess)')
for i in range(len(npimages)):
	name=im_names[i] if args.fullname else os.path.basename(im_names[i])
	print('{}\t{:20s}\t{:.3f}\t{:20s}\t{:.3f}'.format(name, classes_dict['name'][predictions[i]], confidences[i], classes_dict['name'][predictions2[i]], confidences2[i]))

