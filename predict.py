# Imports
'''
Quick and dirty program that loads a model and generates prediction on custom data.

Next upgrades:
	- Put all data in a single pandas dataframe

Launch as:
	python predict.py -modelfullname='./out/trained-models/conv2_image_adam_aug_b8_lr1e-3_L128_t500/keras_model.h5'


The program can also handle multiple models, and apply some ensemble rule. for example:
	python predict.py -testdir 'data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/' -modelfullname './out/trained-models/conv2_image_adam_aug_b8_lr1e-3_L128_t500/keras_model.h5' './out/trained-models/conv2_image_sgd_aug_b32_lr5e-5_L128_t1000/keras_model.h5'


'''


import os, keras, argparse, re, glob, pathlib, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_data as hd, helper_models as hm
import train as t
# from PIL import Image
import tensorflow as tf
from collections import Counter

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


class Cpred:
	''' 
	This class loads a model and makes predictions.

	Class initialization does everything. To obtain the predictions call the Predict() method

	'''
	def __init__(self, modelname='keras_model.h5', modelpath=None, weightsname='bestweights.hdf5', testdir='data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/'):
		
		# Set class variables
		self.modelname   = modelname
		self.modelpath   = modelpath
		self.weightsname = weightsname
		self.testdir	 = testdir

		# Get names of images in testdir
		self.im_names = self.GetImageNames(self.testdir)

		# Read parameters of the model that is loaded
		if modelpath is None: raise ValueError('CPred: modelpath cannot be None.')
		self.params, self.classes = hd.ReadArgsTxt(self.modelpath)

		# Load images
		self.npimages = hd.LoadImageList(self.im_names, self.params['L'], show=False)


		self.LoadModel()

		return
		

	def Predict(self):
		return self.simPred.model.predict(self.npimages)

	def PredictionBundle(self):
		''' Calculates a bunch of quantities related to the predictions '''

		nimages=len(self.npimages)
		self.probs = predictor.Predict()

		self.predictions=self.probs.argmax(axis=1)  # The class that the classifier would bet on
		self.confidences=self.probs.max(axis=1)     # equivalent to: [probs[i][predictions[i]] for i in range(len(probs))] 
		self.predictions_names=np.array([self.classes[self.predictions[i]] for i in range(nimages)],dtype=object)

		self.predictions2=self.probs.argsort(axis=1)[:,-2] # The second most likely class
		self.confidences2=np.array([self.probs[i][self.predictions2[i]] for i in range(nimages)], dtype=float) 
		self.predictions2_names=np.array([self.classes[self.predictions2[i]] for i in range(nimages)],dtype=object)


	# Participation Ratio
	# pr=np.array([PR(prob) for prob in probs], dtype=float)

		self.pr = np.array(list(map(PR, self.probs)), dtype=float)



		return

	def LoadModel(self):

		## Now Create the Model
		# Create the context
		self.simPred=t.Ctrain(verbose=False)

		# If saved model exists, we load the model and possibly load the bestweights
		mname=self.modelpath+'/'+self.modelname
		if os.path.exists(mname):

			# Load the model
			self.simPred.LoadModel(mname)

			# Load best weights
			if self.weightsname is not None: 
				if os.path.exists(self.modelpath+'/'+self.weightsname):
					self.simPred.model.load_weights(self.modelpath+'/'+self.weightsname)
				else:
					print('I was asked to load weights file {}, but it does not exist'.format(self.weightsname) )
					raise IOError

		# If saved model does not exist, we create a model using the read params, and load bestweights
		else:
			raise NotImplementedError('NOT IMPLEMENTED: If saved model does not exist, we create a model using the read params, and load bestweights. For the moment, we need the model to exist.')


	@staticmethod
	def GetImageNames(testdir, verbose=False):
		''' Checks that given paths are good, and obtains the list of images contained in `testdir` '''

		# Check that testdir is a directory
		if not os.path.isdir(testdir):
			print('You told me to read the images from a directory that does not exist:\n{}'.format(testdir))
			raise IOError

		# Read names of images that we want to predict on.
		im_names=np.array(glob.glob(testdir+'/*.jpeg'),dtype=object)
		if verbose:
			print('\nWe will predict the class of {} images'.format(len(im_names)))

		# Check that the paths contained images
		if len(im_names)<1:
			print('We found no .jpeg images')
			print('Folder which should contain the images:',testdir)
			print('Content of the folder:',os.listdir(testdir))
			raise IOError()
		else:
			if verbose:
				print('There are {} images in {}'.format(len(im_names), testdir))

		return im_names

	def SetPRFilter(self, fval=None):
		''' Set a Participation Ratio filter to fval, in case we want one, and check that it is sound '''

		if fval is None: 	
			self.PRfilter=len(self.classes)
		else:
			if (fval<1):
				print('SetPRfilter(): PR filter should be >=1, so we cannot accept ',fval)
				raise ValueError
			self.PRfilter=fval
		return

	def Output(self, outpath, fullname=True, predname='predict.txt', PRfilter=None, stdout=True):
		''' Output with a standard format '''
		# Create output directory
		pathlib.Path(args.outpath).mkdir(parents=True, exist_ok=True)

		# Choose whether to display full filenames or not
		if not fullname:
			# self.im_names=np.array([os.path.basename(im) for im in self.im_names],dtype=object)	
			self.im_names=np.array(list( map(os.path.basename, self.im_names) ), dtype=object)

		#
		# Output to file
		#
		header='Name Prediction Confidence Prediction(2ndGuess) Confidence(2ndGuess) participation-ratio'
		np.savetxt(outpath+'/'+predname, np.c_[self.im_names, self.predictions_names, self.confidences, self.predictions2_names, self.confidences2, self.pr], fmt='%s %s %.3f %s %.3f %.3f', header=header)


		#
		# Output to screen
		#
		self.SetPRFilter(PRfilter)
		if stdout:
			print(header)
			for i in range(len(self.npimages)):
				
				if self.pr[i]<=self.PRfilter:
					print('{}\t{:20s}\t{:.3f}\t{:20s}\t{:.3f}\t{:.3f}'.format(self.im_names[i], self.classes[self.predictions[i]], self.confidences[i], self.classes[self.predictions2[i]], self.confidences2[i], self.pr[i] ) )



		return


if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Load a model and use it to make predictions on images')
	parser.add_argument('-modelfullname', nargs='+', \
		default=[ \
				'./out/trained-models/conv2_image_adam_aug_b8_lr1e-3_L128_t500/keras_model.h5', \
				'./out/trained-models/conv2_image_sgd_aug_b32_lr5e-5_L128_t1000/keras_model.h5', \
				'./out/trained-models/smallvgg_image_sgd_aug_b8_lr5e-6_L192_t5000/keras_model.h5', 	\
				'./out/trained-models/smallvgg_image_adam_aug_b32_lr1e-3_L128_t5000/keras_model.h5'], \
				help='name of the model to be loaded, must include path')
	parser.add_argument('-weightsname', default='bestweights.hdf5', help='Name of alternative weights for the model.')
	parser.add_argument('-testdir', default='data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/', help='directory of the test data')
	parser.add_argument('-predname', default='predict', help='name of the file with the model predictions')
	parser.add_argument('-fullname', action='store_true', help='Output contains full image path instead of only the name')
	parser.add_argument('-verbose', action='store_true', help='Print lots of useless tensorflow information')
	parser.add_argument('-PRfilter', default=None, type=float, help='Give a threshold value, a>1. Screen output is filtered with the value of the participation ratio (PR) and only includes predictions with PR<a.')
	parser.add_argument('-outpath', default='./predict/', help='Output path')
	parser.add_argument('-em', default='unanimity', choices=['unanimity','majority','leader','weighted-majority'], help='Ensembling method. Weighted Majority implements abstention in a different way (a good value is 1).')
	parser.add_argument('-absthres', default=0, type=float, help='Abstention threshold on the confidence (a good value is 0.8, except for weighted-majority, where it can even be >1).')
	args=parser.parse_args()

	# Predictions for every model and image
	nmodels=len(args.modelfullname)
	predictors = []

	for mname in args.modelfullname:

		modelpath,modelname=os.path.split(mname)
		predictor = Cpred(modelname=modelname, modelpath=modelpath, weightsname=args.weightsname, testdir=args.testdir)
		predictor.PredictionBundle()
		predictors.append(predictor)

	classnames=predictor.classes


	#
	# Ensembling
	#
	guesses=[]
	for iim,im_name in enumerate(predictor.im_names):
		predictions = np.array([predictors[imod].predictions[iim] for imod in range(nmodels)])
		confidences = np.array([predictors[imod].confidences[iim] for imod in range(nmodels)])

		# print('\n',predictions, confidences)

		# Abstention
		if args.absthres>0 and (args.em!='weighted-majority'):
			predictions = predictions[list( filter(lambda i: confidences[i]> args.absthres, range(nmodels)))]
			confidences = confidences[list( filter(lambda i: confidences[i]> args.absthres, range(nmodels)))]

		## Different kinds of ensembling

		if args.em == 'unanimity': # Predict only if all models agree
			if len(set(predictions))==1:
				# Since there is unanimity, I can use the prediction of the last one
				guess=(im_name, classnames[predictor.predictions[iim]])
			else:
				guess=(im_name, 'Unclassified')

		elif args.em == 'majority': # Predict with most populat selection

			if len(set(predictions))>0:
				print(predictions)
				counter=Counter(predictions)
				amax = np.argmax( list(counter.values()) )
				imaj = list(counter.keys())[amax]
				guess=(im_name, classnames[imaj])
			else:
				guess=(im_name, 'Unclassified')

		elif args.em == 'leader': # Follow-the-leader: only give credit to the most confident prediction

			if len(set(predictions))>0:
				ileader = np.argmax(confidences)
				guess=(im_name, classnames[predictions[ileader]])
			else:
				guess=(im_name, 'Unclassified')

		elif args.em == 'weighted-majority': # Weighted Majority (like majority, but each is weighted by their confidence)
			if len(set(predictions))>0:
				cum_conf = {} # For each iclass, stores the cumulative confidence
				for ic in predictions:
					cum_conf[ic] = 0
					for imod in range(nmodels):
						if predictions[imod]==ic:
							cum_conf[ic]+=confidences[imod]
				amax = np.argmax(list(cum_conf.values())) 
				imaj = list(cum_conf.keys())[amax]

				if list(cum_conf.values())[amax]>args.absthres:
					guess=(im_name, classnames[imaj]) 
				else:
					guess=(im_name, 'Unclassified')
			else:
				guess=(im_name, 'Unclassified')

		print("{} {}".format(guess[0], guess[1]) )
		guesses.append(guess)


# Create output directory and save predictions
pathlib.Path(args.outpath).mkdir(parents=True, exist_ok=True)
np.savetxt(args.outpath+'/'+args.predname+'.txt', guesses, fmt='%s %s')

