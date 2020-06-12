# Imports
'''
Quick and dirty program that loads a model and generates prediction on custom data.

Next upgrades:
	[DONE] Use PR for abstention 
	[DONE] Validation Recall
	[DONE] Validation False Positives
	[] Use PR for ensembling -  FIRST MAKE A CONFIDENCE VS PR SCATTER PLOT

Launch as:
	python predict.py -modelfullname='./out/trained-models/conv2_image_adam_aug_b8_lr1e-3_L128_t500/keras_model.h5'


The program can also handle multiple models, and apply some ensemble rule. for example:
	python predict.py -testdir 'data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/' -modelfullname './out/trained-models/conv2_image_adam_aug_b8_lr1e-3_L128_t500/keras_model.h5' './out/trained-models/conv2_image_sgd_aug_b32_lr5e-5_L128_t1000/keras_model.h5'


run predict.py -ensMethods 'leader' -testdirs 'data/1_zooplankton_0p5x/validation/tommy_validation/images/conochilus/' 'data/1_zooplankton_0p5x/validation/tommy_validatio
    ...: n/images/chaoborus/' 'data/1_zooplankton_0p5x/validation/tommy_validation/images/bosmina/' 'data/1_zooplankton_0p5x/validation/tommy_validation/images/asplanchna/' -thres
    ...: holds 0.9 -labels 'conochilus' 'chaoborus' 'bosmina' 'asplanchna'

'''


import os, keras, argparse, re, glob, pathlib, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_data as hd, helper_models as hm
import train as t
# from PIL import Image
import tensorflow as tf
from collections import Counter


__UNCLASSIFIED__ = 'Unclassified'

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


class Cpred():
	''' 
	This class loads a model and makes predictions.

	Class initialization does everything. To obtain the predictions call the Predict() method

	'''
	def __init__(self, modelname='keras_model.h5', modelpath=None, weightsname='bestweights.hdf5',verbose=False):
		
		# Set class variables
		self.modelname   = modelname
		self.modelpath   = modelpath
		self.weightsname = weightsname
		self.verbose	 = verbose

		# Read parameters of the model that is loaded
		if modelpath is None: raise ValueError('CPred: modelpath cannot be None.')
		self.params, self.classes = hd.ReadArgsTxt(self.modelpath)

		self.LoadModel()

		return
		

	def Predict(self,npimages):
		return self.simPred.model.predict(npimages)

	def PredictionBundle(self, npimages):
		''' Calculates a bunch of quantities related to the predictions '''

		nimages=len(npimages)
		self.probs = self.Predict(npimages)

		# Predictions
		self.predictions=self.probs.argmax(axis=1)  # The class that the classifier would bet on
		self.confidences=self.probs.max(axis=1)     # equivalent to: [probs[i][predictions[i]] for i in range(len(probs))] 
		self.predictions_names=np.array([self.classes[self.predictions[i]] for i in range(nimages)],dtype=object)

		# Classifier's second choice
		self.predictions2=self.probs.argsort(axis=1)[:,-2] # The second most likely class
		self.confidences2=np.array([self.probs[i][self.predictions2[i]] for i in range(nimages)], dtype=float) 
		self.predictions2_names=np.array([self.classes[self.predictions2[i]] for i in range(nimages)],dtype=object)

		# Participation Ratio
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


	# def Output(self, outpath, fullname=True, predname='predict.txt', PRfilter=None, stdout=True):
	# 	''' Output with a standard format '''
	# 	# Create output directory
	# 	pathlib.Path(args.outpath).mkdir(parents=True, exist_ok=True)

	# 	# Choose whether to display full filenames or not
	# 	if not fullname:
	# 		# self.im_names=np.array([os.path.basename(im) for im in self.im_names],dtype=object)	
	# 		self.im_names=np.array(list( map(os.path.basename, self.im_names) ), dtype=object)

	# 	#
	# 	# Output to file
	# 	#
	# 	header='Name Prediction Confidence Prediction(2ndGuess) Confidence(2ndGuess) participation-ratio'
	# 	np.savetxt(outpath+'/'+predname, np.c_[self.im_names, self.predictions_names, self.confidences, self.predictions2_names, self.confidences2, self.pr], fmt='%s %s %.3f %s %.3f %.3f', header=header)


	# 	#
	# 	# Output to screen
	# 	#
	# 	self.SetPRFilter(PRfilter)
	# 	if stdout:
	# 		print(header)
	# 		for i in range(len(self.npimages)):
				
	# 			if self.pr[i]<=self.PRfilter:
	# 				print('{}\t{:20s}\t{:.3f}\t{:20s}\t{:.3f}\t{:.3f}'.format(self.im_names[i], self.classes[self.predictions[i]], self.confidences[i], self.classes[self.predictions2[i]], self.confidences2[i], self.pr[i] ) )
	# 	return



class Cval:
	def __init__(self):
		print('__init__ Cval')
		pass

	def GetImageNames(self):

		all_labels = []
		all_names  = []
		for itd,td in enumerate(self.testdirs):
			if not os.path.isdir(td):
				print('You told me to read the images from a directory that does not exist:\n{}'.format(td))
				raise IOError

			im_names_here = np.array(glob.glob(td+'/*.jpeg'),dtype=object) 
			all_names.extend( im_names_here)

			if self.labels is not None:
				all_labels.extend([self.labels[itd] for i in range(len(im_names_here))])

		# Check that the paths contained images
		if len(all_names)<1:
			print('We found no .jpeg images')
			print('Folders that should contain the images:',self.testdirs)
			print('Content of the folders:',os.listdir(self.testdirs))
			raise IOError()
		else:
			if self.verbose:
				print('There are {} images in {}'.format(len(im_names), td))

		return np.array(all_names), (None if self.labels==None else np.array(all_labels))


class Censemble(Cval):
	''' Class for ensembling predictions from different models'''

	# def __init__(self, modelnames=None, weightsname='bestweights.hdf5', testdir=None, predname='predict', verbose=False, outpath='./predict/', em='unanimity', absthres=0):
	def __init__(self, modelnames=['./out/trained-models/conv2_image_adam_aug_b8_lr1e-3_L128_t500/keras_model.h5'], weightnames=['bestweights.hdf5'], testdirs=['data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/'], labels=None, verbose=False, ensMethod='unanimity', absthres=0, absmetric='proba'):

		self.modelnames		= modelnames		# List with the model names
		assert(len(self.modelnames)==len(set(self.modelnames))) # No repeated models!
		self.nmodels		= len(modelnames)
		self.weightnames 	= weightnames if (len(modelnames)==len(weightnames)) else np.full(len(modelnames),weightnames) # Usually, the file with the best weights of the run
		self.testdirs		= testdirs			# Directories with the data we want to label
		self.verbose		= verbose			# Whether to display lots of text
		self.ensMethod  	= ensMethod			# Ensembling method
		self.absthres		= absthres			# Abstention threshold
		self.labels 		= labels
		self.absmetric		= absmetric			# use confidence or PR for abstention

		# Initialize predictors
		self.predictors, self.classnames = self.InitPredictors()
		sizes = list(set([self.predictors[im].params['L']  for im in range(self.nmodels)]) )

		# Initialize data  to predict
		self.im_names, self.im_labels = super().GetImageNames()
		self.npimages={} # Models are tailored to specific image sizes
		for L in sizes:
			self.npimages[L] = hd.LoadImageList(self.im_names, L, show=False) # Load images


	def InitPredictors(self):
		''' Initialize the predictors from the files '''
	
		predictors = []

		for imn,mname in enumerate(self.modelnames):

			modelpath,modelname=os.path.split(mname)
			predictors.append(Cpred(modelname=modelname, modelpath=modelpath, weightsname=self.weightnames[imn]))

			# Check that all models use the same classes
			if 0==imn:
				classnames = predictors[imn].classes
			else:
				assert( np.all(classnames==predictors[imn].classes))

		return predictors, classnames

	def MakePredictions(self):

		for pred in self.predictors:
			npimagesL=self.npimages[ pred.params['L'] ]
			pred.PredictionBundle(npimagesL)		

	def Abstain(self, predictions, confidences, pr, thres=0, metric='prob'):
		''' Apply abstention according to softmax probability or to participation ratio '''

		if metric == 'prob':
			filtro = list( filter(lambda i: confidences[i]> thres, range(self.nmodels)))
		elif metric == 'pr':
			filtro = list( filter(lambda i: pr[i]> thres, range(nmodels)))

		predictions = predictions[filtro]
		confidences = confidences[filtro]
		pr 			= pr 		 [filtro]

		return predictions, confidences, pr

	def Unanimity(self, predictions):
		''' Applies unanimity rule to predictions '''
		if len(set(predictions))==1:
			return self.classnames[predictions[0]]
		else:
			return __UNCLASSIFIED__

	def Majority(self, predictions):

		if len(set(predictions))==0:
			return __UNCLASSIFIED__
		elif len(set(predictions))==1:
			return self.classnames[predictions[0]]
		else:
			counter=Counter(predictions)
			# amax = np.argmax( list(counter.values()) )
			amax  = np.argsort(list(counter.values()))[-1] # Index (in counter) of the highest value
			amax2 = np.argsort(list(counter.values()))[-2] # Index of second highest value

			if list(counter.values())[amax] == list(counter.values())[amax2]: # there is a tie
				return __UNCLASSIFIED__
			else:
				imaj = list(counter.keys())[amax]
				return self.classnames[imaj]

	def WeightedMajority(self, predictions, confidences, absthres):
		if len(set(predictions))>0:
			cum_conf = {} # For each iclass, stores the cumulative confidence
			for ic in predictions:
				cum_conf[ic] = 0
				for imod in range(self.nmodels):
					if predictions[imod]==ic:
						cum_conf[ic]+=confidences[imod]
			amax = np.argmax(list(cum_conf.values())) 
			imaj = list(cum_conf.keys())[amax]

			if list(cum_conf.values())[amax]>absthres:
				return self.classnames[imaj]
			else:
				return __UNCLASSIFIED__
		else:
			return __UNCLASSIFIED__

	def Leader(self, predictions, confidences):
		if len(set(predictions))>0:
			ileader = np.argmax(confidences)
			return self.classnames[predictions[ileader]]
		else:
			return __UNCLASSIFIED__


	def Ensemble(self, method, absthres):
		''' 
		Loop over the images
		For each image, generates an ensemble prediction
		'''

		if   method == None:   method = self.ensMethod
		if absthres == None: absthres = self.absthres

		self.guesses = []
		for iim,im_name in enumerate(self.im_names):
			predictions = np.array([self.predictors[imod].predictions[iim] for imod in range(self.nmodels)])
			confidences = np.array([self.predictors[imod].confidences[iim] for imod in range(self.nmodels)])
			pr 			= np.array([self.predictors[imod].pr		 [iim] for imod in range(self.nmodels)])

			# Abstention
			if absthres>0 and (method!='weighted-majority'):
				predictions, confidences, pr = self.Abstain(predictions, confidences, pr, thres=absthres, metric='prob')

			if self.verbose:
				print('\n',predictions, confidences, pr)

			## Different kinds of ensembling
			if method == 'unanimity': # Predict only if all models agree
				guess = (im_name, self.Unanimity(predictions) )

			elif method == 'leader': # Follow-the-leader: only give credit to the most confident prediction
				guess = (im_name, self.Leader(predictions, confidences) )

			elif method == 'majority': # Predict with most popular selection
				guess = (im_name, self.Majority(predictions))

			elif method == 'weighted-majority': # Weighted Majority (like majority, but each is weighted by their confidence)
				guess = (im_name, self.WeightedMajority(predictions, confidences, absthres))

			print("{} {}".format(guess[0], guess[1]) )
			self.guesses.append(guess)
		self.guesses=np.array(self.guesses)


	def WriteGuesses(self, filename):
		'''
		Create output directory and save predictions
			filename includes the path
		'''

		outpath,outname=os.path.split(filename)
		pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
		np.savetxt(filename, self.guesses, fmt='%s %s')
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
	parser.add_argument('-weightnames', nargs='+', default=['bestweights.hdf5'], help='Name of alternative weights for the model.')
	parser.add_argument('-testdirs', nargs='+', default=['data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/'], help='directory of the test data')
	parser.add_argument('-labels', nargs='+', default=None, help='If known, labels of the test data. One per directory.')
	parser.add_argument('-predname', default='./predict/predict.txt', help='name of the file with the model predictions')
	parser.add_argument('-verbose', action='store_true', help='Print lots of useless tensorflow information')
	parser.add_argument('-ensMethods', nargs='+', default=['unanimity'], help='Ensembling methods. Choose from: \'unanimity\',\'majority\', \'leader\', \'weighted-majority\'. Weighted Majority implements abstention in a different way (a good value is 1).')
	parser.add_argument('-thresholds', nargs='+', default=[0], type=float, help='Abstention thresholds on the confidence (a good value is 0.8, except for weighted-majority, where it should be >=1).')
	args=parser.parse_args()


	ensembler=Censemble(modelnames=args.modelfullname, 
						testdirs=args.testdirs, 
						weightnames=args.weightnames,
						labels=args.labels,
						verbose=args.verbose,
						)
	
	ensembler.MakePredictions()

	for method in args.ensMethods:
		for absthres in args.thresholds:
			print(method, absthres)
			ensembler.Ensemble(method=method, absthres=absthres)
			ensembler.WriteGuesses(args.predname)


'''
			## Validation
			nimages=len(ensembler.im_names)

			# Overall validation accuracy
			ncorrect=(ensembler.guesses[:,1]==ensembler.im_labels).astype(int).sum()
			print('Totall Recall:',ncorrect/nimages)
			
			# Per class validation
			for myclass in ensembler.labels:

				## Recall
				# indices of the images that are labeled with this class
				idLabels = list(filter(lambda i: ensembler.im_labels[i]==myclass, range(nimages) ))
				ncorrect=(ensembler.guesses[idLabels,1]==ensembler.im_labels[idLabels]).astype(float).sum()
				print('\n',myclass,'recall:',ncorrect/len(idLabels))

				## Positives
				# indices of the images that are guesses with this class
				idGuesses = list(filter(lambda i: ensembler.guesses[i,1]==myclass, range(nimages) ))
				truePositives = (ensembler.guesses[idGuesses,1]==ensembler.im_labels[idGuesses]).astype(float).sum()/len(idGuesses)
				print('truePositives:',truePositives)

				falsePositives = (ensembler.guesses[idGuesses,1]!=ensembler.im_labels[idGuesses]).astype(float).sum()/len(idGuesses)
				print('falsePositives:',falsePositives)
'''
