# Imports
'''
Quick and dirty program that loads a model and generates prediction on custom data.

Launch as:
	python predict.py  -modelfullnames './trained-models/conv2/keras_model.h5' \
                -weightnames './trained-models/conv2/bestweights.hdf5' \
                -testdirs 'data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/' \
                -thresholds 0.6 \
                -ensMethods 'unanimity' \
                -predname './out/predictions/predict'


The program can also handle multiple models, and apply some ensemble rule. for example:
	python predict.py -testdir 'data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/' -modelfullnames './out/trained-models/conv2_image_adam_aug_b8_lr1e-3_L128_t500/keras_model.h5' './out/trained-models/conv2_image_sgd_aug_b32_lr5e-5_L128_t1000/keras_model.h5'


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
from pathlib import Path

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


class Cpred:
	''' 
	This class loads a model and makes predictions.

	Class initialization does everything. To obtain the predictions call the Predict() method

	'''
	def __init__(self, modelname='./keras_model.h5', #modelpath='./', 
		weightsname='./bestweights.hdf5',verbose=False):
		
		# Set class variables
		self.modelname   = modelname
		self.modelpath   = os.path.dirname(self.modelname)
		self.weightsname = weightsname
		self.verbose	 = verbose

		# Read parameters of the model that is loaded
		# if modelpath is None: raise ValueError('CPred: modelpath cannot be None.')

		self.classes = np.load(self.modelpath+'/classes.npy')
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

		# Create a Ctrain object. Maybe it is not needed, and we could straight out load the model.
		self.simPred=t.Ctrain(verbose=False)
        
# 		try:
# 			self.simPred.params=np.load(self.modelpath+'/params.npy' , allow_pickle=True).item()
            
# 		except:
# 			print('WARNING: We were not able to load ',self.modelpath+'/params.npy')
# 		self.simPred.LoadModel(self.modelname, self.weightsname) # Loads models and updates weights


		try:
			self.simPred.params=np.load(self.modelpath+'/params.npy' , allow_pickle=True).item()
            
		except:
			print('WARNING: We were not able to load ',self.modelpath+'/params.npy')
			try:
				p = Path(self.modelpath).parents[4]
				self.simPred.params=np.load(str(p) +'/params.npy' , allow_pickle=True).item()    
				print('LOADED : ', str(p) +'/params.npy')
			except:
				print('WARNING: We were not able to load ', str(p) +'/params.npy')

		self.simPred.LoadModel(self.modelname, self.weightsname) # Loads models and updates weights

             




class Censemble:
	''' Class for ensembling predictions from different models'''

	def __init__(self, 
				modelnames=['./trained-models/conv2/keras_model.h5'], 
				weightnames=['./trained-models/conv2/bestweights.hdf5'], 
				testdirs=['data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/'], 
				labels=None, verbose=False, 
				ensMethod='unanimity', absthres=0, 
				absmetric='proba', screen=True, 
				training_data=False):

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
		self.screen			= screen

		# Initialize predictors
		self.predictors, self.classnames = self.InitPredictors()

		sizes = list(set([self.predictors[im].simPred.params.L  for im in range(self.nmodels)]) )
        
# 		resize_images = self.predictors[0].simPred.params.resize_images ## This needs to be written elegently later
# 		resize_image_list = list(set([self.predictors[im].simPred.params.resize_images  for im in range(self.nmodels)]) )


		# Initialize data  to predict
		self.im_names, self.im_labels = self.GetImageNames(training_data=training_data)
		self.npimages={} # Models are tailored to specific image sizes
        
# 		for L in sizes:
# # 			self.npimages[L] = hd.LoadImageList(self.im_names, L, show=False) # Load images
# 			self.npimages[L] = hd.LoadImageList(self.im_names, L,resize_images, show=False) # Load 

		counti=0
		for L in sizes:
			resize_image = self.predictors[counti].simPred.params.resize_images

# 			self.npimages[L] = hd.LoadImageList(self.im_names, L, show=False) # Load images
			self.npimages[L] = hd.LoadImageList(self.im_names, L, resize_image, show=False) # Load  
			counti=counti+1
    

	def InitPredictors(self):
		''' Initialize the predictors from the files '''
	
		predictors = []

		for imn,mname in enumerate(self.modelnames):

			# modelpath,modelname=os.path.split(mname)
			predictors.append(Cpred(modelname=mname, weightsname=self.weightnames[imn]))

			# Check that all models use the same classes
			if 0==imn:
				classnames = predictors[imn].classes
			else:
				assert np.all(classnames==predictors[imn].classes), "Class names for different models do not coincide"

		return predictors, classnames


	def MakePredictions(self):

		for pred in self.predictors:
			npimagesL=self.npimages[ pred.simPred.params.L ]
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


	def GetImageNames(self, training_data=False):
		'''
		training_data is an auxiliary variable introduced to solve inconsistencies in the data structure
		'''

		training_data_string='/training_data' if training_data else ''

		all_labels = []
		all_names  = []
		for itd,td in enumerate(self.testdirs):

			if not os.path.isdir(td):
				print('You told me to read the images from a directory that does not exist:\n{}'.format(td))
				raise IOError

			im_names_here = np.array(glob.glob(td+training_data_string+'/*.jpeg'),dtype=object) 
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

			if self.screen:
				print("{} {}".format(guess[0], guess[1]) )
			self.guesses.append(guess)
		
		self.guesses=np.array(self.guesses)
		
		return


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
	parser.add_argument('-modelfullnames', nargs='+', \
		default	= ['./trained-models/conv2/keras_model.h5'], \
		help 	= 'name of the model to be loaded, must include path')
	parser.add_argument('-weightnames', nargs='+', default=['./trained-models/conv2/bestweights.hdf5'], help='Name of alternative weights for the model. Must include path')
	parser.add_argument('-testdirs', nargs='+', default=['data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/'], help='directory of the test data')
	parser.add_argument('-predname', default='./out/predictions/predict', help='Name of the file with the model predictions (without extension)')
	parser.add_argument('-verbose', action='store_true', help='Print lots of useless tensorflow information')
	parser.add_argument('-ensMethods', nargs='+', default=['unanimity'], help='Ensembling methods. Choose from: \'unanimity\',\'majority\', \'leader\', \'weighted-majority\'. Weighted Majority implements abstention in a different way (a good value is 1).')
	parser.add_argument('-thresholds', nargs='+', default=[0], type=float, help='Abstention thresholds on the confidence (a good value is 0.8, except for weighted-majority, where it should be >=1).')
	parser.add_argument('-nosuffix', action='store_true', help='If activated, no suffix is added to the output file')
	args=parser.parse_args()


# 	ensembler=Censemble(modelnames=args.modelfullnames, 
# 						testdirs=args.testdirs, 
# 						weightnames=args.weightnames,
# 						verbose=args.verbose,
# 						)
	
    
	ensembler=Censemble(modelnames=args.modelfullnames, 
						testdirs=args.testdirs, 
						weightnames=args.weightnames,
						)
	
    
    
	ensembler.MakePredictions()

	for method in args.ensMethods:
		for absthres in args.thresholds:
			print('\nMethod:',method, '\tAbs-threshold:',absthres)
			ensembler.Ensemble(method=method, absthres=absthres)


			if args.nosuffix:
				ensembler.WriteGuesses('{}.txt'.format(args.predname))
			else:
				ensembler.WriteGuesses('{}_{}abs{}.txt'.format(args.predname,method,absthres))

