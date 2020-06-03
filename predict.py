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
import train as t
# from PIL import Image
import tensorflow as tf

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
		self.simPred=t.Ctrain()


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
	def GetImageNames(testdir):
		''' Checks that given paths are good, and obtains the list of images contained in `testdir` '''

		# Check that testdir is a directory
		if not os.path.isdir(testdir):
			print('You told me to read the images from a directory that does not exist:\n{}'.format(testdir))
			raise IOError

		# Read names of images that we want to predict on.
		im_names=np.array(glob.glob(testdir+'/*.jpeg'),dtype=object)
		print('\nWe will predict the class of {} images'.format(len(im_names)))

		# Check that the paths contained images
		if len(im_names)<1:
			print('We found no .jpeg images')
			print('Folder which should contain the images:',testdir)
			print('Content of the folder:',os.listdir(testdir))
			raise IOError()
		else:
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
	parser.add_argument('-modelname', default='keras_model.h5', help='name of the model to be loaded')
	parser.add_argument('-modelpath', default='./util-files/trained-conv2/', help='directory of the model to be loaded')
	parser.add_argument('-weightsname', default='bestweights.hdf5', help='name of the model to be loaded. If None, choose latest created hdf5 file in the directory')
	parser.add_argument('-testdir', default='data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/', help='directory of the test data')
	parser.add_argument('-predname', default='predict.txt', help='name of the file with the model predictions')
	parser.add_argument('-fullname', action='store_true', help='Output contains full image path instead of only the name')
	parser.add_argument('-verbose', action='store_true', help='Print lots of useless tensorflow information')
	parser.add_argument('-PRfilter', default=None, type=float, help='Give a threshold value, a>1. Screen output is filtered with the value of the participation ratio (PR) and only includes predictions with PR<a.')
	parser.add_argument('-outpath', default='./prova_predict/', help='Output path')
	args=parser.parse_args()



	predictor = Cpred(modelname=args.modelname, modelpath=args.modelpath, weightsname=args.weightsname, testdir=args.testdir)
	predictor.PredictionBundle()
	predictor.Output(outpath=args.outpath, fullname=args.fullname, predname=args.predname, PRfilter=args.PRfilter, stdout=True)


