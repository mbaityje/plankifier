#!/usr/bin/env python
#
# Helper functions for data manipulation
#
################################################
from PIL import Image
import os, glob, re
import numpy as np, pandas as pd

import sys

def ResizeWithProportions(im, desired_size):
	'''
	Take and image and resize it to a square of the desired size.
	0) If any dimension of the image is larger than the desired size, shrink until the image can fully fit in the desired size
	1) Add black paddings to create a square
	'''

	old_size    = im.size
	largest_dim = max(old_size)
	smallest_dim = min(old_size)

	# If the image dimensions are very different, reducing the larger one to `desired_size` can make the other
	# dimension too small. We impose that it be at least 4 pixels.
	if desired_size*smallest_dim/largest_dim<4:
		print('Image size: ({},{})'.format(largest_dim,smallest_dim ))
		print('Desired size: ({},{})'.format(desired_size,desired_size))
		raise ValueError('Images are too extreme rectangles to be reduced to this size. Try increasing the desired image size.')

	rescaled    = 0 # This flag tells us whether there was a rescaling of the image (besides the padding). We can use it as feature for training.

	# 0) If any dimension of the image is larger than the desired size, shrink until the image can fully fit in the desired size
	if max(im.size)>desired_size:

		ratio = float(desired_size)/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])
		# print('new_size:',new_size)
		sys.stdout.flush()
		im = im.resize(new_size, Image.LANCZOS)
		rescaled = 1

    # 1) Add black paddings to create a square
	new_im = Image.new("RGB", (desired_size, desired_size), color=0)
	new_im.paste(im, (	(desired_size-im.size[0])//2,
						(desired_size-im.size[1])//2))

	return new_im, rescaled

def ReduceClasses(datapath, class_select):
	print('datapath:',datapath)
	print('classes from datapath:',os.listdir(datapath))
	allClasses = [ name for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name)) ]
	if class_select is None:
		class_select = allClasses
	else:
		if not set(class_select).issubset(allClasses):
			print('Some of the classes input by the user are not present in the dataset.')
			print('class_select:',class_select)
			print('all  classes:',allClasses)
			raise ValueError
	return class_select

def LoadMixed(datapath, L, class_select=None, alsoImages=True):
	'''
	Uses the data in datapath to create a DataFrame with images and features. 
	For each class, we read a tsv file with the features. This file also contains the name of the corresponding image, which we fetch and resize.
	For each line in the tsv file, we then have all the features in the tsv, plus class name, image (as numpy array), and a binary variable stating whether the image was resized or not.
	Assumes a well-defined directory structure.

	Arguments:
	datapath 	 - the path where the data is stored. Inside datapath, we expect to find directories with the names of the classes
	L 			 - images are rescaled to a square of size LxL (maintaining proportions)
	class_select - a list of the classes to load. If None (default), loads all classes 
	alsoImages   - flag that tells whether to only load features, or features+images
	Output:
	df 			 - a dataframe with classname, npimage, rescaled, and all the columns in features.tsv
	'''
	df = pd.DataFrame()
	class_select=ReduceClasses(datapath, class_select)	# Decide whether to use all available classes

	# Loop for data loading
	for c in class_select: # Loop over the classes

		print(c)
		dfFeat = pd.read_csv(datapath+c+'/features.tsv', sep = '\t')


		classPath=datapath+c+'/training_data/'

		# Each line in features.tsv should be associated with an image (the url is slightly different than what appears in the file)
		for index, row in dfFeat.iterrows():

			if alsoImages:
				imName=datapath+c+'/training_data/'+os.path.basename(row['url'])
				image=Image.open(imName)
				image,rescaled = ResizeWithProportions(image, L) # Set image's largest dimension to target size, and fill the rest with black pixels
				npimage = np.array(image.copy() , dtype=np.float32)			 # Convert to numpy

				dftemp=pd.DataFrame([[c,npimage,rescaled]+row.to_list()] ,columns=['classname','npimage','rescaled']+dfFeat.columns.to_list())
				image.close()
			else: #alsoImages is False here
				dftemp=pd.DataFrame([[c]+row.to_list()] ,columns=['classname']+dfFeat.columns.to_list())

			df=pd.concat([df,dftemp], axis=0)

	if alsoImages:
		df.npimage = df.npimage / 255.0 # scale the raw pixel intensities to the range [0, 1]

	return df.reset_index(drop=True) # The data was loaded without an index, that we add with reset_index()

def LoadImage(filename, L=None, show=False):
	''' Loads one image '''

	image = Image.open(filename)
	# Set image's largest dimension to target size, and fill the rest with black pixels
	if L is None:
		rescaled=0
	else:
		image,rescaled = ResizeWithProportions(image, L) # width and height are assumed to be the same (assertion at the beginning)
	npimage = np.array(image.copy(), dtype=np.float32)

	if show:
		image.show()
	image.close()
	return npimage, rescaled


def LoadImages(datapath, L, class_select=None, training_data=True):
	'''
	Uses the data in datapath to create a DataFrame with images only. 
	This cannot be a particular case of the mixed loading, because the mixed depends on the files written in the features.tsv file, whereas here we fetch the images directly.

	Arguments:
	datapath 	 - the path where the data is stored. Inside datapath, we expect to find directories with the names of the classes
	L 			 - images are rescaled to a square of size LxL (maintaining proportions)
	class_select - a list of the classes to load. If None (default), loads all classes 
	training_data- a boolean variable to decide the structure of the directories
	Output:
	df 			 - a dataframe with classname, npimage, rescaled.
	'''

	df = pd.DataFrame()
	class_select=ReduceClasses(datapath, class_select)	# Decide whether to use all available classes

	# The following condition is because the taxonomists used different directory structures
	names='/training_data/*.jp*g' if training_data==True else '/*.jp*g'

	for c in class_select:
		

		classImages = glob.glob(datapath+'/'+c+'/'+names)
		dfClass=pd.DataFrame(columns=['classname','npimage'])

		print('class: {} ({})'.format(c, len(classImages)))

		for i,imageName in enumerate(classImages):

			npimage,rescaled=LoadImage(imageName,L)
			dfClass.loc[i] = [c,npimage]

		df=pd.concat([df,dfClass], axis=0)

	df.npimage = df.npimage / 255.0 

	return df.reset_index(drop=True)



def LoadImageList(im_names, L, show=False):
	''' 
	Function that loads a list of images given in im_names, and returns 
	them in a numpy format that can be used by the classifier.
	'''
	npimages=np.ndarray((len(im_names),L,L,3))

	for i,im_name in enumerate(im_names):
		npimage,rescaled=LoadImage(im_name,L, show)
		npimages[i]=npimage
	return npimages/255.0


class Cdata:

	def __init__(self, datapath, L=None, class_select=None, kind='mixed', training_data=True):
		self.datapath=datapath
		if L is None and kind!='feat':
			print('CData: image size needs to be set, unless kind is \'feat\'')
			raise ValueError
		self.L=L
		self.class_select=class_select
		self.kind=kind
		self.df=None
		self.y=None
		self.X=None
		self.Load(self.datapath, self.L, self.class_select, self.kind, training_data=training_data)
		return


	def Load(self, datapath, L, class_select, kind='mixed', training_data=True):
		''' 
		Loads dataset 
		For the moment, only mixed data. Later, also pure images or pure features.
		'''
		self.L=L
		self.datapath=datapath
		self.class_select=class_select
		self.kind=kind

		if kind=='mixed':
			self.df = LoadMixed(datapath, L, class_select, alsoImages=True)
		elif kind=='feat':
			self.df = LoadMixed(datapath, L, class_select, alsoImages=False)
		elif kind=='image':
			self.df = LoadImages(datapath, L, class_select, training_data=training_data)
		else:
			raise NotImplementedError('Only mixed, image or feat data-loading')

		self.classes=self.df['classname'].unique()
		self.kind=kind 		# Now the data kind is kind. In most cases, we had already kind=self.kind, but it the user tested another kind, it must be changed
		self.Check()  		# Some sanity checks on the dataset
		self.CreateXy()		# Creates X and y, i.e. features and labels
		return


	def Check(self):
		''' Basic checks on the dataset '''

		#Number of different classes
		classes=self.classes
		if len(classes)<2:
			print('There are less than 2 classes ({})'.format(len(classes)))
			raise ValueError

		# Columns potentially useful for classification
		ucols=self.df.drop(columns=['classname','url','file_size','timestamp'], errors='ignore').columns
		if len(ucols)<1:
			print('Columns: {}'.format(self.df.columns))
			raise ValueError('The dataset has no useful columns.')

		# Check for NaNs
		if self.df.isnull().any().any():
			print('There are NaN values in the data.')
			raise ValueError

		# Check that the images have the expected size
		if 'npimage' in self.df.columns:
			if self.df.npimage[0].shape != (self.L, self.L, 3):
				print('Cdata Check(): Images were not reshaped correctly: {} instead of {}'.format(self.npimage[0].shape, (self.L, self.L, 3)))

		return

	def CreateXy(self):
		''' 
		Creates features and target
		- removing the evidently junk columns.
		- allowing to access images and features separately and confortably
		'''

		self.y = self.df.classname
		self.X = self.df.drop(columns=['classname','url','file_size','timestamp'], errors='ignore')

		self.Ximage = self.X.npimage if (self.kind != 'feat') else None
		self.Xfeat  = self.X.drop(columns=['npimage'], errors='ignore') if (self.kind != 'image') else None

		return


def ReadArgsTxt(modelpath, verbose=False):
	'''
	Looks with what arguments the model was trained, and makes a series of consistency checks.
	Reads the following two files:
	- params.txt  (txt file with the simulation parameters)
	- classes.npy (numpy file with the list of classes)

	'''
	argsname=modelpath+'/params.txt'
	params={'L':None,
			'model':None,
			'layers':[None,None],
			'datapath':None,
			'outpath':None,
			'datakind':None,
			'ttkind':None
			}

	# Read Arguments
	with open(argsname,'r') as fargs:
		args=fargs.read()
		if verbose:
			print('---------- Arguments for generation of the model ----------')
			print('{} contains the following parameters:\n{}'.format(argsname,args))
			print('-----------------------------------------------------------')
		for s in re.split('[\,,\),\(]',args):
			if 'L=' in s:
				params['L']=np.int64(re.search('(\d+)',s).group(1))
			if 'model=' in s:
				params['model']=re.search('=\'(.+)\'$',s).group(1)
			if 'layers=' in s: #first layer
				params['layers'][0]=np.int64(re.search('=\[(.+)$',s).group(1))
			if re.match('^ \d+',s): #second layer
				params['layers'][1]=np.int64(re.match('^ (\d+)',s).group(1))
			if 'datapath=' in s:
				params['datapath']=re.search('=\'(.+)\'$',s).group(1)
			if 'outpath=' in s:
				params['outpath']=re.search('=\'(.+)\'$',s).group(1)
			if 'datakind=' in s:
				params['datakind']=re.search('=\'(.+)\'$',s).group(1)
			if 'ttkind=' in s:
				params['ttkind']=re.search('=\'(.+)\'$',s).group(1)
		if verbose:
			print('We retrieve this subset of parameters:\n',params)
			print('-----------------------------------------------------------')

	def OutputClassPaths():
		''' Auxiliary function for error messages '''
		print('\nclasses1 are the subdirectories in the following folder: ', datapath)
		print('classes1:',classes1)
		print('\nclasses2 full filename: ',modelpath+'classes2.npy')
		print('classes2:',classes2)
		print('')


	# Extract classes from npy file in output directory
	try:
		classes1=np.load(modelpath+'/classes.npy',allow_pickle=True)
	except FileNotFoundError:
		classes1=None		

	# Extract classes from data directory (if datapath is accessible)
	try:
		classes2 = [ name for name in os.listdir(params['datapath']) if os.path.isdir(os.path.join(params['datapath'], name)) ]
	except FileNotFoundError:
		classes2=None

	# If both class types, I compare them as a check. Otherwise, I take what there is.
	if (classes1 is None) and (classes2 is None):
		raise RuntimeError('Classes cannot be found from dataset nor from the model directory')
	elif classes1 is None:
		print('Unable to retrive classes from {}, so I use the classes in the datapath diretory {}'.format(modelpath+'/classes.npy',params['datapath']))
		classes=classes2
	elif classes2 is None:
		print('Unable to retrive classes from the datapath diretory {}, so I use the classes in {}'.format(params['datapath'],modelpath+'/classes.npy'))
		classes=classes1
	else:
		# If the two classes are not the same, we send an error.
		# In the future, we might have classes.npy dominate
		if len(classes1) != len(classes2):
			OutputClassPaths()
			raise IndexError('Number of classes in data directory is not the same as in classes.npy')
		if not np.all(set(classes1)==set(classes2)):
			OutputClassPaths()
			raise ValueError('Some of the classes in data directory are not the same as in classes.npy')
		classes=classes1 # classes1 and classes2 are the same

	return params, classes



if __name__=='__main__':
	pass





