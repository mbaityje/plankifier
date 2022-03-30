#!/usr/bin/env python
#
# Helper functions for training set management and preprocessing
#
##################################################################

import numpy as np, pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight # Added by SK


def unique_cols(df):
    ''' Returns one value per column, stating whether all the values are the same'''
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a[1:]).all(0)


class CTrainTestSet:
	''' 
	A class for extracting train and test sets from the original dataset, and preprocessing them.
	'''

	def __init__(self, X, y,filenames, ttkind='image',classifier=None,balance_weight=None, rescale=False, testSplit=0.2,valid_set=None, compute_extrafeat=None,random_state=12345):
		''' 
		X and y are dataframes with features and labels
		'''

		self.ttkind=ttkind
		self.testSplit=testSplit
		self.valid_set=valid_set
		self.random_state=random_state
		self.classifier=classifier
		self.balance_weight=balance_weight
		self.compute_extrafeat=compute_extrafeat
  
		# Take care of the labels
		self.y=y
		self.VectorizeLabels(classifier)
		self.filenames=filenames
#		if classifier == 'binary':
#			UnvectorizeLabels(self, y)

		# Now the features
		if ttkind == 'image' and compute_extrafeat =='no':
			self.X=self.ImageNumpyFromMixedDataframe(X)
		elif ttkind == 'feat':
			X = self.DropCols(X, ['npimage','rescaled'])
			X = self.RemoveUselessCols(X)
			self.X=np.array([X.to_numpy()[i] for i in range(len(X.index))])
		else:
			# This checks if there are images, but it also implicitly checks if there are features.
			# In fact, if there are only images, X is a series and has no attribute columns (I am aware this should be coded better). 
			if 'npimage' not in X.columns:
				raise RuntimeError('Error: you asked for mixed Train-Test, but the dataset you gave me does not contain images.')
			self.X=self.RemoveUselessCols(X) #Note that with ttkind=mixed, X stays a dataframe

    
		# Split train and test data
		self.Split(test_size=testSplit,valid_set=valid_set, random_state=random_state)

		# Rescale features
		if rescale == True:
			self.Rescale()
			self.rescale=True
		else:
			self.rescale=False

		return

	def VectorizeLabels(self,classifier):
		''' 
		Transform labels in one-hot encoded vectors 
		This is where we will act if we decide to train with HYBRID LABELS
		'''
		self.lb = LabelBinarizer()
# 		print('Multi_before_binrizer:',self.y)
		self.y = self.lb.fit_transform(self.y.tolist())
# 		print('Classifier',self.classifier)        
# 		print('Multi',self.y)
		if self.classifier == 'binary' or self.classifier=='versusall':
			self.y = np.hstack((1 - self.y,self.y))
# 			print('Binary',self.y)
        
		return

	def UnvectorizeLabels(self, y):
		''' Recovers the original labels from the vectorized ones '''
        
#		if classifier == 'binary':
#			self.y = np.hstack((1 - self.y,self.y))          
        
        
		return self.lb.inverse_transform(y) if classifier == 'multi' else self.lb.inverse_transform(y[:,1])
#		return self.lb.inverse_transform(y) 


	def ImageNumpyFromMixedDataframe(self, X=None):
		''' Returns a numpy array of the shape (nexamples, L, L, channels)'''
		if X is None:
			X=self.X

		# The column containing npimage
		im_col = [i for i,col in enumerate(X.columns) if col == 'npimage'][0] 
		
		return np.array([X.to_numpy()[i, im_col] for i in range( len(X.index) )])


	def Split(self, test_size=0.2,valid_set=None, random_state=12345):
		''' 
		Splits train and test datasets.
		Allows to put all the data in the test set by choosing test_size=1. This is useful for evaluation.
		Handles differently the mixed case, because in that case  X is a dataframe.
		'''
				
		if test_size<1:
            
			if valid_set == 'no':       
				self.trainX, self.testX, self.trainY, self.testY, self.trainFilenames, self.testFilenames = train_test_split(self.X, self.y,self.filenames, test_size=test_size, random_state=random_state, shuffle=True, stratify = self.y)
			elif valid_set == 'yes':  
				train_ratio = 0.70
				validation_ratio = 0.15
				test_ratio = 0.15
				self.trainX, self.test1X, self.trainY, self.test1Y, self.trainFilenames, self.test1Filenames = train_test_split(self.X, self.y,self.filenames, test_size=1-train_ratio,random_state=random_state, shuffle=True, stratify = self.y)   
				self.valX, self.testX, self.valY, self.testY, self.valFilenames, self.testFilenames = train_test_split(self.test1X, self.test1Y,self.test1Filenames, test_size=test_ratio/(test_ratio + validation_ratio), random_state=random_state, shuffle=True) 
                              
			y_integers = np.argmax(self.trainY, axis=1)
			if self.balance_weight=='yes':            
				class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
			else:
				class_weights = compute_class_weight(None, np.unique(y_integers), y_integers)
			self.class_weights = dict(enumerate(class_weights))

		else: # This allows us to pack everything into the test set
			self.trainX, self.testX, self.trainY, self.testY, self.trainFilenames, self.testFilenames = None, self.X, None, self.y


		if self.ttkind == 'mixed':
			# Images
			if self.trainX is not None:
				self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)
			self.testXimage = self.ImageNumpyFromMixedDataframe(self.testX)
			if valid_set=='yes':
				self.valXimage = self.ImageNumpyFromMixedDataframe(self.valX)

			#Features
			if self.trainX is not None:
				Xf=self.DropCols(self.trainX, ['npimage','rescaled'])
				self.trainXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
			Xf=self.DropCols(self.testX, ['npimage','rescaled'])
			self.testXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
			if valid_set=='yes':
				Xf=self.DropCols(self.valX, ['npimage','rescaled'])
				self.valXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
            
		elif self.ttkind == 'image' and self.compute_extrafeat =='yes':
			# Images
			if self.trainX is not None:
				self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)
			self.testXimage = self.ImageNumpyFromMixedDataframe(self.testX)
			if valid_set=='yes':
				self.valXimage = self.ImageNumpyFromMixedDataframe(self.valX)

			#Features
			if self.trainX is not None:
				Xf=self.DropCols(self.trainX, ['npimage','rescaled'])
				self.trainXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
			Xf=self.DropCols(self.testX, ['npimage','rescaled'])
			self.testXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
			if valid_set=='yes':
				Xf=self.DropCols(self.valX, ['npimage','rescaled'])
				self.valXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
            
		return

	def RemoveUselessCols(self, df):
		''' Removes columns with no information from dataframe '''
		# Select all columns except image
		morecols=[]
		cols=df.columns.tolist()

		if 'npimage' in cols:
			cols.remove('npimage')
			morecols=['npimage']

		# Remove all columns with all equal values
		badcols=np.where(unique_cols(df[cols]) == True)[0].tolist()
		badcols.reverse() # I reverse, because otherwise the indices get messed up when I use del

		for i in badcols:
		    del cols[i]

		cols = morecols+cols

		return df[cols]


	def Rescale(self):
		
		raise NotImplementedError('No preprocessing allowed until I write a function that allows to make the same preprocessing on validation data')
		
		if self.ttkind == 'mixed':
			self.RescaleMixed()
		elif self.ttkind == 'feat':
			self.RescaleFeat()
		elif self.ttkind == 'image' and self.compute_extrafeat =='yes':
			self.RescaleMixed()
		elif self.ttkind == 'image' and self.compute_extrafeat =='no':
			pass # We don't rescale the image
		else:
			raise NotImplementedError('CTrainTestSet: ttkind must be feat, image or mixed')
		return

	def RescaleMixed(self):
		''' 
		Rescales all columns except npimage to have mean zero and unit standard deviation 

		To avoid data leakage, the rescaling factors are chosen from the training set
		'''

		if self.trainX is None:
			print('No rescaling is performed because the training set is empty, but the truth is that in this case we should have rescaling parameters coming from elsewhere')
			return

		cols=self.trainX.columns.tolist()

		if 'npimage' in cols:
			cols.remove('npimage')

		# Set to zero mean and unit standard deviation
		x=self.trainX[cols].to_numpy()
		mu=x.mean(axis=0)
		sigma=np.std(x, axis=0, ddof=0)

		# Training set
		self.trainX[cols]-=mu          # Set mean to zero
		self.trainX[cols]/=sigma       # Set standard dev to one
		# Test set
		self.testX[cols]-=mu          # Set mean to zero
		self.testX[cols]/=sigma       # Set standard dev to one

		# These checks are only valid for the training set
		assert( np.all(np.isclose( self.trainX[cols].mean()  , 0, atol=1e-5)) ) # Check that mean is zero
		assert( np.all(np.isclose( np.std(self.trainX[cols], axis=0, ddof=0)  , 1, atol=1e-5)) ) # Check that std dev is unity

		return


	def RescaleFeat(self):
		''' 
		Rescales all columns

		To avoid data leakage, the rescaling factors are chosen from the training set
		'''
		
		# Set to zero mean and unit standard deviation
		mu=self.trainX.mean(axis=0)
		sigma=np.std(self.trainX, axis=0, ddof=0)

		# Training set
		self.trainX-=mu          # Set mean to zero
		self.trainX/=sigma       # Set standard dev to one
		# Test set
		self.testX-=mu          # Set mean to zero
		self.testX/=sigma       # Set standard dev to one

		# These checks are only valid for the training set
		assert( np.all(np.isclose( self.trainX.mean()  , 0, atol=1e-5)) ) # Check that mean is zero
		assert( np.all(np.isclose( np.std(self.trainX, axis=0, ddof=0)  , 1, atol=1e-5)) ) # Check that std dev is unity

		return


	def SelectCols(self, X, cols):
		''' 
		Keeps only the columns cols from the dataframe X. 
		cols is a list with the columns names
		'''

		if isinstance(X, pd.DataFrame): # Make sure it is not a series
			if set(cols).issubset(set(X.columns)): # Check that columns we want to select exist
				return X[cols]
			else:
				print('self.X.columns: {}'.format(self.X.columns))
				print('requested cols: {}'.format(cols))
				raise IndexError('You are trying to select columns that are not present in the dataframe')
		else:
			assert(len(cols)==1) # If it's a series there should be only one column
			assert(self.X.name==cols[0])# And that column should coincide with the series name
			return

	def DropCols(self, X, cols):
		''' 
		Gets rid of the columns cols from the dataframe X. 
		cols is a list with the columns names
		'''
		return X.drop(columns=cols, errors='ignore')

	def MergeLabels(self):
		''' Merges labels to create aggregated classes '''
		raise NotImplementedError
