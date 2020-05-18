#!/usr/bin/env python
#
# Helper functions for training set management and preprocessing
#
##################################################################

import numpy as np, pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


class CTrainTestSet:
	''' 
	A class for extracting train and test sets from the original dataset, and preprocessing them.
	'''

	def __init__(self, X, y, ttkind='mixed', split=True):
		''' X and y are dataframes with features and labels'''
		self.ttkind=ttkind

		# Take care of the labels
		self.y=y
		self.VectorizeLabels()


		# Now the features

		if ttkind == 'image':
			self.X=self.ImageNumpyFromMixedDataframe(X)
		elif ttkind == 'feat':
			X=self.DropCols(X, ['npimage','rescaled'])
			self.X=np.array([X.to_numpy()[i] for i in range(len(X.index))])
		else:
			# This checks if there are images, but it also implicitly checks if there are features.
			# In fact, if there are only images, X is a series and has no attribute columns (I am aware this should be coded better). 
			if 'npimage' not in X.columns:
				raise RuntimeError('Error: you asked for mixed Train-Test, but the dataset you gave me does not contain images.')
			self.X=X #Note that with ttkind=mixed, X stays a dataframe
	
		if split:
			self.Split()

		return

	def VectorizeLabels(self):
		''' 
		Transform labels in one-hot encoded vectors 
		This is where we will act if we decide to train with HYBRID LABELS
		'''

		self.lb = LabelBinarizer()

		self.y = self.lb.fit_transform(self.y.tolist())
		return

	def UnvectorizeLabels(self, y):
		''' Recovers the original labels from the vectorized ones '''
		return self.lb.inverse_transform(y) 


	def ImageNumpyFromMixedDataframe(self, X=None):
		''' Returns a numpy array of the shape (nexamples, L, L, channels)'''
		if X is None:
			X=self.X

		# The column containing npimage
		im_col = [i for i,col in enumerate(X.columns) if col == 'npimage'][0] 
		
		return np.array([X.to_numpy()[i, im_col] for i in range( len(X.index) )])


	def Split(self, test_size=0.2, random_state=12345):
		''' handles differently the mixed case, because in that case  X is a dataframe'''
		self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

		if self.ttkind == 'mixed':
			# Images
			self.trainXimage = self.ImageNumpyFromMixedDataframe(self.trainX)
			self.testXimage = self.ImageNumpyFromMixedDataframe(self.testX)
			#Features
			Xf=self.DropCols(self.trainX, ['npimage','rescaled'])
			self.trainXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])
			Xf=self.DropCols(self.testX, ['npimage','rescaled'])
			self.testXfeat=np.array([Xf.to_numpy()[i] for i in range(len(Xf.index))])

		return


	def Rescale(self, cols=None):
		raise NotImplementedError

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
