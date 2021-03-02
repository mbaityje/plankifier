#!/usr/bin/env python3
#
# Compares counts of species made by taxonomists with that of the classifier
#

import sys, os, argparse, re, glob, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from collections import Counter
# from src import helper_data as hd, helper_models as hm
import predict as pr



#
# Input parameters (hardcoded, for the moment)
#
modelnames	= ['./trained-models/EfficientNetB7/keras_model_finetune.h5']
weightnames	= ['./trained-models/EfficientNetB7/bestweights_finetune.hdf5']
testdirs	= glob.glob('./data/1_zooplankton_0p5x/validation/counts/year_*/*/0000000000_subset_static_html/images/00000/')
testimages	= glob.glob('./data/1_zooplankton_0p5x/validation/counts/year_*/*/0000000000_subset_static_html/images/00000/*')
ensMethod	= 'majority'
threshold	= 0.8



nimages = len(testimages)
print('There are {} images in total'.format(nimages) )





#
# Produce classifier predictions
#

ensembler=pr.Censemble(modelnames  	= modelnames, 
						testdirs	= testdirs, 
						weightnames	= weightnames,
						screen		= False,
						training_data=False
						)
ensembler.MakePredictions()
ensembler.Ensemble(method=ensMethod, absthres=threshold)

# Create a dataframe with the classification counts
histo_cla = Counter(ensembler.guesses[:,1])
df_cla=pd.DataFrame(histo_cla.values(), index=histo_cla.keys(), columns=['cla'])



#
# Data from taxonomists
#
df_tax=pd.read_csv('./data/1_zooplankton_0p5x/validation/counts/counts_cleaned.csv', sep=';',encoding="ISO-8859-1")                                                                             

useless_cols = ['name_folder', 'ID','timestamp','year', 'month', 'day', 'hour', 'name_counter', 'total_ROI', 'comments']
classes = list(set(df_tax.columns)-set(useless_cols))
df_tax = pd.DataFrame(df_tax[classes].sum().values, index=df_tax[classes].sum().index, columns=['tax'])
print('According to the taxonomists, there are {} images'.format(df_tax.sum().values.item()))



#
# Concatenate and plot the two histograms
#

df=pd.concat([df_cla, df_tax], axis=1, ignore_index=False, sort=True)
ax=df[['tax','cla']].sort_values(by='tax', ascending=False).plot.bar(logy=True)
ax.set_ylabel("Taxon count")
plt.tight_layout()
plt.show()
plt.savefig('./out/Validation_counts2.png')


#
# Make a nicer plot that excludes classes that are not present in either tax or cla
#
ax=df[(df.tax>0) & (df.cla>0)].sort_values(by='tax', ascending=False).plot.bar(logy=True)
ax.set_ylabel("Taxon count")
plt.tight_layout()
plt.show()
plt.savefig('./out/Validation_counts_clean2.png')
