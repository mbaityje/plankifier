#!/usr/bin/env python3
#
# Compares counts of species made by taxonomists with that of the classifier
#

import os, argparse, re, glob, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

df=pd.read_csv('../data/1_zooplankton_0p5x/validation/counts/counts_cleaned.csv', sep=';',encoding="ISO-8859-1")                                                                             
years=df['year'].unique()
nsam=len(df)
tot=np.ndarray(nsam, dtype=object)




classes=['eudiaptomus','cyclops','nauplius','daphnia','bosmina','diaphanosoma','kellikottia','keratella_cochlearis','keratella_quadrata','trichocerca', 'conochilus','asplanchna','ceratium','fragilaria','asterionella','dinobryon','uroglena','fish','leptodora','chaoborus','paradileptus',
'dirt_unknown','rotifer_other'
]

	
isample=0
for year in years:	
	for folder in df[df.year==year]['name_folder'].unique():

		# Population for the classifier
		pred_file='../data/1_zooplankton_0p5x/validation/counts/year_{}/{}/0000000000_subset_static_html/predict.txt'.format(year,folder)
		print(pred_file)
		dfcla   = pd.read_csv(pred_file, sep=' ', header=0, names=['name','pred','conf','pred2','conf2','pr'])['name'].value_counts()

		# Population for the taxonomists
		dftax=df[(df.year==year) & (df.name_folder==folder)][classes]

		# Combine them
		dfhisto = pd.DataFrame( np.c_[np.zeros(len(classes)), np.zeros(len(classes)), np.full(len(classes),np.inf)], index=classes, columns=['tax','cla','rel_err'])
		for c in classes:
			dfhisto.loc[c]['tax'] = dftax[c].item()
			dfhisto.loc[c]['cla'] = dfcla[c].item() if c in dfcla.index else 0
			dfhisto.loc[c]['rel_err'] = (dfhisto.loc[c]['cla'] - dfhisto.loc[c]['tax'] ) / dfhisto.loc[c]['tax'] if 0!=dfhisto.loc[c]['tax'] else 1 # if denominator is zero, set rel error to 100%

			print(dfhisto.loc[c]['tax'], dfhisto.loc[c]['cla'],dfhisto.loc[c]['rel_err'])


		# Update the array containing all the data
		tot[isample]=dfhisto
		isample+=1


#########
# PLOTS #
#########
dfhistotot = pd.DataFrame( np.c_[np.zeros(len(classes)), np.zeros(len(classes)), np.full(len(classes),np.inf), np.full(len(classes),np.inf), np.full(len(classes),np.inf)], index=classes, columns=['tax','cla','acc','rel_err','rel_err2'])
tot_acc, tot_sam=0,0
for c in classes:
	# if c in ['dirt_unknown','rotifer_other']: continue
	dfhistotot.loc[c]['tax']=np.sum([tot[isam].loc[c]['tax'] for isam in range(nsam)])
	dfhistotot.loc[c]['cla']=np.sum([tot[isam].loc[c]['cla'] for isam in range(nsam)])
	dfhistotot.loc[c]['acc']=dfhistotot.loc[c]['cla']/dfhistotot.loc[c]['tax'] if dfhistotot.loc[c]['tax']>dfhistotot.loc[c]['cla'] else dfhistotot.loc[c]['tax']/dfhistotot.loc[c]['cla']
	dfhistotot.loc[c]['rel_err']=np.mean([tot[isam].loc[c]['rel_err'] for isam in range(nsam)])
	dfhistotot.loc[c]['rel_err2']=np.mean( [np.square(tot[isam].loc[c]['rel_err']) for isam in range(nsam)])
	print(c,dfhistotot.loc[c]['acc'])

	tot_acc += dfhistotot.loc[c]['tax']*dfhistotot.loc[c]['acc']
	tot_sam += dfhistotot.loc[c]['cla']
tot_acc/=tot_sam
print('Total accuracy:', tot_acc)

# COUNTS
dfhistotot[['tax','cla']].sort_values(by='cla', ascending=False).plot.bar()
plt.show()

# ACCURACY
dfhistotot.sort_values(by='cla', ascending=False)['acc'].plot.bar()
plt.show()

# RELATIVE ERROR
dfhistotot.sort_values(by='cla', ascending=False)['rel_err'].plot.bar()
plt.show()
dfhistotot.sort_values(by='cla', ascending=False)['rel_err2'].plot.bar()
plt.show()

