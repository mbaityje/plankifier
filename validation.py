# Imports
'''
Program that performas a validation on the validation dataset. 
Several models, validation directories, ensembling methods, abstention thresholds are used.
Some parameters (especially those involving paths) are hard-coded because it's more convenient than putting them as command line argument, since they are always the same.
Arguments:
- thresholds
- ensMethods (if None, doesn't ensemble)
- modelnames
- weightnames
There is a memory problem when many models are loaded together.
The several metrics are given
- TP: True Positives
- FP: False Positives
- TN: True Negatives
- FN: False Negatives
- recall: TP/(TP+FN)
- precision: TP/(TP+FP)
- specificity: TN/(TN+FP)
- informedness: specificity + recall - 1
- accuracy: (TP+TN)/(TP+TN+FP+FN)
Runs as:
python validation.py -thresholds 0.0 0.9 -modelnames './trained-models/conv2/keras_model.h5' -weightnames './trained-models/conv2/bestweights.hdf5'
'''


import os, keras, argparse, glob, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_data as hd, helper_models as hm
import train as t
import predict as pred
import shutil
from pathlib import Path
from sklearn.metrics import classification_report,confusion_matrix

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Cval:

	def __init__(self, modelnames, testdirs, weightnames, labels, ensMethods,datapaths,classifier,class_select, thresholds=0, training_data=False):

		self.InitClasses()

		self.thresholds = thresholds
		self.ensMethods = ensMethods
		self.classifier = classifier
		self.datapaths=datapaths
		self.class_select = class_select

		self.ensembler=pred.Censemble(
									modelnames=modelnames, 
									weightnames=weightnames,
									testdirs=testdirs, 
									labels=labels,
									screen=False)
		# self.res     = self.Cvres()

		self.ensembler.MakePredictions()
		self.nimages=len(self.ensembler.im_names)

		return

	def InitClasses(self):
		                    
		self.__PLANKTONCLASSES__ = [\
		'asplanchna','asterionella','aphanizomenon','bosmina','ceratium','chaoborus',
		'conochilus','cyclops','daphnia','diaphanosoma',
		'dinobryon','eudiaptomus','fragilaria','hydra','kellikottia',
		'keratella_cochlearis','keratella_quadrata','leptodora','nauplius','paradileptus',
		'polyarthra','rotifers','synchaeta','trichocerca','uroglena']
		self.__JUNKCLASSES__=[\
		'copepod_skins','daphnia_skins','diatom_chain','dirt','filament','fish','maybe_cyano','unknown','unknown_plankton']
		self.__ALLCLASSES__ = self.__PLANKTONCLASSES__+self.__JUNKCLASSES__


	def PerClassValidation(self, ensembler,labels):

		df_res = pd.DataFrame(index=ensembler.classnames, columns=['TP','FP','TN','FN','recall','precision','specificity','informedness','accuracy'])
		df_res.fillna(0) # Fill with zeros
        
		df_res1 = pd.DataFrame(index=['Overall'], columns=['TP','FP','TN','FN','recall','precision','specificity','informedness','accuracy'])
		df_res1.fillna(0) # Fill with zeros
		df_res=df_res.append(df_res1)
        

		# Per class validation
		for myclass in labels:
			# indices of the truths
			idLabels = list(filter(lambda i: ensembler.im_labels[i]==myclass, range(self.nimages) ))
			# how many of the truths were identified
			TP=(ensembler.guesses[idLabels,1]==ensembler.im_labels[idLabels]).astype(float).sum().astype(float)
			# how many of the truths were NOT identified
			FN=float(len(idLabels)-TP)
			# Indices of the guesses
			idGuesses = list(filter(lambda i: ensembler.guesses[i,1]==myclass, range(self.nimages) ))
			# Number of false positives
			FP=(ensembler.guesses[idGuesses,1]!=ensembler.im_labels[idGuesses]).astype(float).sum().astype(float)
			# Number of true negatives
			TN=float(len(ensembler.im_labels)-TP-FN-FP)

			df_res.loc[myclass]['TP'] = TP
			df_res.loc[myclass]['FN'] = FN
			df_res.loc[myclass]['FP'] = FP
			df_res.loc[myclass]['TN'] = TN

			df_res.loc[myclass]['recall'      ] = TP/(TP+FN) if TP+FN>0 else np.nan
			df_res.loc[myclass]['precision'   ] = TP/(TP+FP) if TP+FP>0 else np.nan
			df_res.loc[myclass]['specificity' ] = TN/(TN+FP) if FP+TN>0 else np.nan
			df_res.loc[myclass]['informedness'] = df_res.loc[myclass]['specificity' ]+df_res.loc[myclass]['recall' ]-1
			df_res.loc[myclass]['accuracy'    ] = (TP+TN)/(TP+TN+FP+FN)
                   
		df_res.loc['Overall']['TP'] = '-'
		df_res.loc['Overall']['FN'] = '-'
		df_res.loc['Overall']['FP'] = '-'
		df_res.loc['Overall']['TN'] = '-'

		df_res.loc['Overall']['recall'      ] = df_res.recall.mean()
		df_res.loc['Overall']['precision'   ] = df_res.precision.mean()
		df_res.loc['Overall']['specificity' ] = df_res.specificity.mean()
		df_res.loc['Overall']['informedness'] = df_res.informedness.mean()
		df_res.loc['Overall']['accuracy'    ] = df_res.accuracy.mean()    

# 		df_res.loc['mean_tot'  ] = df_res.mean()
# 		df_res.loc['mean_junk' ] = df_res.loc[self.__JUNKCLASSES__    ].mean()
# 		df_res.loc['mean_plank'] = df_res.loc[self.__PLANKTONCLASSES__].mean()
		
# 		df_res.loc['mean_tot'].accuracy = 1+df_res.loc[self.__ALLCLASSES__].TP/df_res.loc[self.__ALLCLASSES__].FP # TP/(TP+FP)

		return df_res

	def Sweep(self,labels,outpath,save_misclassified):
		for method in self.ensMethods:
			for absthres in self.thresholds:
				print('\nMethod:',method, '\tAbs-threshold:',absthres)
				self.ensembler.Ensemble(method=method, absthres=absthres)
				self.df_res=self.PerClassValidation(self.ensembler,labels)
                
				if save_misclassified =='yes':
					misclassified = np.argwhere(self.ensembler.im_labels!=self.ensembler.guesses[:,1])
					for x in misclassified:
						for indices in x:     
							destination=(' '.join(map(str, outpath)))+self.ensembler.im_labels[indices]+'_labelled_as_'+self.ensembler.guesses[indices,1]
							Path(destination).mkdir(parents=True, exist_ok=True)
							shutil.copy(self.ensembler.im_names[indices], destination)   
                            
				clf_report=classification_report(self.ensembler.im_labels, self.ensembler.guesses[:,1])
				conf_matrix=confusion_matrix(self.ensembler.im_labels, self.ensembler.guesses[:,1])
				print (self.df_res)
				print (clf_report)           
				print (conf_matrix)             
				filename_to_save=((' '.join(map(str, outpath))+'ValidationReport.txt'))
				Path((' '.join(map(str, outpath)))).mkdir(parents=True, exist_ok=True)
				f = open(filename_to_save,'w')
				f.write('\n\nMethod:{}\n\nAbs-threshold:{}\n\nPerClassValidation:\n{}\n\nClassification Report\n{}\n\nConfusion Matrix\n{}\n'.format(method,absthres, self.df_res,clf_report, conf_matrix))   
				f.close()
                
# 				print(self.df_res.loc['mean_plank'])
# 				self.Plot()

	def Plot(self):

		ax1=plt.subplot(2,1,1)
		ax1.set_ylabel('Recall')
		ax1.set_ylim((0,1))
		ax1.tick_params(axis='x', which='major', labelsize=5)
		plt.xticks(rotation=90)

		# Bar plot for all the plankton classes
		plt.bar(self.df_res.loc[self.__PLANKTONCLASSES__].index, self.df_res.loc[self.__PLANKTONCLASSES__].recall, width=.9, edgecolor='black', label='Per class')
		# Average recall over all plankton classes
		plt.plot(np.arange(len(self.__PLANKTONCLASSES__)), self.df_res.loc['mean_plank'].recall*np.ones(len(self.__PLANKTONCLASSES__)),'--', 
				linewidth=0.5,color='black', label='Average {}'.format(np.round(self.df_res.loc['mean_plank'].recall, decimals=2)))
		ax1.legend(loc='lower right')

		# Put extra labels on the plot with more info
		for i,c in enumerate(self.__PLANKTONCLASSES__):
			if np.isnan(self.df_res.loc[c]['TP']):
				label = 'NaN'
			else:
				hits  = int(self.df_res.loc[c]['TP'])
				total = int(self.df_res.loc[c][['TP','FN']].to_numpy().sum())
				label = str(hits)+'/'+str(total)
			plt.annotate(label , xy=(i-0.4, 0.5), fontsize=5)


		ax2=plt.subplot(2,1,2)
		ax2.set_ylabel('Precision')
		ax2.set_ylim((0,1))
		ax2.tick_params(axis='x', which='major', labelsize=5)
		plt.xticks(rotation=90)

		# Bar plot for all the plankton classes
		plt.bar(self.df_res.loc[self.__PLANKTONCLASSES__].index, self.df_res.loc[self.__PLANKTONCLASSES__].precision, width=.9, edgecolor='black',label='Per class')
		# Average precision over all plankton classes
		plt.plot(np.arange(len(self.__PLANKTONCLASSES__)),self.df_res.loc['mean_plank'].precision*np.ones(len(self.__PLANKTONCLASSES__)),'--', linewidth=0.5,color='black', label='Average {}'.format(np.round(self.df_res.loc['mean_plank'].precision,decimals=2)))
		ax2.legend(loc='lower right')


		# Put extra labels on the plot with more info
		for i,c in enumerate(self.__PLANKTONCLASSES__):
			if np.isnan(self.df_res.loc[c]['TP']):
				label = 'NaN'
			else:
				hits  = int(self.df_res.loc[c]['TP'])
				total = int(self.df_res.loc[c][['TP','FP']].to_numpy().sum())
				label = str(hits)+'/'+str(total)
			plt.annotate(label , xy=(i-0.4, 0.5), fontsize=5)


		plt.show()

		return




if __name__=='__main__':

    
	parser = argparse.ArgumentParser(description='Load a model and use it to make predictions on images')
	parser.add_argument('-ensMethods', nargs='+', default=None, help='Ensembling methods. Choose from: \'unanimity\',\'majority\', \'leader\', \'weighted-majority\'. Weighted Majority implements abstention in a different way (a good value is 1).')
	parser.add_argument('-class_select', nargs='*', default=None, help='List of classes to be looked at (put the class names one by one, separated by spaces). If None, all available classes are studied.')
	parser.add_argument('-classifier', choices=['binary','multi','versusall'], default='multi', help='Choose "binary class " or "multiclass" classifier')
	parser.add_argument('-datapaths', nargs='*', default=['/local/kyathasr/plankifier/data/1_zooplankton_0p5x/validation/validation_2020.04.28/'], help="Directories with the data.")
	parser.add_argument('-outpath', nargs='*', default=['/local/kyathasr/plankifier/out/'], help="Directories with the data.")
	parser.add_argument('-thresholds', nargs='+', default=[0.0], type=float, help='Abstention thresholds on the confidence (a good value is 0.8, except for weighted-majority, where it should be >=1).')
	parser.add_argument('-weightnames', nargs='*', default=['/local/kyathasr/plankifier/trained-models/mobilenet3/bestweights.hdf5'], help='Name of the weights file (.hdf5, with path)')
	parser.add_argument('-modelnames', nargs='*', default=['/local/kyathasr/plankifier/trained-models/mobilenet3/keras_model.h5'], help='')
	parser.add_argument('-save_misclassified', choices=['yes','no'], default='no', help="Whether to save misclassified images in a directory or not")  
# 	parser.add_argument('-misclassified_dir', nargs='*', default=['/local/kyathasr/plankifier/out/'], help="Directory to save misclassified images")
	args=parser.parse_args()

    
# 	datapaths=['/local/kyathasr/plankifier/data/1_zooplankton_0p5x/validation/validation_2020.04.28/']
# 	classifier='binary'
# 	class_select=None
# 	testdirs=[]
# 	labels=[]

	labels,Class_labels_original,testdirs=hm.get_testdirs_and_labels(args.datapaths,
                                                                     args.classifier,args.class_select)

# 	# testdirs = testdirs_tommy
# 	testdirs=testdirs_all
# 	labels = [os.path.split(td)[1] for td in testdirs]

	if args.ensMethods is None:

		for m in args.modelnames:
			print('model:',m)

			validator = Cval(modelnames=[m],
								testdirs=testdirs,
								datapaths=args.datapaths,
								labels=labels,
								ensMethods=['leader'],
								classifier=args.classifier,
								class_select=args.class_select,
								thresholds=args.thresholds,
								weightnames=args.weightnames,
								training_data=True
							)
			validator.Sweep(labels,args.outpath,args.save_misclassified)

	else:
		print('models:',args.modelnames)

		validator = Cval(modelnames=args.modelnames, 
							testdirs=testdirs,
							datapaths=args.datapaths,
							labels=labels,
							ensMethods=args.ensMethods,
							classifier=args.classifier,
							class_select=args.class_select,
							thresholds=args.thresholds,
							weightnames=args.weightnames,
							training_data=True
							)
		validator.Sweep(labels,args.outpath,args.save_misclassified)
