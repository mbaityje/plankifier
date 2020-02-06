#!/usr/bin/env python3
# 
# Gives features of a given dataset.
# 
# Launch as:
# 	python analyze_dataset.py -datapath='./data/2020.02.02_zooplankton_trainingset_EWA/' -kind='mixed'
#
#	or:
#
# 	python analyze_dataset.py -datapath=./data/2019.11.20_zooplankton_trainingset_15oct_TOM -kind='image'
# 
#########################################################################

import os, sys, argparse, glob, numpy as np
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
parser.add_argument('-datapath', default='./data/2020.02.02_zooplankton_trainingset_EWA/', help="Path of the dataset")
parser.add_argument('-outpath', default='./out/', help="Path of the output")
parser.add_argument('-name', default=None, help="Name for output")
parser.add_argument('-kind', default='mixed', choices=['tsv','image','mixed'], help="If tsv, expect a single tsv file; if images, each class directory has only images inside; if mixed, expect a more complicated structure defined by the output of SPCConvert")
args=parser.parse_args()
if args.name==None:
	args.name = os.path.basename(os.path.dirname(args.datapath+'/')) # Append a slash because if it is not present dirname gives a different result
outdir=args.outpath+'/dataset_info/'
try: 
    os.mkdir(outdir)
except FileExistsError: 
    pass
outname =outdir+'/'+args.name+'.txt'

# glob.glob("data/2020.02.02_zooplankton_trainingset_EWA/bosmina/training_data/*.*[!b]")
# Loop through the dataset
sizes=[]
classes = {'name': [ name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name)) ]} # Every directory corresponds to a class. Files that aren't directories are ignored
classes['num']    = len(classes['name'])
classes['num_ex'] =  np.zeros(classes['num'], dtype=int)
for ic in range(classes['num']):
	c=classes['name'][ic]

	if args.kind == 'image':
		classPath=args.datapath+'/'+c+'/*.jp*g'
	elif args.kind == 'mixed':
		classPath=args.datapath+'/'+c+'/training_data/*.jp*g'
	elif args.kind == 'tsv':
		raise NotImplementedError('I did not implement yet tsv only')
	else:
		raise ValueError('Unknown args.kind {}'.format(args.kind))

	# classImages = os.listdir(classPath)
	# classImages = [im for im in classImages if any(ext in im for ext in ['jpg','jpeg','JPG','JPEG'])] # Keep only reasonable extensions

	classImages = glob.glob(classPath)


	classes['num_ex'][ic] = len(classImages) # number of examples per class
	sizes.append( np.ndarray( (classes['num_ex'][ic],3),dtype=int) ) # Container for raw image sizes of class ic
	for i,imageName in enumerate(classImages):
		image = Image.open(imageName)
		npimage = np.array(image.copy() )
		sizes[ic][i]=npimage.shape
		assert( npimage.shape[2]==3) #We expect these images to have three channels
		image.close()
classes['tot_ex'] = classes['num_ex'].sum()
isort=np.argsort(classes['num_ex'])[::-1] #sort the classes by their abundance


# Text Output
# with (sys.stdout if args.stdout else open(outname,'w')) as fout:
print('Printing dataset information to',outname)
with open(outname,'w') as fout:
	print('Dataset name:         ', args.name,        file=fout)
	print('Dataset path:         ', args.datapath,    file=fout)
	print('Number of classes:    ', classes['num'],   file=fout)
	print('Total size of dataset:', classes['tot_ex'],file=fout)
	print('Class names:          ', classes['name'],  file=fout)
	print('Class abundances:     ', classes['num_ex'],file=fout)
	for ic in range(classes['num']):
		print('{:20s}:\t{:5d}'.format(classes['name'][isort[ic]],classes['num_ex'][isort[ic]]), file=fout)


# Figures
# Histogram of class abundances
num_ex_sort=classes['num_ex'][isort]
name_sort=[classes['name'][i] for i in isort]
plt.title('Class abundances')
plt.xlabel('Class')
plt.ylabel('Abundance')
plt.bar(name_sort,num_ex_sort, color='red') #sorted
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.savefig(outdir+args.name+'_class-abundances.png')
plt.show()

# Total Distribution of Image heights
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.75, top=0.9, wspace=0.2, hspace=1.)
ax1=plt.subplot(2, 1, 1)
ax1.set_xlabel('Width in pixels')
ax1.set_ylabel('Number of raw images')
ax1.set_xscale('log')
plt.title('Raw image width distribution')
for ic in range(classes['num']):
	plt.hist(sizes[ic][:,0], label=classes['name'][ic], histtype='step')
legend=plt.legend(bbox_to_anchor=(1.0, 1))
ax2=plt.subplot(2, 1, 2)
plt.title('Raw image height distribution')
ax2.set_xscale('log')
ax2.set_xlabel('Height in pixels')
ax2.set_ylabel('Number of raw images')
for ic in range(classes['num']):
	plt.hist(sizes[ic][:,1], label=classes['name'][ic], histtype='step')
plt.savefig(outdir+args.name+'_image-sizes.png')
plt.show()

