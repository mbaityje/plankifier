#!/usr/bin/env python3
# 
# Gives features of a given dataset.
# 
# Launch as:
# 	python analyze_dataset.py 
#
# 
#########################################################################

import os, sys, argparse, numpy as np
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
parser.add_argument('-datapath', default='./data/zooplankton_trainingset/', help="Path of the dataset")
parser.add_argument('-outpath', default='./out/', help="Path of the output")
parser.add_argument('-name', default=None, help="Name for output")
args=parser.parse_args()
if args.name==None:
	args.name = os.path.basename(os.path.dirname(args.datapath+'/')) # Append a slash because if it is not present dirname gives a different result
outdir=args.outpath+'/dataset_info/'
try: 
    os.mkdir(outdir)
except FileExistsError: 
    pass
outname =outdir+'/'+args.name+'.txt'


# Loop through the dataset
sizes=[]
classes = {'name': os.listdir(args.datapath)}
classes['num']    = len(classes['name'])
classes['num_ex'] =  np.zeros(classes['num'], dtype=int)
for ic in range(classes['num']):
	c=classes['name'][ic]
	classPath=args.datapath+c+'/'
	classImages = os.listdir(classPath)
	classes['num_ex'][ic] = len(classImages) # number of examples per class
	sizes.append( np.ndarray( (classes['num_ex'][ic],3),dtype=int) ) # Container for raw image sizes of class ic
	for i,imageName in enumerate(classImages):
		imagePath = classPath+imageName
		image = Image.open(imagePath)
		npimage = np.array(image.copy() )
		sizes[ic][i]=npimage.shape
		assert( npimage.shape[2]==3) #We expect these images to have three channels
		image.close()
classes['tot_ex'] =  classes['num_ex'].sum()
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

