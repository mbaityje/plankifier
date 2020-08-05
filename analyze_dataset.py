#!/usr/bin/env python3
# 
# Gives features of a given dataset.
# 
# To know the input parameters, launch as:
#
# 	python analyze_dataset.py -h
#
# Example of usage that studies two datasets together:
#
# 	python analyze_dataset.py -datapath ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06
# 
#########################################################################

import os, sys, argparse, glob, numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#
# Input
#

parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
parser.add_argument('-datapath', nargs='*', default=['./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/'], help="Path of the dataset")
parser.add_argument('-outpath', default='./out/', help="Path of the output")
parser.add_argument('-name', default=None, help="Name for output")
args=parser.parse_args()

#
# Set Output
#

# Decide name for the output

ndatasets = len(args.datapath)
if args.name==None:
	if ndatasets==1:
		args.name = os.path.basename(os.path.dirname(args.datapath[0]+'/')) # Append a slash because if it is not present dirname gives a different result
	else:
		args.name = 'combined'

# Output directory and files

outdir=args.outpath+'/dataset_info/'
try: 
    os.mkdir(outdir)
except FileExistsError: 
    pass
outname =outdir+'/'+args.name+'.txt'


#
# Read Data
#

# Loop through the dataset

sizes=[[] for idata in range(ndatasets)]
classes = {'name': list(set([ name for idata in range(ndatasets) for name in os.listdir(args.datapath[idata]) if os.path.isdir(os.path.join(args.datapath[idata], name))]))} # Every directory corresponds to a class. Files that aren't directories are ignored. All datasets are searched, and repeated directories count as a single class.
classes['num']    = len(classes['name'])
classes['num_ex'] =  np.zeros(classes['num'], dtype=int)


for ic in range(classes['num']):

	c=classes['name'][ic]
	# print(c)

	for idata in range(ndatasets):
		# print('idata:',idata)
		classPath=args.datapath[idata]+'/'+c+'/training_data/*.jp*g'
		classImages = glob.glob(classPath)
		classes['num_ex'][ic] += len(classImages) # number of examples per class

		# print(np.ndarray( (classes['num_ex'][ic],3),dtype=int) )
		sizes[idata].append( np.ndarray( (len(classImages), 3),dtype=int) ) # Container for raw image sizes of class ic
		for i,imageName in enumerate(classImages):
			image = Image.open(imageName)
			npimage = np.array(image.copy() )

			# print(sizes[idata][ic])
			# print(npimage.shape)
			assert( npimage.shape[2]==3) #We expect these images to have three channels
			sizes[idata][ic][i]=npimage.shape
			image.close()

classes['tot_ex'] = classes['num_ex'].sum()
isort=np.argsort(classes['num_ex'])[::-1] #sort the classes by their abundance


# Make a list that contains the sizes of the images from all datasets
all_sizes=np.ndarray(classes['num'] ,dtype=object)
for ic in range(classes['num']):
	all_sizes[ic] = sizes[0][ic].tolist()
	for idata in range (1,ndatasets):
		all_sizes[ic].extend(sizes[idata][ic])
	all_sizes[ic] = np.array(all_sizes[ic])

#
# Text Output
#

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


#
# Figures
#

# Histogram of class abundances

num_ex_sort=classes['num_ex'][isort]
name_sort=[classes['name'][i] for i in isort]
plt.subplots_adjust(bottom=0.35)
ax=plt.subplot(1, 1, 1)
ax.set_title('Class abundances - '+str(classes['tot_ex'])+' images')
ax.set_xlabel('Class')
ax.set_ylabel('Abundance')
ax.set_yscale('log')
ax.bar(name_sort,num_ex_sort, color='red') #sorted
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.savefig(outdir+args.name+'_class-abundances.png')
plt.show()


# Total Distribution of image widths (first image) and image heights (second image)

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.5, top=0.9, wspace=0.2, hspace=1.)
ax1=plt.subplot(2, 1, 1)
ax1.set_xlabel('Width in pixels')
ax1.set_ylabel('Number of raw images')
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.title('Raw image width distribution')
for ic in range(classes['num']):
	plt.hist(all_sizes[ic][:,0], label=classes['name'][ic], histtype='step')
legend=plt.legend(bbox_to_anchor=(1.0, 1), ncol=2, fontsize=8)
ax2=plt.subplot(2, 1, 2)
plt.title('Raw image height distribution')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Height in pixels')
ax2.set_ylabel('Number of raw images')
for ic in range(classes['num']):
	plt.hist(all_sizes[ic][:,1], label=classes['name'][ic], histtype='step')
plt.savefig(outdir+args.name+'_image-sizes.png')
plt.show()

