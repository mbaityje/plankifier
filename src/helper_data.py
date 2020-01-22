#!/usr/bin/env python
#
# Helper functions for data manipulation
#
################################################
from PIL import Image


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
	# print('zzz:',im.size)

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




if __name__=='__main__':
	pass