'''
purpose: learn to apply thresholding on pathology images

author: Hongming Xu
if you found any bugs to run the code, please email to:
email: mxu@ualberta.ca
'''

import openslide
import matplotlib.pyplot as plt
import numpy as np

# you need to download the example wsi slide from tcga website.
wsi='../../TCGA-BL-A3JM-01Z-00-DX1.33E53972-CEA4-4D84-A5D2-7DAD7B0C27F8.svs'

# open slide image
Slide = openslide.OpenSlide(wsi)

# slide has multi-levels, get the number of levels
lc=Slide.level_count
# get width, height of image at the lowest resolution
lrHeight = Slide.level_dimensions[lc-1][1]
lrWidth = Slide.level_dimensions[lc-1][0]

# read in whole slide at the lowest magnification
wsi_low = Slide.read_region((0, 0), lc-1, (lrWidth, lrHeight))

# convert PIL image to array
image=np.asarray(wsi_low)
# print image shape, not that image has RGBA format, so the third dimension has 4 channels
print('image shape is (%d,%d,%d)' % (image.shape))

# if you want to read only RGB channels, you can:
wsi_low = Slide.read_region((0, 0), lc-1, (lrWidth, lrHeight)).convert('RGB')
image=np.asarray(wsi_low)
print('image shape is (%d,%d,%d)' % (image.shape))

# show low resolution image
plt.subplot(131)
plt.imshow(image)
plt.title('low resoltuion image')

# thresholding
thr=230 # let us empically set a threshold
# we generate a binary mask with pixels in RGB channkels < 230 as the foreground tissues
temp=np.logical_and(image[:,:,0]<230, image[:,:,1]<230)
tissue=np.logical_and(temp,image[:,:,2]<230)

plt.subplot(132)
plt.imshow(tissue)
plt.title('tissue foreground mask')

# stack on the third dimension
tissue_3d=np.stack((tissue,tissue,tissue), axis=2)
tissue_image=np.multiply(tissue_3d,image) # element multiplication
plt.subplot(133)
plt.imshow(tissue_image)
plt.title('tissue foreground regions')
plt.show()

