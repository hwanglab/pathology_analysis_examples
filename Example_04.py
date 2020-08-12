'''
purpose: nuclei segmentation in pathoogy images

author: Hongming Xu
email: mxu@ualberta.ca
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries

I=plt.imread('TCGA-3M-AB46-01Z-00-DX1.jpg')

# color image to gray scale image
# the equation for gray image: Y = 0.2125 R + 0.7154 G + 0.0721 B
I_gray=(rgb2gray(I)*255).astype(np.uint8)

# note that: plt.subplot(nrows, ncolumns, index)
plt.subplot(121)
plt.imshow(I)
plt.subplot(122)
plt.imshow(I_gray,cmap='gray')
plt.pause(2)
plt.close()

# ostu's threhsold segmentation
thresh=threshold_otsu(I_gray)
bw_n0=I_gray<thresh
# remove small binary nosiy regions with the size less than noise_size
noise_size=100
bw_n1=remove_small_objects(bw_n0, noise_size, connectivity=8)

# find the nuclei boundaries
bb=find_boundaries(bw_n1)
I2=I.copy()  # use copy() function here to ensure I2 is writable
I2[:,:,0][bb]=255
I2[:,:,1][bb]=0
I2[:,:,2][bb]=0

plt.subplot(221)
plt.imshow(bw_n0, cmap='gray')
plt.subplot(222)
plt.imshow(bw_n1, cmap='gray')
plt.subplot(223)
plt.imshow(I2)
plt.pause(3)
plt.close()