'''
purpose: morphological processing+watershed segmentation

author: Hongming Xu
email: mxu@ualberta.ca
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, \
    opening, closing, square, label, erosion, watershed

from skimage.segmentation import find_boundaries
from scipy import ndimage as ndi

I=plt.imread('TCGA-3M-AB46-01Z-00-DX1.jpg')

# color image to gray scale image
# the equation for gray image: Y = 0.2125 R + 0.7154 G + 0.0721 B
I_gray=(rgb2gray(I)*255).astype(np.uint8)

# ostu's threhsold segmentation
thresh=threshold_otsu(I_gray)
bw_n0=I_gray<thresh
# remove small binary nosiy regions with the size less than noise_size
noise_size=200
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


# practice morphological processing
# You should observe what the purpose of opening and closing operations?????
bw_n2=opening(bw_n1,square(5))
bw_n3=closing(bw_n2,square(5))
bw_n3=remove_small_objects(bw_n3, noise_size, connectivity=8)
bw_n4=remove_small_holes(bw_n3,noise_size/2)
plt.subplot(224)
plt.imshow(bw_n4,cmap='gray')
plt.pause(3)
plt.close()

bb=find_boundaries(bw_n4)
I2=I.copy()  # use copy() function here to ensure I2 is writable
I2[:,:,0][bb]=255
I2[:,:,1][bb]=255
I2[:,:,2][bb]=0

plt.subplot(221)
plt.imshow(I2)
plt.close()

## iteratively erosion algorithm to find markers for watershed
bw_nn=label(bw_n4)
obj_n=np.max(bw_nn)
bw_mark=np.zeros(bw_n4.shape,dtype=bool)
for tt in range(1,obj_n+1):
    bw_temp=np.zeros(bw_n4.shape)
    bw_temp[bw_nn==tt]=1
    thr=np.sum(bw_temp)*0.2
    while np.sum(bw_temp)>thr:
        bw_temp=erosion(bw_temp,square(3))
    bw_mark[bw_temp==1]=1

# marker-controlled watershed segmentation for nuclei
markers = ndi.label(bw_mark)[0]
distance = -ndi.distance_transform_edt(bw_n4)
labels = watershed(distance, markers=markers, mask=bw_n4)

bb=find_boundaries(labels)
I2=I.copy()  # use copy() function here to ensure I2 is writable
I2[:,:,0][bb]=255
I2[:,:,1][bb]=0
I2[:,:,2][bb]=0
I2[:,:,0][bw_mark]=0
I2[:,:,1][bw_mark]=0
I2[:,:,2][bw_mark]=255


plt.imshow(I2)

plt.pause(3)
plt.close()
