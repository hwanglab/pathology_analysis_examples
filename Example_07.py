'''
purpose: measure properties of segmented cell nuclei

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
from skimage.measure import regionprops,regionprops_table
import pandas as pd

I=plt.imread('TCGA-3M-AB46-01Z-00-DX1.jpg')

# color image to gray scale image
# the equation for gray image: Y = 0.2125 R + 0.7154 G + 0.0721 B
I_gray=(rgb2gray(I)*255).astype(np.uint8)

# ostu's threhsold segmentation
thresh=threshold_otsu(I_gray)
bw_n0=I_gray<thresh

# remove small binary nosiy regions with the size less than noise_size
bw_n2=opening(bw_n0,square(3))
noise_size=150
bw_n3=remove_small_objects(bw_n2, noise_size, connectivity=8)
bw_n4=remove_small_holes(bw_n3,noise_size/2)


## iteratively erosion algorithm to find markers for watershed
bw_nn=label(bw_n4)
obj_n=np.max(bw_nn)
bw_mark=np.zeros(bw_n4.shape,dtype=bool)
for tt in range(1,obj_n+1):
    bw_temp=np.zeros(bw_n4.shape)
    bw_temp[bw_nn==tt]=1
    thr=np.sum(bw_temp)*0.25
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
plt.pause(2)
plt.close()

##### measure the properties of segmented cell nuclei from the mask labels###
props = regionprops_table(labels,intensity_image=I_gray, properties=['area','centroid','eccentricity','major_axis_length',
                                             'minor_axis_length','orientation','perimeter','solidity',
                                             'mean_intensity'])
df=pd.DataFrame(props)

# visualization for better understanding
plt.imshow(I)
plt.plot(df['centroid-1'],df['centroid-0'],'r*') # plot nuclei centroids on the image
plt.pause(3)
plt.close()

plt.hist(df['area'])   # plot histogram of nuclei size
plt.xlabel('the size of cell nuclei')
plt.ylabel('the number of cell nuclei')
plt.title('histogram of nuclei size')
plt.pause(3)
plt.close()

