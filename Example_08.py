'''
purpurse: be familiar with image edge operators

author: Hongming Xu
email: mxu@ualberta.ca
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage import filters, feature


I=plt.imread('TCGA-3M-AB46-01Z-00-DX1.jpg')

# color image to gray scale image
# the equation for gray image: Y = 0.2125 R + 0.7154 G + 0.0721 B
I_gray=(rgb2gray(I)*255).astype(np.uint8)

edge_roberts = filters.roberts(I_gray)
edge_sobel = filters.sobel(I_gray)
edge_prewitt = filters.prewitt(I_gray)
edge_canny1 = feature.canny(I_gray,sigma=3)
edge_canny2 = feature.canny(I_gray,sigma=5)


ax1=plt.subplot(231)
plt.imshow(I_gray,cmap='gray')
ax1.title.set_text('Gray Image')
ax2=plt.subplot(232)
plt.imshow(edge_roberts,cmap='gray')
ax2.title.set_text('Roberts Operator')

ax3=plt.subplot(233)
plt.imshow(edge_sobel,cmap='gray')
ax3.title.set_text('Sobel Operator')
ax4=plt.subplot(234)
plt.imshow(edge_prewitt,cmap='gray')
ax4.title.set_text('Prewitt Operator')

ax5=plt.subplot(235)
plt.imshow(edge_canny1,cmap='gray')
ax5.title.set_text('Canny sigma=3')

ax6=plt.subplot(236)
plt.imshow(edge_canny2,cmap='gray')
ax6.title.set_text('Canny sigma=5')

plt.show()
plt.pause(2)
plt.close()

