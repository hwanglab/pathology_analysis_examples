'''
marker-controlled watershed example:
see: https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.watershed

author: Hongming Xu
email: mxu@ualberta.ca
'''

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import numpy as np
import matplotlib.pyplot as plt

x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
image = np.logical_or(mask_circle1, mask_circle2)
plt.imshow(image)

distance = ndi.distance_transform_edt(image)
plt.imshow(distance)
local_maxi = peak_local_max(distance, labels=image,footprint=np.ones((3, 3)),indices=False)
plt.imshow(local_maxi)
markers = ndi.label(local_maxi)[0]

labels = watershed(-distance, markers, mask=image)
plt.imshow(labels)
plt.show()