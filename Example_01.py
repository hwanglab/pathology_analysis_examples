'''
purpose: learn how to read .svs pathology slides into python

author: Hongming Xu
if you found any bugs to run the code, please email to:
email: mxu@ualberta.ca
'''
# to run this code, you need to install below python packages
import openslide
import matplotlib.pyplot as plt

# you need to download the example wsi slide from tcga website.
wsi='../../TCGA-BL-A3JM-01Z-00-DX1.33E53972-CEA4-4D84-A5D2-7DAD7B0C27F8.svs'

# open slide image
Slide = openslide.OpenSlide(wsi)

# get slide pixel resolutions at the highest magnification
xr = float(Slide.properties['openslide.mpp-x'])  # pixel resolution at x direction
yr = float(Slide.properties['openslide.mpp-y'])  # pixel resolution at y direction
print('This slide pixel resolutions are (%f, %f) micro-meters\n' % (xr,yr))

# get slide dimensions at the highest magnification
dim=Slide.dimensions
print('This slide dimensions are (%d,%d) pixels\n' % (dim[0],dim[1]))

# from pixel resolution and the number of pxiels, you can calculate the slide physical size
print('Do you know what the physical size of this slide is??')
print('------------------------')
print('the answer is (%f,%f) millimeters\n' % (xr*dim[0]/1000,yr*dim[1]/1000))

# slide has multi-levels, get the number of levels
lc=Slide.level_count
# get width, height of image at the lowest resolution
lrHeight = Slide.level_dimensions[lc-1][1]
lrWidth = Slide.level_dimensions[lc-1][0]

# read in whole slide at the lowest magnification
wsi_low = Slide.read_region((0, 0), lc-1, (lrWidth, lrHeight))

# show the slide
plt.figure()
plt.imshow(wsi_low)
plt.title('The WSI at the lowest magnification!!!')
plt.show()
