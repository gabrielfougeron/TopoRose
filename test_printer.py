import os
import shutil

import math as m
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages


img_ext_list = ['.pdf']

base_folder = os.path.dirname(__file__) 
img_output_folder = os.path.join(base_folder,'output')

fig_width_cm = 21.                                
fig_height_cm = 29.7
  

inches_per_cm = 1 / 2.54                         # Convert cm to inches
fig_width = fig_width_cm * inches_per_cm         # width in inches
fig_height = fig_height_cm * inches_per_cm       # height in inches
fig_size = [fig_width, fig_height]        

plt.rc('text', usetex=False) # so that LaTeX is not needed when creating a PDF with PdfPages later on




fig = plt.figure()
fig.set_size_inches(fig_size)
ax = plt.axes()


xmin = 0.
xmax = 210.
ymin = 0.
ymax = 297.

lw = 2


styles_list = [
    'solid',
    'dotted',
    'dashed',
    'dashdot',
]   

c = 'k'

n_lines = len(styles_list)

ds = 2.

for i in range(n_lines):

    linestyle = styles_list[i]

    x = xmin + i * ds
    plt.plot((x,x), (ymin,ymax), c = c, linestyle = linestyle)
    x = xmax - i * ds
    plt.plot((x,x), (ymin,ymax), c = c, linestyle = linestyle)

    y = ymin + i * ds
    plt.plot((xmin,xmax), (y,y), c = c, linestyle = linestyle)
    y = ymax - i * ds
    plt.plot((xmin,xmax), (y,y), c = c, linestyle = linestyle)






ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

ax.axis('off')
fig.tight_layout(pad=0)

for ext in img_ext_list:
    img_filename = os.path.join(img_output_folder,'test_printer'+ext)
    plt.savefig(img_filename)

plt.close()

