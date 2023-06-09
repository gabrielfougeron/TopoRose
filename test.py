import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import geopandas as gpd

import rasterio as rio
import rasterio.plot

import fiona

import shapely
import rasterstats

import shutil


base_folder = os.path.dirname(__file__) 
img_filename = os.path.join(base_folder,'geo_data/DEM/AP_02191_FBS_F0770_RT1/AP_02191_FBS_F0770_RT1.dem.tif')

with rio.open(img_filename) as img_open:

    img = img_open.read()[0,...]
    #     

    # print(img.shape)

    nx = img.shape[0]
    ny = img.shape[1]
    print('nx = ',nx)
    print('ny = ',ny)

    print('dtype = ',img.dtype)

    minval = img.min()
    maxval = img.max()
    
    print('minval = ',minval)
    print('maxval = ',maxval)

    xmin,xmax,ymin,ymax = rio.plot.plotting_extent(img_open)
    
    print('xmin = ',xmin)
    print('xmax = ',xmax)
    print('ymin = ',ymin)
    print('ymax = ',ymax)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    Y, X = np.meshgrid(y, x)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, img)
    ax.plot_wireframe(X, Y, img)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()