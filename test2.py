import os
import shutil

import math as m
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from descartes_patch import PolygonPatch

import pandas as pd
import geopandas as gpd

import rasterio as rio
import rasterio.plot
import rasterio.mask
import fiona
import shapely
import rasterstats




img_ext_list = ['.png','.pdf']

base_folder = os.path.dirname(__file__) 

img_output_folder = os.path.join(base_folder,'output')



geo_data_folder = os.path.join(base_folder,'geo_data')
img_filename = os.path.join(geo_data_folder,'DEM/AP_02191_FBS_F0770_RT1/AP_02191_FBS_F0770_RT1.dem.tif')

model_bou_filename = os.path.join(geo_data_folder,'shp/model_bounds.shp')

# DoVideo = True
DoVideo = False

ipoly = 0
ipoly = 1
    
Portrait = (ipoly == 0)

with rio.open(img_filename) as img_open:

    bou_outline = gpd.read_file(model_bou_filename)
    bou_outline_match = bou_outline.to_crs(img_open.crs)

    npoly = bou_outline_match.geometry.shape[0]

    # assert npoly == 1

    zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax = np.array(bou_outline_match.geometry.bounds)[ipoly]

    # print(zoom_xmin, zoom_ymin, zoom_xmax, zoom_ymax)
    
    img_tot = img_open.read()[0,...] # Elevation values in meters

    nx = img_tot.shape[1]
    ny = img_tot.shape[0]

    xmin,xmax,ymin,ymax = rio.plot.plotting_extent(img_open)

    zoom_imin = m.floor(nx * (zoom_xmin - xmin) / (xmax - xmin))
    zoom_imax = m.ceil(nx * (zoom_xmax - xmin) / (xmax - xmin))
    zoom_jmin = m.floor(ny * (zoom_ymin - ymin) / (ymax - ymin))
    zoom_jmax = m.ceil(ny * (zoom_ymax - ymin) / (ymax - ymin))

    zoom_nx = zoom_imax - zoom_imin
    zoom_ny = zoom_jmax - zoom_jmin

    # print(xmin,xmax,ymin,ymax)
    # print(zoom_jmin,zoom_jmax,zoom_imin,zoom_imax)

    img = np.flip(img_tot,axis=0)[zoom_jmin:zoom_jmax,zoom_imin:zoom_imax]

    # print('dtype = ',img.dtype)

    minval = img.min()
    maxval = img.max()

    # minval = 2400
    # maxval = 3000
    # 
    minval = 2360
    maxval = 2960

    
    print('minval = ',minval)
    print('maxval = ',maxval)

    xmin,xmax,ymin,ymax = rio.plot.plotting_extent(img_open)
    
    # print('xmin = ',xmin)
    # print('xmax = ',xmax)
    # print('ymin = ',ymin)
    # print('ymax = ',ymax)

    x = np.linspace(zoom_xmin, zoom_xmax, zoom_nx)
    y = np.linspace(zoom_ymin, zoom_ymax, zoom_ny)
    X, Y = np.meshgrid(x, y)
    
    n_levels = 100 +1
    levels =  np.linspace(minval, maxval, n_levels)

    if Portrait:                                    # A4 page
        fig_width_cm = 21                                
        fig_height_cm = 29.7
    else:
        fig_width_cm = 29.7                    
        fig_height_cm = 21            

    inches_per_cm = 1 / 2.54                         # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm         # width in inches
    fig_height = fig_height_cm * inches_per_cm       # height in inches
    fig_size = [fig_width, fig_height]

    plt.rc('text', usetex=False) # so that LaTeX is not needed when creating a PDF with PdfPages later on
    fig = plt.figure()
    fig.set_size_inches(fig_size)
    
    linewidths = 0.5

    print("dx = ",zoom_xmax-zoom_xmin)
    print("dy = ",zoom_ymax-zoom_ymin)
    print("dz = ",maxval-minval)


    ax = plt.axes()
    ax.axis('equal')
    ax.axis('off')

    poly_color = '#6699cc'
    # ax.add_patch(PolygonPatch(bou_outline_match.geometry[ipoly], fc=poly_color, ec=poly_color, alpha=0.5, zorder=2 ))

    ax.pcolor(X, Y, img, vmin = minval, vmax = maxval)

    CS = ax.contour(X, Y, img, colors = 'k', levels = levels, linewidths = linewidths)
    # ax.clabel(CS, inline=True, fontsize=5)

    fig.tight_layout(pad=0)

    for ext in img_ext_list:
        img_filename = os.path.join(img_output_folder,'contour'+ext)
        plt.savefig(img_filename)

    plt.close()



    if DoVideo:

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(12, 12))
        ax.plot_wireframe(X, Y, img, rstride=5, cstride=5)

#         max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), img.max()-img.min()]).max() / 2.0
# 
#         mid_x = (X.max()+X.min()) * 0.5
#         mid_y = (Y.max()+Y.min()) * 0.5
#         mid_z = (img.max()+img.min()) * 0.5
#         ax.set_xlim(mid_x - max_range, mid_x + max_range)
#         ax.set_ylim(mid_y - max_range, mid_y + max_range)
#         ax.set_zlim(mid_z - max_range, mid_z + max_range)


        # ax.set_box_aspect([1,1,1])
        ax.axis('off')

        def init():
            return fig,

        def animate(i):
            ax.view_init(elev=10., azim=i)
            # ax.view_init(azim=i)
            return fig,

        # Animate
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=360, interval=20, blit=True)
        # Save
        video_filename = os.path.join(img_output_folder,'3d.mp4')
        anim.save(video_filename, fps=30, extra_args=['-vcodec', 'libx264'])