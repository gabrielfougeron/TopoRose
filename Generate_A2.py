import os
import shutil

import math as m
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
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

DoVideo = True
# DoVideo = False

# ipoly = 0     # Portrait
ipoly = 1       # Landscape
    
if (ipoly == 0) :           # A4 page
    fig_width_cm = 21.                                
    fig_height_cm = 29.7

else:
    fig_width_cm = 29.7                    
    fig_height_cm = 21.    



inches_per_cm = 1 / 2.54                         # Convert cm to inches
fig_width = fig_width_cm * inches_per_cm         # width in inches
fig_height = fig_height_cm * inches_per_cm       # height in inches
fig_size = [fig_width, fig_height]        

plt.rc('text', usetex=False) # so that LaTeX is not needed when creating a PDF with PdfPages later on

# Transfer to A2
nx_pages = 2
ny_pages = 2

model_width = fig_width_cm * nx_pages / 100
model_height = fig_height_cm * ny_pages / 100


with rio.open(img_filename) as img_open:

    bou_outline = gpd.read_file(model_bou_filename)
    bou_outline_match = bou_outline.to_crs(img_open.crs)

    npoly = bou_outline_match.geometry.shape[0]

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

    # print(zoom_nx,zoom_ny)
    # print(img.shape)

    # print('dtype = ',img.dtype)

    minval = img.min()
    maxval = img.max()

    # minval = 2400
    # maxval = 3000
    # 
    minval = 2360
    maxval = 2960

    # 
    # print('minval = ',minval)
    # print('maxval = ',maxval)

    xmin,xmax,ymin,ymax = rio.plot.plotting_extent(img_open)
    
    # print('xmin = ',xmin)
    # print('xmax = ',xmax)
    # print('ymin = ',ymin)
    # print('ymax = ',ymax)

    x = np.linspace(zoom_xmin, zoom_xmax, zoom_nx)
    y = np.linspace(zoom_ymin, zoom_ymax, zoom_ny)
    X, Y = np.meshgrid(x, y)

#     print(zoom_nx)
#     print(x.shape)
# 
#     exit()
#     
    n_levels = 100 +1
    levels =  np.linspace(minval, maxval, n_levels)

    real_height = zoom_ymax-zoom_ymin
    real_width = zoom_xmax-zoom_xmin
    real_thick = maxval-minval



    print("real height = ",real_height)
    print("real width = ",real_width)
    print("real thickness = ",real_thick)

    scale = 0.5 * (real_height/model_height + real_width/model_width)

    print('scale = ',scale)

    model_thick = real_thick / scale

    print('model thickness (mm) = ', 1000 * model_thick)

    linewidths = 0.5


    fig = plt.figure()
    fig.set_size_inches(fig_size)
    ax = plt.axes()

    ax.pcolor(X, Y, img, vmin = minval, vmax = maxval)

    CS = ax.contour(X, Y, img, colors = 'k', levels = levels, linewidths = linewidths)
    # ax.clabel(CS, inline=True, fontsize=5)

    ax.axis('equal')
    ax.axis('off')
    fig.tight_layout(pad=0)

    for ext in img_ext_list:
        img_filename = os.path.join(img_output_folder,'contour'+ext)
        plt.savefig(img_filename)

    plt.close()


    pdf_filename = os.path.join(img_output_folder,'contour_multipage.pdf')
    with PdfPages(pdf_filename) as pdf:

        for zoom_ix in range(nx_pages):
            for zoom_jy in range(ny_pages):

                fig = plt.figure()
                fig.set_size_inches(fig_size)
                ax = plt.axes()
                
                di = zoom_nx // nx_pages
                dj = zoom_ny // ny_pages

                ip_min =  zoom_ix      * di
                ip_max = (zoom_ix + 1) * di + 1
                jp_min =  zoom_jy      * dj
                jp_max = (zoom_jy + 1) * dj + 1


                X_zoom = X[jp_min:jp_max,ip_min:ip_max]
                Y_zoom = Y[jp_min:jp_max,ip_min:ip_max]
                img_zoom = img[jp_min:jp_max,ip_min:ip_max]

                poly_color = '#6699cc'
                # ax.add_patch(PolygonPatch(bou_outline_match.geometry[ipoly], fc=poly_color, ec=poly_color, alpha=0.5, zorder=2 ))

                ax.pcolor(X_zoom, Y_zoom, img_zoom, vmin = minval, vmax = maxval)

                CS = ax.contour(X_zoom, Y_zoom, img_zoom, colors = 'k', levels = levels, linewidths = linewidths)
                # ax.clabel(CS, inline=True, fontsize=5)


                ax.axis('equal')
                ax.axis('off')
                fig.tight_layout(pad=0)

                pdf.savefig(fig) 
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