#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from image_processor import process_all_images

#DEFINE CONSTANTS
fov=14600/1.2

#DEFINE MESH
def getMesh(xmin,xmax,ymin,ymax,nx=200,ny=200):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)

xv,yv=getMesh(-0.5,0.5,-0.5,0.5)
xv=xv*fov ; yv=yv*fov

#LOAD DATA
#raw data
with h5py.File('data/apaine2.11.image0000.h5', "r") as f:
    frames = np.array(f['images']).T.astype('float64')
#dark
with h5py.File('data/apaine2.17.image0000.h5', "r") as f:
    dark_frames = np.array(f['images']).T.astype('float64')
dark=np.median(dark_frames,axis=0)

i=1770

#PROCESS
raw = frames[i,:,:]

data_beam = raw

plt.figure()
plt.imshow(raw,cmap='gray',extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Raw')

plt.figure()
plt.imshow(5*dark,cmap='gray',vmin=np.min(raw),vmax=np.max(raw),extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Dark (5x)')

#subtract the dark count
raw = raw - dark

plt.figure()
plt.imshow(raw,cmap='gray',vmin=np.min(raw),vmax=np.max(raw),extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Img')

from scipy.ndimage import gaussian_filter

#subtract the reference beem
gb1 = gaussian_filter(raw, sigma=6)
subt = raw - gb1

plt.figure()
plt.imshow(gb1,cmap='gray',vmin=np.min(raw),vmax=np.max(raw),extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('blur(Img,$\\sigma=6$)')

plt.figure()
plt.imshow(subt,cmap='gray',vmin=-0.5*np.max(np.abs(subt)),vmax=0.5*np.max(np.abs(subt)),extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Subt')

gb2 = gaussian_filter(raw, sigma=15)
iPSF = subt / gb2

plt.figure()
plt.imshow(gb2,cmap='gray',vmin=np.min(raw),vmax=np.max(raw),extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('blur(Img,$\\sigma=15$)')

maxValP=0.3

plt.figure()
plt.imshow(iPSF,cmap='gray',vmin=-maxValP,vmax=maxValP,extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('iPSF')

iPSFs = process_all_images(frames,dark,subt_median=False)
noise = np.median(iPSFs,axis=0)

iPSF_data = iPSF - noise

plt.figure()
plt.imshow(noise,cmap='gray',vmin=-maxValP,vmax=maxValP,extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('median noise')

plt.figure()
plt.imshow(iPSF_data,cmap='gray',vmin=-maxValP,vmax=maxValP,extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('iPSF corrected')

crop=(60,160,60,160)

plt.figure()
plt.imshow(iPSF_data,vmin=-maxValP,vmax=maxValP,cmap='gray',extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.gca().add_patch(Rectangle((xv[0,crop[0]],yv[crop[2],0]),xv[0,crop[1]]-xv[0,crop[0]],yv[crop[3],0]-yv[crop[2],0],linewidth=1,edgecolor='r',facecolor='none'))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('iPSF corrected')
    