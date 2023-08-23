#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import holopy as hp

#READ DATA
folder='data/apaine3.19.'

imgs=[]
cs=[]
for i in range(0,10):
    img = np.array(Image.open(folder+'image' + str(i).zfill(4) + '.tif'))
    imgs.append(img)

imgs=np.array(imgs)

#SIMPLE MEDIAN SUBTRACTION
imgs=imgs-np.median(imgs,axis=0)

#CENTER FINDING
cs=[]
for i in range(imgs.shape[0]):
    c=hp.core.process.center_find(hp.core.metadata.data_grid(imgs[i,:,:]), centers=1)
    cs.append(c)

cs=np.array(cs)

#VOLTAGES
vs=np.array([0.03, 3.75, 7.48, 11.20, 14.93, 18.67, 22.40, 26.13, 29.85, 33.59])

#FIT
p=np.polyfit(vs, cs[:,1], 1)
print("dpix/dV = %f" % p[0])

dnm_dV = 20e3/75 #nanometer per volt from Thorlabs spec
dnm_dpix = dnm_dV/p[0] #nanometer per pixel
fov=dnm_dpix*200 #total field of view

print("dnm/dpix = %f" % dnm_dpix)
print("fov = %f" % fov)

#PLOT
plt.figure()
plt.plot(vs,cs[:,1],'o-')
plt.plot(vs,p[1]+p[0]*vs,'r--')
plt.grid()
plt.xlabel('Piezo Voltage')
plt.ylabel('x pixel center')

i=0

fig, ax = plt.subplots()
im=plt.imshow(imgs[i,:,:],cmap='gray',origin='lower')
sc=plt.scatter(cs[i,1],cs[i,0],s=10)
plt.xlabel('$x$')
plt.ylabel('$y$')

ax.margins(x=0)
plt.subplots_adjust(bottom=0.25)

axslid = plt.axes([0.15, 0.08, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(
    ax=axslid,
    label='i',
    valmin=0,
    valmax=imgs.shape[0]-1,
    valinit=i,
    valstep=1
)

def update(i):
    im.set_data(imgs[i,:,:])
    sc.set_offsets(np.flip(cs[i,:]))

    fig.canvas.draw_idle()

slider.on_changed(update)
