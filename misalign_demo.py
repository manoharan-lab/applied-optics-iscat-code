#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from simulator import iPSF_proc_no_beam

#DEFINE CONSTANTS
k=2*np.pi/(635/1.5)/1.222 #empirically established

fov=14600

#DEFINE MESH
def getMesh(xmin,xmax,ymin,ymax,nx=200,ny=200):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)

xv,yv=getMesh(-0.5,0.5,-0.5,0.5)
xv=xv*fov ; yv=yv*fov

#calculate iPSF
iPSF = iPSF_proc_no_beam(xv, yv, k, 0, 0, 300, 0.1, 0, 10, 45)

fig, ax = plt.subplots()
im=plt.imshow(iPSF,cmap='gray',extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
plt.xlabel('$x$ [nm]')
plt.ylabel('$y$ [nm]')

ax.margins(x=0)
plt.subplots_adjust(bottom=0.25)

axslidTheta = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
sliderTheta = Slider(
    ax=axslidTheta,
    label='$\\theta$',
    valmin=0,
    valmax=90,
    valinit=10,
)

axslidPhi = plt.axes([0.15, 0.12, 0.65, 0.03], facecolor='lightgoldenrodyellow')
sliderPhi = Slider(
    ax=axslidPhi,
    label='$\\phi$',
    valmin=0,
    valmax=360,
    valinit=45,
)

def update(v):
    ma_theta=sliderTheta.val
    ma_phi=sliderPhi.val
    iPSF=iPSF_proc_no_beam(xv, yv, k, 0, 0, 300, 0.1, 0, ma_theta, ma_phi)

    im.set_data(iPSF)

    fig.canvas.draw_idle()

sliderTheta.on_changed(update)
sliderPhi.on_changed(update)
