#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import numpy as np
from image_processor import process_all_images
from inferencer import infer_particle_no_beam,get_corrected_phi0


#LOAD DATA
file='data/agoldfain2.08.image0000.h5'

# Get raw data from file
with h5py.File(file, "r") as f:
    frames = np.array(f['images'][:,:,0:1000]).T.astype('float64')

dark=np.zeros((frames.shape[1],frames.shape[2]))

iPSFs = process_all_images(frames, dark, subt_median=False, sig1=1.5)
iPSFs = iPSFs - np.median(iPSFs[-100:,:,:],axis=0)[np.newaxis,:,:]

print('image processed')

#DEFINE CONSTANTS
k=2*np.pi/(405/1.5)/1.40 #empirically found

sig_noise=0.01 #choose between fixed or variable when the fit function is called

fov=14600*380/200

#DEFINE MESH
def getMesh(xmin,xmax,ymin,ymax,nx=200,ny=200):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)

xv,yv=getMesh(-0.5,0.5,-0.5,0.5,nx=380,ny=380)
xv=xv*fov ; yv=yv*fov

#FIT iPSFs

summaries=[]
corrected_phi0s=[]

summary=None

for i in range(0,250):
#for i in range(0,5+1):

    print('fitting iPSF #%i' % i)

    iPSF_data = iPSFs[i,:,:]

    crop=(252,269,348,365)
    x0_mu = xv[0,int((crop[2]+crop[3])/2)] ; y0_mu = yv[int((crop[0]+crop[1])/2),0]
    #zpp_mu=130, zpp_sig=50
    _,trace,summary = infer_particle_no_beam(iPSF_data, xv, yv, k, crop=crop,
                            sig_noise_mu=sig_noise, sig_noise_sig=sig_noise,
                            fixed_zpp=130, x0_mu=x0_mu, y0_mu=y0_mu, xy0_sig=150, #use fixed_zpp or a variable zpp
                            E0_mu=0.03, E0_sig=0.01, fixed_ma_theta=5.6, fixed_ma_phi=52)

    summaries.append(summary)

    _,corrected_phi0,corrected_phi0_sd=get_corrected_phi0(trace)
    corrected_phi0s.append((corrected_phi0,corrected_phi0_sd))

#SAVE
import pickle

with open("data/summaries_lambda.pyob", "wb") as fp:
    pickle.dump(summaries,fp)
    fp.close()

with open("data/corrected_phi0s_lambda.pyob", "wb") as fp:
    pickle.dump(corrected_phi0s,fp)
    fp.close()
