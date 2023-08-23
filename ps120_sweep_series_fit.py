#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import numpy as np
from image_processor import process_all_images
from inferencer import infer_beam,infer_particle_w_beam,get_corrected_phi0,get_std_dzf_plus_zpp

#DEFINE CONSTANTS
k=2*np.pi/(635/1.5)/1.222 #empirically established

fov=14600

sig_noise=0.05
sig_noise_beam = 2000

#DEFINE MESH
def getMesh(xmin,xmax,ymin,ymax,nx=200,ny=200):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)

xv,yv=getMesh(-0.5,0.5,-0.5,0.5)
xv=xv*fov ; yv=yv*fov

#LOAD and PROCESS DATA
#raw data
with h5py.File('data/apaine2.11.image0000.h5', "r") as f:
    frames = np.array(f['images']).T.astype('float64')
#dark
with h5py.File('data/apaine2.17.image0000.h5', "r") as f:
    dark_frames = np.array(f['images']).T.astype('float64')
dark=np.median(dark_frames,axis=0)

#process
iPSF_datas = process_all_images(frames,dark)
print('images processed')

summaries_beam=[]
summaries=[]
corrected_phi0s=[]
std_dzf_plus_zpps=[]

for i in range(1600,1600+207):
    iPSF_data = iPSF_datas[i,:,:]
    data_beam = frames[i,:,:] - dark

    #FIT BEAM
    _,_,summary_beam = infer_beam(data_beam, xv, yv, sig_noise_beam, downsample=2)

    summaries_beam.append(summary_beam)

    Eref0=summary_beam['mean']['Eref0']
    Eref_mu_x=summary_beam['mean']['Eref_mu_x']
    Eref_mu_y=summary_beam['mean']['Eref_mu_y']
    Eref_sig=summary_beam['mean']['Eref_sig']
    E2_base=summary_beam['mean']['E2_base']

    Eref = Eref0*np.exp(-((xv-Eref_mu_x)**2+(yv-Eref_mu_y)**2)/Eref_sig**2)
    Eref2 = Eref**2 + E2_base
    beam_spec = (Eref0,Eref_mu_x,Eref_mu_y,Eref_sig)

    #FIT iPSF
    crop=(60,160,60,160)
    _,trace,summary = infer_particle_w_beam(iPSF_data, xv, yv, Eref, Eref2, beam_spec, k,
                            sig_noise_mu=sig_noise, sig_noise_sig=sig_noise, crop=crop,
                            zpp_mu=max((1814-i)*10,300), x0_mu=570, y0_mu=810, xy0_sig=120)

    summaries.append(summary)

    _,corrected_phi0,corrected_phi0_sd=get_corrected_phi0(trace)
    corrected_phi0s.append((corrected_phi0,corrected_phi0_sd))
    std_dzf_plus_zpp=get_std_dzf_plus_zpp(trace,k)
    std_dzf_plus_zpps.append(std_dzf_plus_zpp)

#SAVE
import pickle

with open("data/summaries_beam.pyob", "wb") as fp:
    pickle.dump(summaries_beam,fp)
    fp.close()

with open("data/summaries.pyob", "wb") as fp:
    pickle.dump(summaries,fp)
    fp.close()

with open("data/corrected_phi0s.pyob", "wb") as fp:
    pickle.dump(corrected_phi0s,fp)
    fp.close()

with open("data/std_dzf_plus_zpps.pyob", "wb") as fp:
    pickle.dump(std_dzf_plus_zpps,fp)
    fp.close()
    