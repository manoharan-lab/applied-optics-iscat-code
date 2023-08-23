#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import numpy as np
from image_processor import process_all_images
from inferencer import infer_beam,infer_particle_w_beam#,get_corrected_phi0
from simulator import iPSF_proc_with_beam

#DEFINE CONSTANTS
k=2*np.pi/(635/1.33)/1.222*1.05 #empirically established

fov=14600

#sig_noise=0.05
sig_noise=0.02
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
with h5py.File('data/apaine3.06.image0000.h5', "r") as f:
    frames = np.array(f['images']).T.astype('float64')
#dark
with h5py.File('data/apaine3.12.image0000.h5', "r") as f:
    dark_frames = np.array(f['images']).T.astype('float64')
dark=np.median(dark_frames,axis=0)

#process
iPSF_datas = process_all_images(frames,dark)
print('image processed')

summaries_beam=[]
summaries=[]
#corrected_phi0s=[]

summary=None

for i in range(0,300):
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
    if i==0:
        x0_mu=2000 ; y0_mu=-4800 ; zpp_mu=1580
    else:
        if summary['mean']['E0'] > 0.01: #discard too dim frames
            x0_mu=summary['mean']['x0']
            y0_mu=summary['mean']['y0']
            zpp_mu=summary['mean']['zpp']
        else:
            print('i=%i is discarded' % i)

    ix0 = (np.abs(xv[0,:] - x0_mu)).argmin() ; iy0 = (np.abs(yv[:,0] - y0_mu)).argmin()
    crop_hw=50
    crop=(max(iy0-crop_hw,0),min(iy0+crop_hw,iPSF_data.shape[0]),max(ix0-crop_hw,0),min(ix0+crop_hw,iPSF_data.shape[1]))

    fixed_phi0=-0.95
    _,trace,summary = infer_particle_w_beam(iPSF_data, xv, yv, Eref, Eref2, beam_spec, k,
                            sig_noise_mu=sig_noise, sig_noise_sig=sig_noise, crop=crop,
                            fixed_phi0=fixed_phi0, zpp_mu=zpp_mu, zpp_sig=150,
                            x0_mu=x0_mu, y0_mu=y0_mu, xy0_sig=150, E0_mu=0.06, E0_sig=0.03,
                            fixed_ma_theta=5.6, fixed_ma_phi=52)

    summaries.append(summary)

#    _,corrected_phi0,corrected_phi0_sd=get_corrected_phi0(trace)
#    corrected_phi0s.append((corrected_phi0,corrected_phi0_sd))


    #OPTIONAL PLOT
    zpp=summary['mean']['zpp']
#    aphi0=summary['mean']['aphi0']
#    sphi0=summary['mean']['sphi0']
#    phi0=summary['mean']['phi0']
    phi0=fixed_phi0
    E0=summary['mean']['E0']
    #ma_theta=summary['mean']['ma_theta']
    ma_theta=5.6
    #ma_phi=summary['mean']['ma_phi']
    ma_phi=52
    x0=summary['mean']['x0']
    y0=summary['mean']['y0']

#    _,phi0,_=get_corrected_phi0(trace)

    # calculate iPSF
    iPSF_fit = iPSF_proc_with_beam(xv,yv,k,x0,y0,zpp,E0,phi0,ma_theta,ma_phi,Eref0,Eref_mu_x,Eref_mu_y,Eref_sig,E2_base)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    maxValP=0.08

    plt.figure()
    plt.imshow(iPSF_data,vmin=-maxValP,vmax=maxValP,cmap='gray',extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
    plt.gca().add_patch(Rectangle((xv[0,crop[2]],yv[crop[0],0]),xv[0,crop[3]]-xv[0,crop[2]],yv[crop[1],0]-yv[crop[0],0],linewidth=1,edgecolor='r',facecolor='none'))
    plt.scatter(x0,y0,s=10)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Original data i=%i' % i)
    plt.savefig('data/plots_diff/orig_i=%i.png' % i)

    plt.figure()
    plt.imshow(iPSF_fit,vmin=-maxValP,vmax=maxValP,cmap='gray',extent=[np.amin(xv),np.amax(xv),np.amin(yv),np.amax(yv)],origin='lower')
    plt.gca().add_patch(Rectangle((xv[0,crop[2]],yv[crop[0],0]),xv[0,crop[3]]-xv[0,crop[2]],yv[crop[1],0]-yv[crop[0],0],linewidth=1,edgecolor='r',facecolor='none'))
    plt.scatter(x0,y0,s=10)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Fit i=%i' % i)
    plt.savefig('data/plots_diff/fit_i=%i.png' % i)

    plt.figure()
    plt.plot(xv[iy0,:],iPSF_data[iy0,:],label='Data')
    plt.plot(xv[iy0,:],iPSF_fit[iy0,:],label='Fit')
    plt.axvline(x=xv[0,crop[2]],linewidth=1, color='r')
    plt.axvline(x=xv[0,crop[3]],linewidth=1, color='r')
    plt.legend()
    plt.title('y=%f, i=%i' % (yv[iy0,0], i))
    plt.savefig('data/plots_diff/cross_i=%i.png' % i)


#SAVE
import pickle

with open("data/summaries_diff_beam.pyob", "wb") as fp:
    pickle.dump(summaries_beam,fp)
    fp.close()

with open("data/summaries_diff.pyob", "wb") as fp:
    pickle.dump(summaries,fp)
    fp.close()

#with open("data/corrected_phi0s.pyob", "wb") as fp:
#    pickle.dump(corrected_phi0s,fp)
#    fp.close()
    
