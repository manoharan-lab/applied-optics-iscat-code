#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements the different Bayesian inference schemes using PyMC
"""

import numpy as np
import pymc as pm
import arviz as az
from IPython.display import display

def infer_beam(data_beam,xv,yv,sig_noise_beam,downsample=1, samples=500, cores=1,
               init='jitter+adapt_diag_grad'):
    """
    Infer the parameters describing the reference beam

    Parameters
    ----------
    data_beam : ndarray
        processed image
    xv : ndarray
        x-coordinates
    yv : ndarray
        y-coordinates
    sig_noise_beam : float
        noise level for the likelihood function
    downsample : float (optional)
        the rate of downsampling (default 1 for no downsampling)
    samples : int (optional)
        the number of posterior samples to take (default 500)
    cores : int (optional)
        the number of parralel cores to employ (default 1 for serial execution)
    init : string (optional)
        the initializer for the PyMC sampler (default 'jitter+adapt_diag_grad'))

    Returns
    -------
    beam_model : object
        the PyMC model for the beam
    idata : object
        the PyMC inference data
    summary : object
        the PyMC inference summary
    """
    #DOWNSAMPLE
    data_beam=data_beam[::downsample,::downsample]
    xv=xv[::downsample,::downsample] ; yv=yv[::downsample,::downsample]

    beam_model = pm.Model()
    with beam_model:

        # Priors for unknown model parameters
        Eref0=pm.Gamma("Eref0", mu=150, sigma=50) #central amplitude of reference field
        Eref_mu_x=pm.Normal("Eref_mu_x", mu=0, sigma=2000)
        Eref_mu_y=pm.Normal("Eref_mu_y", mu=0, sigma=2000)
        Eref_sig=pm.Gamma("Eref_sig", mu=6000, sigma=2000)

        E2_base=pm.Normal("E2_base", mu=4000, sigma=2000) #baseline intensity

        Eref = Eref0*pm.math.exp(-((xv-Eref_mu_x)**2+(yv-Eref_mu_y)**2)/Eref_sig**2)

        beam = Eref**2 + E2_base

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=beam, sigma=sig_noise_beam, observed=data_beam)

    with beam_model:
        # draw 500 posterior samples
        idata = pm.sample(samples, cores=cores, init=init, return_inferencedata=True)

    with beam_model:
        summary=az.summary(idata)
        display(summary)

    return beam_model, idata, summary


def infer_particle_w_beam(iPSF_data,xv,yv,Eref,Eref2,beam_spec,k_0,k_rel_var=None,
                          sig_noise_mu=0.05,sig_noise_sig=0.05,fixed_sig_noise=None,
                          crop=(0,-1,0,-1),zpp_mu=300,zpp_sig=300,E0_mu=0.3,E0_sig=0.2,
                          x0_mu=0,y0_mu=0,xy0_sig=1000,fixed_phi0=None,fixed_ma_theta=None,
                          fixed_ma_phi=None, samples=500, cores=1, init='jitter+adapt_diag_grad'):
    """
    Infer the parameters of the scatterer with explicit beam description

    Parameters
    ----------
    iPSF_data : ndarray
        processed image
    xv : ndarray
        x-coordinates
    yv : ndarray
        y-coordinates
    Eref : float
        central amplitude of reference beam
    Eref2 : float
        central intensity of reference beam including baseline
    beam_spec : tuple of floats
        (inferred) specs of the beam: 0 Eref0, 1 Eref_mu_x, 2 Eref_my_y, 3 Eref_sig
    k_0 : float
        wavenumber in vacuum of reference beam in nm^-1
    k_rel_var : float (optional)
        if set, the std of relative correction factor to k_0
    sig_noise_mu : float (optional)
        mean of noise level (default 0.05)
    sig_noise_sig : float (optional)
        std of noise level (default 0.05)
    fixed_sig_noise : float (optional)
        if set, the fixed noise level
    crop : tuple of ints (optional)
        the region to crop the image to (default is no cropping)
    zpp_mu : float (optional)
        mean of the axial position of scatterer in nm (default 300)
    zpp_sig : float (optional)
        std of the axial position of scatterer in nm (default 300)
    E0_mu : float (optional)
        mean of scattering amplitude (default 0.3)
    E0_sig : float (optional)
        std of scattering amplitude (default 0.2)
    x0_mu : float (optional)
        mean of x-coordinate of scatterer in nm (default 0)
    y0_mu : float (optional)
        mean of y-coordinate of scatterer in nm (default 0)
    xy0_sig : float (optional)
        std of x- and y-coordinate of scatterer in nm (default 1000)
    fixed_phi0 : float (optional)
        if set, the fixed lumped phase, otherwise, uniform prior from 0 to 2pi
    fixed_ma_theta : float (optional)
        if set, the fixed misalignment angle theta in degrees
    fixed_ma_phi : float (optional)
        if set, the fixed misalignment angle phi in degrees
    samples : int (optional)
        the number of posterior samples to take (default 500)
    cores : int (optional)
        the number of parralel cores to employ (default 1 for serial execution)
    init : string (optional)
        the initializer for the PyMC sampler (default 'jitter+adapt_diag_grad'))

    Returns
    -------
    iPSF_model : object
        the PyMC model for the iPSF
    idata : object
        the PyMC inference data
    summary : object
        the PyMC inference summary
    """
    Eref0=beam_spec[0] ; Eref_mu_x=beam_spec[1] ; Eref_mu_y=beam_spec[2] ; Eref_sig=beam_spec[3]

    #CROP
    iPSF_data=iPSF_data[crop[0]:crop[1],crop[2]:crop[3]]
    xv=xv[crop[0]:crop[1],crop[2]:crop[3]] ; yv=yv[crop[0]:crop[1],crop[2]:crop[3]]
    Eref=Eref[crop[0]:crop[1],crop[2]:crop[3]] ; Eref2=Eref2[crop[0]:crop[1],crop[2]:crop[3]]

    #DEFINE MODEL
    iPSF_model = pm.Model()
    with iPSF_model:
        # Priors

        zpp = pm.Gamma("zpp", mu=zpp_mu, sigma=zpp_sig)
        #zpp = pm.Gamma("$z_p'$ [nm]", mu=zpp_mu, sigma=zpp_sig)
        if fixed_phi0 is None:
            aphi0 = pm.Uniform("aphi0", lower=0, upper=np.pi)
            sphi0 = pm.Bernoulli("sphi0", p=0.5)
            phi0=(2*sphi0-1)*aphi0 #sign phi0
            phi0=pm.Deterministic("phi0", phi0)
            #phi0=pm.Deterministic("$\phi_0$", phi0)
        else:
            phi0=fixed_phi0
        E0 = pm.Gamma("E0", mu=E0_mu, sigma=E0_sig)
        #E0 = pm.Gamma("$\hat{E}_0$", mu=E0_mu, sigma=E0_sig)

        if fixed_ma_theta is None:
            ma_theta=pm.TruncatedNormal("ma_theta", mu=5, sigma=3, lower=0.0, upper=15)
            #ma_theta=pm.TruncatedNormal("$\\theta_b$ [$^\circ$]", mu=5, sigma=3, lower=0.0, upper=15)
        else:
            ma_theta=fixed_ma_theta
        if fixed_ma_phi is None:
            ma_phi=pm.TruncatedNormal("ma_phi", mu=45, sigma=20, lower=0, upper=90)
            #ma_phi=pm.TruncatedNormal("$\\varphi_b$ [$^\circ$]", mu=45, sigma=20, lower=0, upper=90)
        else:
            ma_phi=fixed_ma_phi

        x0 = pm.Normal('x0', mu=x0_mu, sigma=xy0_sig)
        #x0 = pm.Normal('$x_0$ [nm]', mu=x0_mu, sigma=xy0_sig)
        y0 = pm.Normal('y0', mu=y0_mu, sigma=xy0_sig)
        #y0 = pm.Normal('$y_0$ [nm]', mu=y0_mu, sigma=xy0_sig)

        if k_rel_var is None:
            k=k_0
        else:
            kfac = pm.Normal("kfac", mu=1, sigma=k_rel_var)
            k=k_0*kfac

        if fixed_sig_noise is None:
            sig_noise = pm.Gamma('sig_noise', mu=sig_noise_mu, sigma=sig_noise_sig)
            #sig_noise = pm.Gamma('$\sigma$', mu=sig_noise_mu, sigma=sig_noise_sig)
        else:
            sig_noise=fixed_sig_noise

        # calculate iPSF
        if fixed_ma_theta is None or fixed_ma_phi is None:
            ma=k*((xv-x0)*pm.math.cos(ma_phi*np.pi/180)+(yv-y0)*pm.math.sin(ma_phi*np.pi/180))*pm.math.sin(ma_theta*np.pi/180)
        else:
            ma=k*((xv-x0)*np.cos(ma_phi*np.pi/180)+(yv-y0)*np.sin(ma_phi*np.pi/180))*np.sin(ma_theta*np.pi/180)

        Eref_sc = Eref0*pm.math.exp(-(((x0-Eref_mu_x)**2+(y0-Eref_mu_y)**2)/Eref_sig**2))

        rpp = pm.math.sqrt((xv-x0)**2 + (yv-y0)**2 + zpp**2) #from particle to focal plane
        cos_theta = zpp / rpp #cos of scattering angle
        phi_inc=k*zpp #phase shift due to incedent OPD, zf is lumped into phi0
        phi_sca=k*rpp #phase shift due to return OPD
        fac=pm.math.sqrt(1+cos_theta**2)*1/(k*rpp) #amplitude factor
        Escat=E0*fac #scattering amplitude

        phi_diff=ma-(phi0+phi_inc+phi_sca)
        iPSF = 2*Eref*Eref_sc*Escat*pm.math.cos(phi_diff) / Eref2

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=iPSF, sigma=sig_noise, observed=iPSF_data)

    with iPSF_model:
        if fixed_phi0 is None:
            # draw 500 posterior samples, feel free to increase the number of employed cores
            idata = pm.sample(samples, cores=cores, init=init, return_inferencedata=True)
        else:
            # SMC sampling robust to multimodal posterior, feel free to increase the number of chains
            # Parallel computing does not seem to work with this SMC implementation (bug)
            idata = pm.sample_smc(samples, cores=cores, parallel=True)
            #trace = az.from_pymc3(trace_pymc3)
    with iPSF_model:
        summary = az.summary(idata, round_to=10)
        display(summary)

    return iPSF_model, idata, summary

def infer_particle_no_beam(iPSF_data,xv,yv,k_0,k_rel_var=None,crop=(0,-1,0,-1),
                           sig_noise_mu=0.001,sig_noise_sig=0.001,fixed_sig_noise=None,
                           zpp_mu=300,zpp_sig=300,E0_mu=0.3,E0_sig=0.2,x0_mu=0,y0_mu=0,
                           xy0_sig=1000,fixed_phi0=None,fixed_ma_theta=None,fixed_ma_phi=None,
                           fixed_zpp=None,samples=500, cores=1, init='jitter+adapt_diag_grad'):
    """
    Infer the parameters of the scatterer without explicit beam description

    Parameters
    ----------
    iPSF_data : ndarray
        processed image
    xv : ndarray
        x-coordinates
    yv : ndarray
        y-coordinates
    k_0 : float
        wavenumber in vacuum of reference beam in nm^-1
    k_rel_var : float (optional)
        if set, the std of relative correction factor to k_0
    crop : tuple of ints (optional)
        the region to crop the image to (default is no cropping)
    sig_noise_mu : float (optional)
        mean of noise level (default 0.001)
    sig_noise_sig : float (optional)
        std of noise level (default 0.001)
    fixed_sig_noise : float (optional)
        if set, the fixed noise level
    zpp_mu : float (optional)
        mean of the axial position of scatterer in nm (default 300)
    zpp_sig : float (optional)
        std of the axial position of scatterer in nm (default 300)
    E0_mu : float (optional)
        mean of scattering amplitude (default 0.3)
    E0_sig : float (optional)
        std of scattering amplitude (default 0.2)
    x0_mu : float (optional)
        mean of x-coordinate of scatterer in nm (default 0)
    y0_mu : float (optional)
        mean of y-coordinate of scatterer in nm (default 0)
    xy0_sig : float (optional)
        std of x- and y-coordinate of scatterer in nm (default 1000)
    fixed_phi0 : float (optional)
        if set, the fixed lumped phase, otherwise, uniform prior from 0 to 2pi
    fixed_ma_theta : float (optional)
        if set, the fixed misalignment angle theta in degrees
    fixed_ma_phi : float (optional)
        if set, the fixed misalignment angle phi in degrees
    fixed_zpp : float (optional)
        if set, the fixed axial position of scatterer in nm
    samples : int (optional)
        the number of posterior samples to take (default 500)
    cores : int (optional)
        the number of parralel cores to employ (default 1 for serial execution)
    init : string (optional)
        the initializer for the PyMC sampler (default 'jitter+adapt_diag_grad'))

    Returns
    -------
    iPSF_model : object
        the PyMC model for the iPSF
    idata : object
        the PyMC inference data
    summary : object
        the PyMC inference summary
    """
    #CROP
    iPSF_data=iPSF_data[crop[0]:crop[1],crop[2]:crop[3]]
    xv=xv[crop[0]:crop[1],crop[2]:crop[3]] ; yv=yv[crop[0]:crop[1],crop[2]:crop[3]]

    #DEFINE MODEL
    iPSF_model = pm.Model()
    with iPSF_model:
        # Priors
        if fixed_zpp is None:
            zpp = pm.Gamma("zpp", mu=zpp_mu, sigma=zpp_sig)
        else:
            zpp = fixed_zpp
        if fixed_phi0 is None:
            aphi0 = pm.Uniform("aphi0", lower=0, upper=np.pi)
            sphi0 = pm.Bernoulli("sphi0", p=0.5)
            phi0=(2*sphi0-1)*aphi0 #sign phi0
            phi0=pm.Deterministic("phi0", phi0)
        else:
            phi0=fixed_phi0
        E0 = pm.Gamma("E0", mu=E0_mu, sigma=E0_sig)

        if fixed_ma_theta is None:
            ma_theta=pm.TruncatedNormal("ma_theta", mu=5, sigma=3, lower=0.0, upper=15)
        else:
            ma_theta=fixed_ma_theta
        if fixed_ma_phi is None:
            ma_phi=pm.TruncatedNormal("ma_phi", mu=45, sigma=20, lower=0, upper=90)
        else:
            ma_phi=fixed_ma_phi

        x0 = pm.Normal('x0', mu=x0_mu, sigma=xy0_sig)
        y0 = pm.Normal('y0', mu=y0_mu, sigma=xy0_sig)

        if fixed_sig_noise is None:
            sig_noise = pm.Gamma('sig_noise', mu=sig_noise_mu, sigma=sig_noise_sig)
        else:
            sig_noise=fixed_sig_noise

        if k_rel_var is None:
            k=k_0
        else:
            kfac = pm.Normal("kfac", mu=1, sigma=k_rel_var)
            k=k_0*kfac

        # calculate iPSF
        if fixed_ma_theta is None or fixed_ma_phi is None:
            ma=k*((xv-x0)*pm.math.cos(ma_phi*np.pi/180)+(yv-y0)*pm.math.sin(ma_phi*np.pi/180))*pm.math.sin(ma_theta*np.pi/180)
        else:
            ma=k*((xv-x0)*np.cos(ma_phi*np.pi/180)+(yv-y0)*np.sin(ma_phi*np.pi/180))*np.sin(ma_theta*np.pi/180)

        rpp = pm.math.sqrt((xv-x0)**2 + (yv-y0)**2 + zpp**2) #from particle to focal plane
        cos_theta = zpp / rpp #cos of scattering angle
        phi_inc=k*zpp #phase shift due to incedent OPD, zf is lumped into phi0
        phi_sca=k*rpp #phase shift due to return OPD
        fac=pm.math.sqrt(1+cos_theta**2)*1/(k*rpp) #amplitude factor
        Escat=E0*fac #scattering amplitude

        phi_diff=ma-(phi0+phi_inc+phi_sca)
        iPSF = 2*Escat*pm.math.cos(phi_diff)

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=iPSF, sigma=sig_noise, observed=iPSF_data)

    with iPSF_model:
        if fixed_phi0 is None:
            # draw 500 posterior samples, feel free to increase the number of employed cores
            idata = pm.sample(samples, cores=cores, init=init, return_inferencedata=True)
        else:
            # SMC sampling robust to multimodal posterior, feel free to increase the number of chains
            # Parallel computing does not seem to work with this SMC implementation (bug)
            idata = pm.sample_smc(samples, cores=cores, parallel=True)
            #trace = az.from_pymc3(trace_pymc3)
    with iPSF_model:
        summary = az.summary(idata, round_to=10)
        display(summary)

    return iPSF_model, idata, summary


def get_corrected_phi0(trace):
    if np.mean(trace['posterior']['aphi0'])>np.pi/2:
        phi0s=trace['posterior']['phi0']

        cphi0s = np.mod(phi0s,2*np.pi)

        mphi0 = np.mean( cphi0s )
        mphi0 = np.mod(mphi0+np.pi,2*np.pi)-np.pi
    else:
        cphi0s = trace['posterior']['phi0']
        mphi0 = np.mean(cphi0s)

    return cphi0s,float(mphi0),float(np.std(cphi0s))

def get_std_dzf_plus_zpp(trace,k):
    cphi0s,_,_=get_corrected_phi0(trace)

    dzfs=0.5*cphi0s/k
    zpps=trace['posterior']['zpp']

    dzfs_plus_zpps = dzfs + zpps

    return float(np.std(dzfs_plus_zpps))
