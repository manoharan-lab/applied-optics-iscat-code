#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module can be used to simulate iPSF images
"""

import numpy as np

def iPSF_proc_with_beam(xv,yv,k,x0,y0,zpp,E0,phi0,ma_theta,ma_phi,
                        Eref0,Eref_mu_x,Eref_mu_y,Eref_sig,E2_base):
    """
    Simulate iPSF image with explicit beam description

    Parameters
    ----------
    xv : ndarray
        x-coordinates
    yv : ndarray
        y-coordinates
    k : float
        wavenumber in vacuum of reference beam in nm^-1
    x0 : float
        x-coordinate of scatterer in nm
    y0 : float
        y-coordinate of scatterer in nm
    zpp : float
        axial position of scatterer in nm
    E0 : float
        scattering amplitude
    phi0 : float
        lumped phase
    ma_theta : float
        misalignment angle theta in degrees
    ma_phi : float
        misalignment angle phi in degrees
    Eref0 : float
        central amplitude of reference beam
    Eref_mu_x : float
        center of reference beam in x-direction
    Eref_mu_y : float
        center of reference beam in y-direction
    Eref_sig
        width (std) of reference beam
    E2_base
        baseline intensity
    

    Returns
    -------
    iPSF : ndarray
        the simulated iPSF image
    """
    Eref = Eref0*np.exp(-((xv-Eref_mu_x)**2+(yv-Eref_mu_y)**2)/Eref_sig**2)
    Eref2 = Eref**2 + E2_base

    Eref_sc = Eref0*np.exp(-(((x0-Eref_mu_x)**2+(y0-Eref_mu_y)**2)/Eref_sig**2))

    ma=k*((xv-x0)*np.cos(ma_phi*np.pi/180)+(yv-y0)*np.sin(ma_phi*np.pi/180))*np.sin(ma_theta*np.pi/180)

    rpp = np.sqrt((xv-x0)**2 + (yv-y0)**2 + zpp**2) #particle to position on focal plane
    cos_theta = zpp / rpp #cos of theta angle
    phi_inc=k*zpp #phase shift due to incedent OPD, zf is lumped into phi0
    phi_sca=k*rpp #phase shift due to return OPD
    fac=np.sqrt(1+cos_theta**2)*1/(k*rpp)
    Escat=E0*fac #NOTE: phase is handled separately to avoid complex numbers

    phi_diff=ma-(phi0+phi_inc+phi_sca)
    iPSF = 2*Eref*Eref_sc*Escat*np.cos(phi_diff) / Eref2

    return iPSF

def iPSF_proc_no_beam(xv,yv,k,x0,y0,zpp,E0,phi0,ma_theta,ma_phi):
    """
    Simulate iPSF image without explicit beam description

    Parameters
    ----------
    xv : ndarray
        x-coordinates
    yv : ndarray
        y-coordinates
    k : float
        wavenumber in vacuum of reference beam in nm^-1
    x0 : float
        x-coordinate of scatterer in nm
    y0 : float
        y-coordinate of scatterer in nm
    zpp : float
        axial position of scatterer in nm
    E0 : float
        scattering amplitude
    phi0 : float
        lumped phase
    ma_theta : float
        misalignment angle theta in degrees
    ma_phi : float
        misalignment angle phi in degrees
    

    Returns
    -------
    iPSF : ndarray
        the simulated iPSF image
    """
    ma=k*((xv-x0)*np.cos(ma_phi*np.pi/180)+(yv-y0)*np.sin(ma_phi*np.pi/180))*np.sin(ma_theta*np.pi/180)

    rpp = np.sqrt((xv-x0)**2 + (yv-y0)**2 + zpp**2) #particle to position on focal plane
    cos_theta = zpp / rpp #cos of theta angle
    phi_inc=k*zpp #phase shift due to incedent OPD, zf is lumped into phi0
    phi_sca=k*rpp #phase shift due to return OPD
    fac=np.sqrt(1+cos_theta**2)*1/(k*rpp)
    Escat=E0*fac #NOTE: phase is handled separately to avoid complex numbers

    phi_diff=ma-(phi0+phi_inc+phi_sca)
    iPSF = 2*Escat*np.cos(phi_diff)

    return iPSF
