#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module processes the raw images according to J. Ortega Arroyo, D. Cole, 
and P. Kukura, “Interferometric scattering microscopy and its combination with
single-molecule fluorescence imaging,” Nat. Protoc. 11, 617–633 (2016)
"""

import numpy as np
from scipy.ndimage import gaussian_filter

def process_all_images(raws, dark, subt_median=True, sig1=6, sig2=15):
    """
    Process a full stack of raw images.

    Parameters
    ----------
    raws : ndarray
        images to be processed
    dark : ndarray
        dark counts to be subtracted
    subt_median : bool
        boolean to indicate whether a median subtraction should be performed
    sig1 : float
        std of the gaussian filter for reference beam subtraction
    sig2 : float
        std of the gaussian filter for reference beam division

    Returns
    -------
    stack : ndarray
        processed images
    """
    stack = np.array([
        process_single_image(raws[i,:,:],dark,sig1=sig1,sig2=sig2) for i in range(raws.shape[0])])

    if subt_median:
        noise = np.median(stack,axis=0)
        stack = stack - noise[np.newaxis,:,:]

    return stack

def process_single_image(raw, dark, sig1=6, sig2=15):
    """
    Process a single raw images.

    Parameters
    ----------
    raw : ndarray
        image to be processed
    dark : ndarray
        dark counts to be subtracted
    sig1 : float
        std of the gaussian filter for reference beam subtraction
    sig2 : float
        std of the gaussian filter for reference beam division

    Returns
    -------
    divided : ndarray
        processed image
    """
    #subtract the dark count
    raw = raw - dark

    #subtract the reference beem
    subtracted = raw - gaussian_filter(raw, sigma=sig1)

    #divide the reference beam
    divided = subtracted / gaussian_filter(raw, sigma=sig2)

    #return subtracted
    return divided
