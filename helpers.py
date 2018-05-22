# -*- coding: utf-8 -*-
"""
This module contains helpful functions used by the libspec module
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/snieblin/work/libspec')
sys.path.insert(0,'/home/snieblin/work/')
import libspec
import scipy.signal as ssi

def gauss(x, *params):
    '''
    Inputs x values and gaussian parameters
    Outputs y values
    '''
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

def lorentz(x, *params):
    '''
    Inputs x values and gaussian parameters
    Outputs y values
    '''
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        tau = params[i+1]
        wid = params[i+2]
        y = y + tau * wid / (2*np.pi) / ((x-ctr)**2+(wid/2)**2)
    return y
