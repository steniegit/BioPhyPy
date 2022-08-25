'''
BLI/SPR/GCI class
'''

import os
import numpy as np
import pandas as pd
import scipy.signal as ssi
from scipy.optimize import curve_fit
from .helpers import *

# To do

class BLI_data:
    '''
    Class to load and fit BLI/SPR/GCI data
    '''

    def __init__(self, fn=''):
        '''
        fn: filename of raw data file (csv file)
        '''
        self.folder = folder
        # Check if file exists
        if not (os.path.isfile(self.fn)):
            raise Exception("File %s not found! Exiting" % self.fn)
        return None
        # Load data
        #data = np.genfromtxt(self.fn, skip_header=2)
        dat = pd.read_csv(self.fn, skiprows=3)
        # Get sensor names (index) and column names
        self.sensors = dat.iloc[0,1:-1] #.string.replace(' ','')
        # Get original file name
        self.orig_fn = dat.columns[0]
        # Get times and response
        self.responses = dat.iloc[1:,1:-1].astype(float) #.dropna(how='all', axis=1)
        # Replace filename with unit
        dat.columns[0] = 'Time / s'
        # Load times
        self.responses.index = dat.iloc[1:,0].astype(float)
        self.responses.name = 'Time / s'
        print("Imported %i columns from %s" % (len(self.responses.columns), self.fn))
        return None
        
        
    def plot_raw():
        '''
        Simple plotting function
        '''
        ax = self.responses.plot()
        return ax
        
