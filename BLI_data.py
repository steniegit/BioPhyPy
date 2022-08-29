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
        self.fn = fn
        # Obtain folder
        self.folder = '/'.join(self.fn.split('/')[:-1]) + '/'
        # Check if file exists
        if not (os.path.isfile(self.fn)):
            print("File %s not found! Exiting" % self.fn)
            return None
        # Load data
        #data = np.genfromtxt(self.fn, skip_header=2)
        dat = pd.read_csv(self.fn, skiprows=3)
        # Just for debugging, remove later
        self.dat = dat
        # Replace X-values with actual unit
        dat = dat.rename(columns={dat.columns[0]: 'Time / s'})
        # Get sensor names (index) and column names
        self.sensor_names = list(dat.iloc[0,1:-1].str.strip()) #.string.replace(' ','')
        self.sensor_files = list(dat.iloc[1,1:-1].index.str.strip())
        # Get original file name
        self.orig_fn = dat.columns[0]
        # Get times and response
        self.responses = dat.iloc[1:,1:-1].astype(float) #.dropna(how='all', axis=1)
        # Load times
        self.responses.index = dat.iloc[1:,0].astype(float)
        self.responses.name = 'Time / s'
        print("Imported %i columns from %s" % (len(self.responses.columns), self.fn))
        self.read_metadata()
        return None

    def read_metadata(self):
        '''
        Reads in metadata from frd files and stores it in self.metadata
        '''
        # Check if frd files are present
        for fn in self.sensor_files:
            # Check if file exists
            if not (os.path.isfile(self.fn)):
                print("File %s not found! Exiting" % self.fn)
                return None         
        # List of metadata entries
        metadata_list = ['StepName', 'ActualTime', 'SampleID']
        # Create pandas dataframe
        self.metadata = pd.DataFrame([self.sensor_names, self.sensor_files])
        metadata = {}
        for metadata_entry in metadata_list:
            metadata_list = []
            temp_sensors = []
            for fn in self.sensor_files:
                with open(self.folder + fn,'r') as f:
                    temp_line = []
                    for line in f:
                        if metadata_entry in line:
                            temp = re.search('<%s>(.+?)</%s>' % (metadata_entry, metadata_entry), line).group(1)
                            if metadata_entry=='ActualTime':
                                temp_line.append(float(temp))
                            else:
                                temp_line.append(temp)
                temp_sensors.append(temp_line)
            # Only SampleID differs for different sensors
            if metadata_entry == 'SampleID':
                metadata[metadata_entry] = temp_sensors
            else:
                metadata[metadata_entry] = temp_sensors[0]
        self.metadata = metadata
        self.step_limits = np.cumsum(np.array(self.metadata['ActualTime']))
        return None
        
    def plot_raw(self):
        '''
        Simple plotting function
        '''
        ax = self.responses.plot()
        ax.set_ylabel('Response / nm')
        for x in self.step_limits:
            ax.axvline(x, linestyle='--', color='gray', zorder=-20, alpha=.5)
        return ax
        
