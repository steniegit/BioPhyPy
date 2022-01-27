'''
Class for circular dichroism data
'''

from .helpers import *
import pandas as pd
import numpy as np

class CD_data():
    '''
    This is a class to read in Chirascan CD data 
    that has been exported as csv
    '''
    def __init__(self, fn=''):
        '''
        Initialize ms data
        fn: File name
        '''
        self.fn = fn
        #self.path = path
        # Load data
        self.read_CD()
        self.average()
        return None

    def read_CD(self):
        '''
        Reads comma separated csv from Chirascan export
        
        Required: 
        fn: File name
        
        Returns: spectrum as ndarray
        '''
        # List with entries
        spectrum = []
        with open(self.fn, 'r') as f:
            for line in f:
                if line[0].isdigit():
                    spectrum.append(line.strip().split(','))
                    # Convert to array
        cd = np.array(spectrum, dtype=float)
        self.x = cd[:,0]
        self.y = cd[:,1:]
        return cd

    def average(self):
        '''
        Averages y values
        '''
        self.av_y = np.average(self.y, axis=1)
        self.std_y = np.std(self.y, axis=1)
        return None

    def savetxt(self):
        # Generate first part of header
        header = '%18s' % 'Wavelength/nm'
        for i in range(self.y.shape[1]):
            header += '%21s' % ('Repeat %i' % (i+1))
        if hasattr(self, 'av_y'):
            dat = np.hstack((self.x.reshape(-1,1), self.y, self.av_y.reshape(-1,1)))
            header += '%21s' % 'Average'
        else:
            dat = np.hstack((self.x.reshape(-1,1), self.y))
        np.savetxt(self.fn.replace('.csv', '') + '.txt', dat, fmt='%20.10f', header=header)
        print("Text file saved as %s" % (self.fn.replace('.csv', '') + '.txt'))
    

