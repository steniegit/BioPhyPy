'''
BLI/SPR/GCI class
'''

import os, glob
import numpy as np
import pandas as pd
import base64
import scipy.signal as ssi
import xml.etree.ElementTree as ET
from scipy.optimize import curve_fit
from .helpers import etree_to_dict, combine_dicts

# To do


class BLI_data:
    '''
    Class to load and fit BLI/SPR/GCI data
    '''

    def __init__(self, folder=''):
        '''
        This initialises the instance
        and loads the frd files in the folder
        folder : folder containing the frd files
        '''
        self.folder = folder
        fns = glob.glob(folder + '/*.frd')
        if len(fns) < 1:
            print("No frd files found in folder %s" % self.folder)
            print("Exit!")
            return None
        else:
            print("Found %i frd files" % len(fns))
            print(fns)
            self.fns = fns
        # Initialize dictionaries with data
        all_xs, all_ys, sep_xs, sep_ys, all_expinfo, all_stepinfo, more_info = [], [], [], [], [], [], []
        for fn in fns:
            # Load file
            tree = ET.parse(fn)
            root = tree.getroot()

            # Extract experimental info 
            all_expinfo.append(etree_to_dict(root.find('ExperimentInfo')))

            # Initialize lists for each file
            x_values, y_values, step_info = [], [], []
            more_dict = {'FlowRate': [], 'StepType': [], 'StepName':[], 'StepStatus':[], 'ActualTime':[], 'CycleTime':[]}
            for step in root.find('KineticsData'):
                for step_x in step.findall('AssayXData'):
                    # Convert string to binary
                    data_text = bytes(step_x.text, 'utf-8')
                    # Convert to base64 
                    decoded = base64.decodebytes(data_text)
                    # And now convert to float32 array
                    data_values = np.frombuffer(decoded, dtype=np.float32)
                    #print(data_values) 
                    x_values.append(data_values)
                for step_y in step.findall('AssayYData'):
                    # Convert string to binary
                    data_text = bytes(step_y.text, 'utf-8')
                    # Convert to base64 
                    decoded = base64.decodebytes(data_text)
                    # And now convert to float32 array
                    data_values = np.frombuffer(decoded, dtype=np.float32)
                    #print(data_values) 
                    y_values.append(data_values)  
                for step_data in step.findall('CommonData'):
                    step_info.append(etree_to_dict(step_data))
                for tag in ['FlowRate', 'StepType', 'StepName', 'StepStatus', 'ActualTime', 'CycleTime']:
                    for step_data in step.findall(tag):
                        more_dict[tag].append(step_data.text)     
            # Combine dictionaries for each file
            all_xs.append(np.concatenate(x_values))
            all_ys.append(np.concatenate(y_values))
            sep_xs.append(x_values)
            sep_ys.append(y_values)
            all_stepinfo.append(combine_dicts(step_info))
            more_info.append(more_dict)
        # Merge all_stepinfo and more_info
        for i in range(len(all_stepinfo)):
            all_stepinfo[i] = {**all_stepinfo[i], **more_info[i]}
        # Fill instance
        self.sep_xs   = sep_xs
        self.sep_ys   = sep_ys
        self.exp_info = all_expinfo
        self.step_info = all_stepinfo
        # Concatenate signal
        self.concatenate_signal()
        return None

    def concatenate_signal(self):
        '''
        This concatenates self.x_values and self.y_values
        and stores it in self.conc_x and self.conc_y
        '''
        # Concatenate x values
        x_concat, y_concat = [], []
        for i in range(len(self.sep_xs)):
            x_concat.append(np.concatenate(self.sep_xs[i]))
            y_concat.append(np.concatenate(self.sep_ys[i]))
        x_concat = np.array(x_concat)
        y_concat = np.array(y_concat)
        # Each column should be identical
        # Otherwise throw and error
        if np.sum(np.diff(x_concat, axis=0)) > np.finfo(float).eps:
            print("x values are not identical for files")
            print("Stopping")
            return None
        else:
            self.x = x_concat[0]
            self.y = y_concat.T
        return None

    def plot_signal(self):
        '''
        Plots signal
        '''
        # Initialize figure
        fig, ax = plt.subplots(1)
        # Plot data
        ax.plot(self.x, self.y)
        
        

# class BLI_data:
#     '''
#     Class to load and fit BLI/SPR/GCI data
#     '''

#     def __init__(self, fn=''):
#         '''
#         fn: filename of raw data file (csv file)
#         '''
#         self.fn = fn
#         # Obtain folder
#         self.folder = '/'.join(self.fn.split('/')[:-1]) + '/'
#         # Check if file exists
#         if not (os.path.isfile(self.fn)):
#             print("File %s not found! Exiting" % self.fn)
#             return None
#         # Load data
#         #data = np.genfromtxt(self.fn, skip_header=2)
#         dat = pd.read_csv(self.fn, skiprows=3)
#         # Just for debugging, remove later
#         self.dat = dat
#         # Replace X-values with actual unit
#         dat = dat.rename(columns={dat.columns[0]: 'Time / s'})
#         # Get sensor names (index) and column names
#         self.sensor_names = list(dat.iloc[0,1:-1].str.strip()) #.string.replace(' ','')
#         self.sensor_files = list(dat.iloc[1,1:-1].index.str.strip())
#         # Get original file name
#         self.orig_fn = dat.columns[0]
#         # Get times and response
#         self.responses = dat.iloc[1:,1:-1].astype(float) #.dropna(how='all', axis=1)
#         # Load times
#         self.responses.index = dat.iloc[1:,0].astype(float)
#         self.responses.name = 'Time / s'
#         print("Imported %i columns from %s" % (len(self.responses.columns), self.fn))
#         self.read_metadata()
#         return None

#     def read_metadata(self):
#         '''
#         Reads in metadata from frd files and stores it in self.metadata
#         '''
#         # Check if frd files are present
#         for fn in self.sensor_files:
#             # Check if file exists
#             if not (os.path.isfile(self.fn)):
#                 print("File %s not found! Exiting" % self.fn)
#                 return None         
#         # List of metadata entries
#         metadata_list = ['StepName', 'ActualTime', 'SampleID']
#         # Create pandas dataframe
#         self.metadata = pd.DataFrame([self.sensor_names, self.sensor_files])
#         metadata = {}
#         for metadata_entry in metadata_list:
#             metadata_list = []
#             temp_sensors = []
#             for fn in self.sensor_files:
#                 with open(self.folder + fn,'r') as f:
#                     temp_line = []
#                     for line in f:
#                         if metadata_entry in line:
#                             temp = re.search('<%s>(.+?)</%s>' % (metadata_entry, metadata_entry), line).group(1)
#                             if metadata_entry=='ActualTime':
#                                 temp_line.append(float(temp))
#                             else:
#                                 temp_line.append(temp)
#                 temp_sensors.append(temp_line)
#             # Only SampleID differs for different sensors
#             if metadata_entry == 'SampleID':
#                 metadata[metadata_entry] = temp_sensors
#             else:
#                 metadata[metadata_entry] = temp_sensors[0]
#         self.metadata = metadata
#         self.step_limits = np.cumsum(np.array(self.metadata['ActualTime']))
#         return None
        
#     def plot_raw(self):
#         '''
#         Simple plotting function
#         '''
#         ax = self.responses.plot()
#         ax.set_ylabel('Response / nm')
#         for x in self.step_limits:
#             ax.axvline(x, linestyle='--', color='gray', zorder=-20, alpha=.5)
#         return ax
        
