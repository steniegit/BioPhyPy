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
import matplotlib.pyplot as plt
import scipy.signal as ssi

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
        xs, ys, all_expinfo, all_stepinfo, more_info = [], [], [], [], []
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
                    data_values = np.array(np.frombuffer(decoded, dtype=np.float32))
                    #print(data_values) 
                    x_values.append(data_values)
                for step_y in step.findall('AssayYData'):
                    # Convert string to binary
                    data_text = bytes(step_y.text, 'utf-8')
                    # Convert to base64 
                    decoded = base64.decodebytes(data_text)
                    # And now convert to float32 array
                    data_values = np.array(np.frombuffer(decoded, dtype=np.float32))
                    #print(data_values) 
                    y_values.append(data_values)  
                for step_data in step.findall('CommonData'):
                    step_info.append(etree_to_dict(step_data))
                for tag in ['FlowRate', 'StepType', 'StepName', 'StepStatus', 'ActualTime', 'CycleTime']:
                    for step_data in step.findall(tag):
                        more_dict[tag].append(step_data.text)     
            # Combine dictionaries for each file
            #all_xs.append(np.concatenate(x_values))
            #all_ys.append(np.concatenate(y_values))
            xs.append(x_values)
            ys.append(y_values)
            all_stepinfo.append(combine_dicts(step_info))
            more_info.append(more_dict)
        # Merge all_stepinfo and more_info
        for i in range(len(all_stepinfo)):
            all_stepinfo[i] = {**all_stepinfo[i], **more_info[i]}
        # Fill instance
        self.xs   = xs
        self.ys   = ys
        self.exp_info = all_expinfo
        self.step_info = all_stepinfo
        # Convert text to floats
        self.convert_to_numbers()
        # Get assay_times, write to instance for more direct access
        self.assay_time = self.step_info[0]['ActualTime']
        self.assay_time_cum = np.cumsum(self.assay_time)
        self.no_steps = len(self.assay_time)
        # # Check if x-values for each file are identical and reduce dimension
        # self.concatenate_x()
        return None

    def convert_to_numbers(self):
        '''
        Converts lists in step info into float arrays
        '''
        # List of entries in step info
        entries = ['Concentration', 'MolarConcentration', 'MolecularWeight', 'Temperature', 'StartTime',
                   'AssayTime', 'FlowRate', 'ActualTime', 'CycleTime']
        for entry in entries:
            for sensor in range(len(self.fns)):
                self.step_info[sensor][entry] = np.array(self.step_info[sensor][entry], dtype=float)
        return None

    def sanity_check(self):
        '''
        Checks that length of steps for each sensor matches
        '''
        ### To do 
        return None

    def remove_jumps(self, xshift=5):
        '''
        Remove jumps between steps by subtracting the
        difference between steps
        xshift: Take the nth point (default 5)
        
        '''
        ### To do
        for sensor in range(len(self.fns)):
            for step in range(self.no_steps-1):
                curr_y = self.ys[sensor][step][-1]
                next_y = self.ys[sensor][step+1][0+xshift]
                diff = next_y - curr_y
                # Adjust height of next step
                self.ys[sensor][step+1] -= diff
        return None

    def align(self, step=-1, location='start', median_range=20):
        '''
        Align sensorgrams
        which_step: Number of step
        step: which step to use
        location: 'start' or 'end'
        time_step: nth point will be used for averaging
        '''
        # If step is not defined, choose first one
        if step==-1:
            step = 0
        if location == 'start':
            start = 0
            end = median_range
        elif location == 'end':
            start = - median_range
            end = -1
        else:
            print("Please define location with either 'start' or 'end'")
            return None
        # Go through sensors and define
        for sensor in range(len(self.fns)):
            # Obtain median for selected step and subtract it from whole curve
            bl = np.median(self.ys[sensor][step][start:end])
            # Subtract from all segments
            for segment in self.ys[sensor]:
                segment -= bl
        return None

    def smooth(self, sensors=[], window_length=21, polyorder=2):
        '''
        Smooth data with Savitzky-Golay filter
        window_length: Window size for SG filter, has to be odd number
        polyorder: Polynomial for SG filter (default: 2)
        sensors: which sensors to smoothen
        '''
        if len(sensors) == 0:
            sensors = range(len(self.fns))
        for sensor in sensors:
            for step in range(self.no_steps):
                self.ys[sensor][step] = ssi.savgol_filter(self.ys[sensor][step], window_length, polyorder)
        return None
    
    def subtract(self, ref_sensor=-1, sample_sensors=[]):
        '''
        Subtract ref_sensor from chosen sample_sensors
        ref_sensor: sensor number of reference
        sample_sensors: list of sample sensors
        '''
        # Loop through sample_sensors and subtract reference
        for sensor in sample_sensors:
            for step in range(self.no_steps):
                self.ys[sensor][step] -= self.ys[ref_sensor][step]
        return None
        

    def plot(self, steps=[], sensors=[], show_step_name=True, legend='', legend_step=-1):
        '''
        Plots signal
        steps: which steps to include (list of integers)
               if not specified, all steps are plotted
        sensors: which sensors to include (list of integers)
               if not specified, all steps are plotted
        legend: which entry to use as legend, leave empty string if no
                legend shall be shown
                'SampleID', 'Concentration', 'MolarConcentration'
        legend_step: in case 'SampleID' is chosen, this defines which SampleID
                     is chosen (step number). Default is the first step specified in
                     sensors.
        show_step_name: Show step name in plot
        '''
        # Check steps
        if len(steps) == 0:
            steps = range(len(self.xs[0]))
        if len(sensors) == 0:
            sensors = range(len(self.fns))
        # Define legend_step if set to -1
        if legend_step == -1:
            legend_step = steps[0]
        # Define legend entries
        if legend in ['SampleID', 'Concentration', 'MolarConcentration']:
            legend_entries = [self.step_info[sensor][legend][legend_step] for sensor in range(len(self.fns))]
        else:
            legend_entries = ['']*len(self.fns)
        # Initialize figure
        fig, ax = plt.subplots(1)
        # Get color cycle
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Plot data
        for sensor in sensors:
            # Pick color
            color = cycle[sensor]
            for step in steps:
                if step == steps[0]:
                    ax.plot(self.xs[sensor][step], self.ys[sensor][step], color=color, label=legend_entries[sensor])
                else:
                    ax.plot(self.xs[sensor][step], self.ys[sensor][step], color=color)
                # Plot dashed lines for limits
                ax.axvline(self.assay_time_cum[step], linestyle='--', color='gray', lw=.5)
        # Plot legend
        ax.legend()
        # Show limits of steps and step names
        if show_step_name:
            # Get ylim
            ylim = ax.get_ylim()[1]
            # Add 0 to beginning
            times = np.concatenate(([0], self.assay_time_cum))
            # Step text
            step_text = self.step_info[0]['StepName']
            # Iterate through steps
            for step in steps:
                pos = np.mean((times[step], times[step+1]))
                ax.text(pos, ylim, step_text[step], va='bottom', ha='center')
                
        # Set xlims
        ax.set_xlim([np.min(self.xs[0][steps[0]]), np.max(self.xs[0][steps[-1]])])
        ax.set_xlabel('Time / s')
        ax.set_ylabel('Response / nm')
        return None
            

        
    # def concatenate_signal(self):
    #     '''
    #     This concatenates self.x_values and self.y_values
    #     and stores it in self.conc_x and self.conc_y
    #     '''
    #     # Concatenate x values
    #     x_concat, y_concat = [], []
    #     for i in range(len(self.sep_xs)):
    #         x_concat.append(np.concatenate(self.sep_xs[i]))
    #         y_concat.append(np.concatenate(self.sep_ys[i]))
    #     x_concat = np.array(x_concat)
    #     y_concat = np.array(y_concat)
    #     # Each column should be identical
    #     # Otherwise throw and error
    #     if np.sum(np.diff(x_concat, axis=0)) > np.finfo(float).eps:
    #         print("x values are not identical for files")
    #         print("Stopping")
    #         return None
    #     else:
    #         self.x = x_concat[0]
    #         self.y = y_concat.T
    #     return None  
                          

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
        
