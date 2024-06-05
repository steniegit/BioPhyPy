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
from .helpers import *

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
            # Sort them
            fns.sort()
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
        self.no_sensors = len(self.fns)
        # Convert concentrations to readable numbers
        self.convert_conc()
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
                # Do sanity check
                try:
                    self.step_info[sensor][entry] = np.array(self.step_info[sensor][entry], dtype=float)
                except:
                    print("Erroneous entry found for %s and sensor %i: %s" % (entry, sensor, self.step_info[sensor][entry]))
                    print("Will set it to -1. Needs to be corrected")
                    # Correct erroneous value
                    for i in range(len(self.step_info[sensor][entry])):
                        try:
                            float(self.step_info[sensor][entry][i])
                        except:
                            self.step_info[sensor][entry][i] = -1
                    self.step_info[sensor][entry] = np.array(self.step_info[sensor][entry], dtype=float)
        return None

    def convert_conc(self):
        '''
        Converts concentrations to mg/ml and M
        '''
        for sensor in range(self.no_sensors):
            # Convert weight concentration
            if self.step_info[sensor]['ConcentrationUnits'][0] == 'µg/ml':
                fact = 1E-3
            elif self.step_info[sensor]['ConcentrationUnits'][0] == 'mg/ml':
                fact = 1
            else:
                print("Could not convert the following concentration unit: %s" % self.step_info[sensor]['ConcentrationUnits'])
                return None
            # Convert weight concentration
            self.step_info[sensor]['Concentration_mg/ml'] = fact * self.step_info[sensor]['Concentration']
            # Now convert molar conc            
            if self.step_info[sensor]['MolarConcUnits'][0] == 'nM':
                fact = 1E-9
            elif self.step_info[sensor]['MolarConcUnits'][0] == 'µM':
                fact = 1E-6
            elif self.step_info[sensor]['MolarConcUnits'][0] == 'mM':
                fact = 1E-3
            else:
                print("Could not convert the following molar concentration unit: %s" % self.step_info[sensor]['MolarConcUnits'])
                return None
            # Convert weight concentration
            self.step_info[sensor]['MolarConcentration_M'] = fact * self.step_info[sensor]['MolarConcentration']
        return None

    def sanity_check(self):
        '''
        Checks that length of steps for each sensor matches
        '''
        ### To do 
        return None

    def remove_ends(self, xremove=15):
        '''
        Artifacts at the beginning and end of each step
        can cause trouble with the fits. With this function the first and last 15 points
        of each step are removed
        '''
        # If xremove is 0 do not do anything
        if xremove == 0:
            return None
        # Otherwise remove points
        for sensor in range(len(self.fns)):
            for step in range(self.no_steps):
                #print("Before cutting: %i points" % len(self.xs[sensor][step]))
                self.xs[sensor][step] = self.xs[sensor][step][xremove:-xremove]
                self.ys[sensor][step] = self.ys[sensor][step][xremove:-xremove]
                #print("After cutting: %i points" % len(self.xs[sensor][step]))
        return None

    def remove_jumps(self, xshift=5):
        '''
        Remove jumps between steps by subtracting the
        difference between steps
        xshift: Take the nth point (default 5)
        '''
        # Catch exception
        if xshift==0:
            return None
        # Otherwise remove jumps
        for sensor in range(len(self.fns)):
            for step in range(self.no_steps-1):
                curr_y = np.mean(self.ys[sensor][step][-xshift:])
                next_y = np.mean(self.ys[sensor][step+1][0:xshift])
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

    def normalize(self, step=-1, location='start', median_range=20):
        '''
        Normalize sensorgrams, e.g. regarding to loading levels
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
        norm_values = []
        for sensor in range(len(self.fns)):
            # Obtain median for selected step and subtract it from whole curve
            bl = np.median(self.ys[sensor][step][start:end])
            norm_values.append(bl)
            # Divide for all segments
            for segment in self.ys[sensor]:
                segment /= bl
        self.norm_values = norm_values
        return norm_values
    
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
        
    def extract_values(self, sensors=[], step=0, time=0):
        '''
        Extract values for a certain step at time (of that step)
        sensors: which sensors to include (list of integers)
                 if not specified, all steps are plotted
        steps: which step to use
        time: Time in s, if too large, last point will be used
        '''
        # Check steps
        if len(sensors) == 0:
            sensors = range(len(self.fns))
        # Go through sensors and extract values
        xs, ys = [], []
        for sensor in sensors:
            time_array = self.xs[sensor][step] - self.xs[sensor][step][0]
            pos = np.argmin(np.abs(time_array - time))
            xs.append(self.xs[sensor][step][pos])
            ys.append(self.ys[sensor][step][pos])
        return (xs, ys)

    def merge_steps(self, steps=[0,1], new_step_name='Merged'):
        '''
        Merges two subsequent steps into one
        '''
        # Sanity check
        if np.abs(steps[0] - steps[1]) !=1:
            print("Steps have to be consecutive! Exit")
            return None
        # Go through all sensors
        sensors = range(len(self.fns))
        for sensor in sensors:
            joined_xs = np.concatenate((self.xs[sensor][steps[0]], self.xs[sensor][steps[0]]))
            joined_ys = np.concatenate((self.ys[sensor][steps[0]], self.ys[sensor][steps[0]]))
            # Replace
            self.xs[sensor][steps[0]] = joined_xs
            self.ys[sensor][steps[0]] = joined_ys
            # Remove the following one
            del(self.xs[sensor][steps[1]])
            del(self.ys[sensor][steps[1]])
            # Adjust step info
            self.step_info[sensor]['AssayTime'][steps[0]] = self.step_info[sensor]['AssayTime'][steps[0]] + self.step_info[sensor]['AssayTime'][steps[1]]
            self.step_info[sensor]['ActualTime'][steps[0]] = self.step_info[sensor]['AssayTime'][steps[0]] + self.step_info[sensor]['AssayTime'][steps[1]]
            for key in self.step_info[sensor].keys():
                if type(self.step_info[sensor][key]) == list:
                    del(self.step_info[sensor][key][steps[1]])
                elif type(self.step_info[sensor][key]) == np.ndarray:
                    self.step_info[sensor][key] = np.delete(self.step_info[sensor][key], steps[1])
            self.step_info[sensor]['StepName'][steps[0]]= new_step_name
        return None
        
    def plot(self, steps=[], sensors=[], ylim=[], show_step_name=True, legend='', legend_step=-1, ax=None, linestyle='-', alpha=1, abbrev_step_names=False):
        '''
        Plots signal
        steps: which steps to include (list of integers)
               if not specified, all steps are plotted
        sensors: which sensors to include (list of integers)
               if not specified, all steps are plotted
        ylim: y-limits for plot
        legend: which entry to use as legend, leave empty string if no
                legend shall be shown
                'SampleID', 'Concentration', 'MolarConcentration', 'SensorID',
                'Concentration_M', 'MolarConcentration_M'
        legend_step: in case 'SampleID' is chosen, this defines which SampleID
                     is chosen (step number). Default is the first step specified in
                     sensors.
        linestyle: solid line '-', dotted line ':', points 'o' (matplotlib format)
        alpha: alpha value for plot
        show_step_name: Show step name in plot
        abbrev_step_names: Abbreviate step names in plot
        '''
        # Check steps
        if len(steps) == 0:
            steps = range(len(self.xs[0]))
        if len(sensors) == 0:
            sensors = range(len(self.fns))
        # Define legend_step if set to -1
        if legend_step == -1:
            legend_step = steps[0]
        # Define legend entries, for concentrations add units
        if legend in ['Concentration', 'MolarConcentration']:
            legend_entries = [str(self.step_info[sensor][legend][legend_step]) + '$\,$' + self.step_info[sensor][legend.replace('MolarConcentration','MolarConc') + 'Units'][legend_step] for sensor in range(len(self.fns))]
        elif legend == 'Concentration_mg/ml':
            legend_entries = [('%.0e$\,$mg/ml' % self.step_info[sensor][legend][legend_step])  for sensor in range(len(self.fns))]
        elif legend == 'MolarConcentration_M':
            legend_entries = [('%.0e$\,$M' % self.step_info[sensor][legend][legend_step])  for sensor in range(len(self.fns))]
        else:
            # Check that entry actually exists
            if legend in self.step_info[0].keys():
                legend_entries = [('%s' % self.step_info[sensor][legend][legend_step])  for sensor in range(len(self.fns))]
            # Otherwise just print out indices
            else:
                print("Did not find entry %s for legend. Will just use indices" % legend)
                legend_entries = list(map(str, list(range(len(self.fns)))))
        # Initialize figure
        if ax==None:
            fig, ax = plt.subplots(1)
        else:
            fig = ax.figure
        # Get color cycle
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Plot data
        for sensor in sensors:
            # Pick color
            color = cycle[sensor]
            for step in steps:
                if step == steps[0]:
                    ax.plot(self.xs[sensor][step], self.ys[sensor][step], linestyle, color=color, label=legend_entries[sensor], alpha=alpha)
                else:
                    ax.plot(self.xs[sensor][step], self.ys[sensor][step], linestyle, color=color, alpha=alpha)
                # Plot dashed lines for limits
                ax.axvline(self.assay_time_cum[step], linestyle='--', color='gray', lw=.5)
        # Plot legend
        if len(legend_entries) > 0:
            ax.legend()
        # Adjust ylim if desired
        if len(ylim) ==2 :
            ax.set_ylim(ylim)
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
                if abbrev_step_names:
                      text = step_text[step].replace('Baseline','BL').replace('Loading','Load.').replace('Association','Assoc.').replace('Dissociation','Dissoc.')
                else:
                      text = step_text[step]
                ax.text(pos, ylim, text, va='bottom', ha='center')
        # Set xlims
        ax.set_xlim([np.min(self.xs[0][steps[0]]), np.max(self.xs[0][steps[-1]])])
        ax.set_xlabel('Time / s')
        ax.set_ylabel('Response / nm')
        return fig, ax

    def export_txt(self, sensors=[], steps=[], prefix='export', delimiter='\t', precision='%10.3e'):
        '''
        Export csv of sensors to the parental data folder
        sensors: List of sensors (starts with 0!), if empty all are used
        steps:   List of steps (starts with 0!), if empty all are used
        prefix:  Prefix for file name
        delimiter: Delimiter for txt
        precision: Number format for export, default %10.3e
        '''
        if len(sensors) < 1:
            sensors = range(self.no_sensors)
        if len(steps) < 1 :
            steps = range(self.no_steps)
        # Sanity check for steps, have to be consecutive
        if np.any(np.diff(steps) != 1):
            print("Steps have to be consecutive!")
            print("Cannot do export")
            return None
        # Loop through sensors and export
        for sensor in sensors:
            # Create new list for xs and ys
            xs = [self.xs[sensor][step] for step in steps]
            ys = [self.ys[sensor][step] for step in steps]
            # Now concetanate
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            # Create one array with x and y
            xys = np.stack((xs,ys), axis=1)
            # Save cvs
            np.savetxt(self.folder + '/%s_sensor%i_step%i-%i.txt' % (prefix, sensor, steps[0], steps[1]), xys,
                       delimiter=delimiter, fmt=precision)
            print("Exported file to %s/%s_sensor%i_step%i-%i.txt" % (self.folder, prefix, sensor, steps[0], steps[1]))
        return None
            
            


    def fit_data(self, sensors=[0], step_assoc=3, step_dissoc=4, func='biexp', plot=True, order='a', norm=True):
        '''
        Fit rise and decay for data
        sensors: List of sensors
        step_assoc: Association step
        step_dissoc: Dissociation step
        func: 'biexp' oder 'monoexp'
        plot: Plot result
        '''
        # Get color cycle
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Create instance with fit results if not already there
        if not hasattr(self, 'fit_results'):
                self.fit_results = {}
        for sensor in sensors:
            # Select data
            assoc, assoc_time = self.ys[sensor][step_assoc], self.xs[sensor][step_assoc]
            dissoc, dissoc_time = self.ys[sensor][step_dissoc], self.xs[sensor][step_dissoc]
            # Normalize assoc and dissoc
            #assoc = (assoc - np.min(assoc))/(np.max(assoc) - np.min(assoc))
            #dissoc = (dissoc - np.min(dissoc))/(np.max(dissoc) - np.min(dissoc))
            # Normalize times, necessary for the fits
            assoc_time_offset = np.min(assoc_time)
            assoc_time -= assoc_time_offset
            dissoc_time_offset = np.min(dissoc_time)
            dissoc_time -= dissoc_time_offset
            # Choose function
            if func=='biexp':
                func_assoc = biexp_rise_offset
                func_dissoc = biexp_decay_offset
                guess = (.5, .5, .1, .1, np.max(assoc), 0)
            elif func == 'monoexp':
                func_assoc = exp_rise_offset
                func_dissoc = exp_decay_offset
                guess = (.5, np.max(assoc), 0)
            # Perform association fitting
            try:
                fit_popt_assoc, fit_pcov_assoc = curve_fit(func_assoc, assoc_time, assoc, p0=guess, bounds=(0, np.inf))
            except:
                print("Could not fit association for sensor %i" % (sensor))
                pass
            fitted_assoc = func_assoc(assoc_time, *fit_popt_assoc)
            # Perform dissociation fitting
            try:
                fit_popt_dissoc, fit_pcov_dissoc = curve_fit(func_dissoc, dissoc_time, dissoc, p0=guess, bounds=(0, np.inf))
            except:
                print("Could not fit dissociation for sensor %i" % (sensor))
                pass
            fitted_dissoc = func_dissoc(dissoc_time, *fit_popt_dissoc)
            # Undo normalization of times
            assoc_time += assoc_time_offset
            dissoc_time += dissoc_time_offset
            # Save fit results to instance
            #if hasattr(self, 'fit_results'):
            print(self.fit_results)
            if sensor in self.fit_results:
                print("There are already fits available!")
                print("Will overwrite selected ones")
            else:
                self.fit_results[sensor] = {}
            self.fit_results[sensor]['fit_popt_assoc'] = fit_popt_assoc
            self.fit_results[sensor]['fit_pcov_assoc'] = fit_pcov_assoc
            self.fit_results[sensor]['fit_popt_dissoc'] = fit_popt_dissoc
            self.fit_results[sensor]['fit_pcov_dissoc'] = fit_pcov_dissoc
            # Write R2
            self.fit_results[sensor]['r2_assoc'] = r_sq(fitted_assoc, assoc)
            self.fit_results[sensor]['r2_dissoc'] = r_sq(fitted_dissoc, dissoc)
            # Calculate Kd
            if func=='monoexp':
                kobs = fit_popt_assoc[0]
                kdiss = fit_popt_dissoc[0]
                conc = self.step_info[sensor]['MolarConcentration_M'][step_assoc]
                r2_assoc = self.fit_results[sensor]['r2_assoc']
                r2_dissoc = self.fit_results[sensor]['r2_dissoc']
                # Calculate kon
                kon = (kobs + kdiss) / conc
                print("Sensor %i: Kobs is %.1e 1/s" % (sensor, kobs))
                print("Sensor %i: Kdis is %.1e 1/s" % (sensor, kdiss))
                print("Sensor %i: Kon is %.1e 1/s" % (sensor, kon))
                # Calculate Kd
                Kd = kdiss/kon
                print("Sensor %i: Determined Kd of %.1e M" % (sensor, Kd))
            else:
               pass
            # Plot results
            if plot:
                fig, ax = plt.subplots(1)
                # Pick color
                color = cycle[sensor]
                # Plot assoc and fit
                ax.plot(assoc_time, assoc, '.') #, color=color)
                ax.plot(assoc_time, fitted_assoc, '--') #, color='C01', lw=2)
                # Plot dissoc and fit
                ax.plot(dissoc_time, dissoc, '.') #, color=color)
                ax.plot(dissoc_time, fitted_dissoc, '--') #, '--', color='C01', lw=2)
                ax.set_xlabel('Time / s')
                ax.set_ylabel('Response / nm')
                # Add fitting info
                if func == 'monoexp':
                    # Get dimensions
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    # Create string for fit data
                    fit_str = 'Fit association:\n' +\
                        'k$_\mathrm{obs}$: %.1e 1/s\n' % kobs +\
                        'R$^2$: %.4f\n\n' % r2_assoc +\
                        'Fit dissociation:\n'  +\
                        'k$_\mathrm{diss}$: %.1e 1/s\n' % kdiss +\
                        'R$^2$: %.4f\n\n' % r2_dissoc +\
                        'Calculated:\n' +\
                        'k$_\mathrm{on}$: %.1e 1/sM\n' % kon +\
                        'K$_d$: %.1e M' % Kd
                    print(fit_str)
                    text = ax.text(0.95, .93, fit_str,transform=ax.transAxes,
                                   ha='right', va='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round' ,pad=.5))
                elif func == 'biexp':
                    pass
        return fig, ax

    def calculate_Kd(self):
        '''
        Calculates Kd value from fitting results
        '''
        # Check for which sensors, fitting was done
        if hasattr(self, 'fit_results'):
            sensors = list(self.fit_results.keys())
            print("Fit results found for these sensors (counting starts at 0)")
            print(sensors)
        else:
            print("No fitting done yet! Run fit_data first.")
            return None
        for sensor in sensors:
            print('Sensor %i' % sensor)
        
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
        
