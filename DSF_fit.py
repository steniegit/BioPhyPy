#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib
# For color bars
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import math
import scipy.optimize
import scipy.stats
# import math
# import sys
import copy

# import ipdb
# import datetime
# import copy
import os
import time
from scipy.optimize import curve_fit
# import shutil
from sklearn.metrics import r2_score
import pandas as pd
from matplotlib.patches import Rectangle
import ipdb
import natsort
from scipy.signal import savgol_filter

#### To do
'''
* Try out units with pint
* Outlier rejection
* Color map: rewrite so that it's based on conc
'''

# Move this to helpers later on
def kd_unit(kd):
    '''
    Converts Kd to label with suitable unit
    '''
    # Adjust labels to mM, muM or nM
    if 1E-9 < kd < 1E-6:
        kd_label = r"%.3g$\,\mathrm{nM}$" % (kd*1E9)
    elif 1E-6 < kd < 1E-3:
        kd_label = r"%.3g$\,\mu$M" % (kd*1E6)
    elif 1E-3 < kd < 1:
        kd_label = r"%.3g$\,\mathrm{mM}$" % (kd*1E3)
    else:
        kd_label = r"%.1E$\,\mathrm{M}$" % kd
    return kd_label

# Function for units
def unit_converter(unit):
    '''
    Converts unit strings into molar concentrations
    '''
    if unit == 'nM':
        return 10**-9
    elif unit == 'uM':
        return 10**-6
    elif unit == 'mM':
        return 10**-3
    elif unit == 'M':
        return 1
    else:
        print("Could not convert unit: %s" % unit)
        return None


class DSF_binding:
    '''
    Class for nanoDSF binding studies
    This class was written by Stephan Niebling and is
    based on isothermal analysis by Bai et al., Sci. Rep. 2019.

    No warranty whatsoever
    If you have questions please contact me:
    stephan.niebling@embl-hamburg.de

    Please cite both this work and my work
    Bai, N.; Roder, H.; Dickson, A. & Karanicolas, J. Scientific Reports, 9, 2019.
    Niebling et al. (unpublished)
    '''

    print("\nThis class was written by Stephan Niebling and is\n\
based on isothermal analysis by Bai et al., Sci. Rep. 2019.\n\
No warranty whatsoever\n\n\
If you have questions please contact me:\n\
stephan.niebling@embl-hamburg.de\n\n\
Please cite both the original on Thermofluor data analysis and my work:\n\
Bai, N.; Roder, H.; Dickson, A. & Karanicolas, Scientific Reports (2019), 9, 1-15.\n\
Niebling, S. et al., Scientific Reports (2021), 11, 1-17.\n\
Please also acknowledge the SPC core facility at EMBL Hamburg\n")

    def __init__(self, cp=0, folder='./'):
        '''
        Initialize instance

        Args:
            cp: deltaCp of unfolding
            folder: Folder in which to save figures/files
        '''
        self.folder = folder
        # Check if folder exists, otherwise create it
        if not os.path.isdir(folder):
            os.makedirs(folder)
            print("Created folder %s" % folder)
        self.fn = ''
        self.which = 'unknown'
        self.window = []
        self.concs = []
        self.isothermal_ts = []
        self.pconc = np.nan
        self.cp = cp
        self.tms = []
        # General plot parameters
        self.plot_alpha = 1  # Transparency
        self.plot_lw = 2  # Line width
        self.plot_fs = 12  # Font size
        matplotlib.rcParams.update({'font.size': self.plot_fs})
        return None

    def get_concs(self, concs, fn_concs):
        '''
        This helper function populates the self.concs of the class instance
        Ligand concentrations in the array concs have priority
        Otherwise it reads concentrations from fn_concs or tries to
        obtain them from a generic file called sample_concs.txt

        Args:
             concs:    ligand concentrations (in M)
             fn_concs: file name with ligand concentrations
        Returns: None
        '''

        # Get ligand concentrations
        if len(concs) > 0:
            # Make sure it's an array
            self.concs = np.array(concs)
        elif len(fn_concs) > 0:
            self.concs = np.genfromtxt(fn_concs)
        else:
            # Try to load from folder
            try:
                self.concs = np.genfromtxt(self.folder + '/sample_concs.txt')
            except:
                print("Neither concentrations (concs) nor concentration file (fn_concs) was given!")
                return None
            print("Loaded concentrations from %s/sample_concs.txt" % self.folder)
        return None

    def crop_temp_window(self, window=None):
        '''
        This crops the data below and above the limits specified
        '''
        # Overwrite window if specified, otherwise leave it unchanged
        if len(window) == 2:
            self.window = window
        if len(self.window) == 2:
            # First three rows are comments
            ind_lower = np.argmin(np.abs(self.temps - self.window[0]))
            ind_upper = np.argmin(np.abs(self.temps - self.window[1])) 
        else:
            # Otherwise do nothing
            return None
        # Change arrays
        self.temps = self.temps[ind_lower:ind_upper]
        self.fluo = self.fluo[ind_lower:ind_upper, :]
        return None
        

    ## Loading functions
    def load_xlsx(self, fn='', fn_concs='', concs=[], caps=[], window=[], which='Ratio', load_fit=True,outliers=[]):
        '''
        Load nanotemper processed xlsx and extract data

        Args:
            fn: filename
            fn_concs: file name with ligand concentrations
            concs: ligand concentrations (in M)
            caps: if length > 1: select these columns 
                caps is a list of integers with the capillary numbers+1

            window: temperature window in which to extract data
            which: 'Ratio', '350nm', '330nm' or 'Scattering', also 'Ratio (Unfolding)' etc.
            load_fit: Try to load previous fit parameters
            outliers: take out these columns
        Returns: None

        '''
        
        # Get folder and filename
        self.fn = os.path.basename(fn)
        self.folder = os.path.dirname(fn)
        #Get the capillary numbers
        self.caps = caps

        # Load ligand concentrations into instance
        self.get_concs(concs, fn_concs)
        
        # Temperature window to extract data
        self.window = np.sort(window)
        # Which data to use
        self.which = which
        # Outliers
        self.outliers = outliers
        # Load excel file
        # this needs to be the processed file!
        xlsx = pd.ExcelFile(self.folder + '/' + self.fn)
        # Check if file was generated from Prometheus Panta
        # Check if data is from Prometheus or Panta
        # Load data
        if 'Data Export' in xlsx.sheet_names:
            # For Panta
            print("Panta file format detected")
            # Load overview with capillary/sample information
            metadata = pd.read_excel(xlsx, 'Overview', index_col=None, header=0)
            self.metadata = metadata
            self.load_panta(xlsx)
        else:
            # For Prometheus
            self.load_prometheus(xlsx)
        # Fill concentrations if empty
        if len(self.concs) < 1:
            self.concs = np.arange(self.fluo.shape[1])+1
        # Cut window
        self.crop_temp_window(window=window)
        # Save signal and concs
        self.save_signal()
        # Also export full data
        # self.fluo_full = np.array(dat.iloc[3:,sort_fluo])
        print("Loaded xlsx file")
        # Try to load previous fit parameters if desired
        if load_fit:
            print("Try to load fit parameters")
            self.load_fit_fluo()
        # Initialize outlier indices
        self.inds_in = np.ones(self.fluo.shape[1])==1
        return None

    def load_prometheus(self, xlsx):
        '''
        '''
        dat = pd.read_excel(xlsx, self.which, index_col=None, header=None)

        # If the capillary argument was selected, select those columns
        if len(self.caps) >1:
            dat = dat.iloc[:,[0,1]+self.caps]        

        # Find first index with numbers (above is header)
        if self.which in ['Ratio', '350nm', '330nm', 'Scattering',
                     'Ratio (Unfolding)', '350nm (Unfolding)', '330nm (Unfolding)', 'Scattering (Unfolding)']:
            first_row = int(np.argwhere(list(dat.iloc[:, 0] == 'Time [s]'))) + 1
            first_column = 1
        elif self.which == 'RFU':
            first_row = 1
            first_column = 0
        else:
            print('Please choose which keyword: Ratio, 350nm, 330nm, Scattering or RFU')
            return None
        
        # Indices for extracting the whole range
        ind_lower = first_row
        ind_upper = -1
        # Create sample_concs.txt
        sort_ind = np.argsort(self.concs)
        # Take out outliers
        sort_ind = list(sort_ind)
        for out in self.outliers:
            sort_ind.remove(out)

        # Extract temperatures
        self.temps = np.array(dat.iloc[ind_lower:ind_upper, first_column]).astype('float')
        # and modify sort index for saving the fluorescence data
        sort_fluo = list(np.array(sort_ind) + first_column + 1)
        # Save concentrations
        self.concs = np.array(self.concs)
        # Order
        self.concs = self.concs[sort_ind]
        # Extract fluo
        self.fluo = np.array(dat.iloc[ind_lower:ind_upper, sort_fluo]).astype('float')
        # In case there are spaces in self.which, remove them now
        self.which = self.which.replace(' ','')
        if self.which in ['Ratio', '350nm', '330nm', 'Scattering',
                     'Ratio (Unfolding)', '350nm (Unfolding)', '330nm (Unfolding)', 'Scattering (Unfolding)']:
            # Extract sample IDs
            self.sample_ID = np.array(dat.iloc[first_row-3, sort_fluo])
            # Extract sample comments
            self.sample_comment = np.array(dat.iloc[first_row-2, sort_fluo])
        elif self.which=='RFU':
            self.sample_ID = np.array(dat.columns)[1:]
            self.sample_comment = np.array(dat.iloc[0,1:])
        # Extract capillary numbers
        self.capillary_no = np.array(dat.iloc[0, sort_fluo])
        return None

    def load_panta(self, xlsx):
        '''
        This function reads xlsx generated by a Prometheus Panta
        '''
        dat = pd.read_excel(xlsx, 'Data Export', index_col=None, header=0)

        # Determine which dataset needs to be generated
        if self.which=='Ratio':
            cols_pos = [dat.columns.get_loc(col) for col in dat if col.startswith('Ratio')]
        elif self.which=='350nm':
            cols_pos = [dat.columns.get_loc(col) for col in dat if col.startswith('350 nm')]
        elif self.which=='330nm':
            cols_pos = [dat.columns.get_loc(col) for col in dat if col.startswith('330 nm')]
        elif self.which=='Scattering':
            cols_pos = [dat.columns.get_loc(col) for col in dat if col.startswith('Scattering')]
        elif self.which=='Turbidity':
            cols_pos = [dat.columns.get_loc(col) for col in dat if col.startswith('Turbidity')]
        elif self.which == 'Cumulant Radius':
            cols_pos = [dat.columns.get_loc(col) for col in dat if col.startswith('Cumulant Radius')]
        cols_pos = np.array(cols_pos)

        # Get temperatures
        temps = np.array(dat.iloc[:, cols_pos-1])
        specs = np.array(dat.iloc[:, cols_pos])

        # Temps
        n_points = temps.shape[0]
        min_temp = int(np.nanmin(temps))
        max_temp = int(np.nanmax(temps))
        temp_int = np.linspace(min_temp, max_temp+1, n_points)

        # Extract columns
        # Interpolate column by column
        for i in range(specs.shape[1]):
            specs[:,i] = np.interp(temp_int, temps[:,i], specs[:,i])

        # Fill instance
        self.temps = temp_int
        self.fluo = specs
        # Get capillary number and sample ID
        self.sample_ID = np.array(self.metadata['Capillary'])
        # To keep the same style as the previous Prometheus
        self.capillary_no = np.array(self.metadata['Capillary'])
        self.sample_comment = np.array(self.metadata['Sample ID'])
        self.solven = np.array(self.metadata['Solvent'])

        return None

    def load_thermofluor(self, fn='', fn_concs='', concs=[], caps=[], window=[], outliers=[], load_fit=False):
        '''
        Load thermofluor data from ThermoFisher QuantStudio 3

        Args:
            fn: filename
            concs: ligand concentrations
            caps: if length > 1: select these columns 
                caps is a list of integers with the capillary numbers+1

            window: temperature window in which to extract data
            load_fit: Try to load previous fit parameters
            outliers: take out these columns
        Returns: None

        '''
        
        # Get folder and filename
        self.fn = os.path.basename(fn)
        self.folder = os.path.dirname(fn)

        #Get the capillary numbers
        self.caps = caps

        # Load ligand concentrations into instance
        self.get_concs(concs, fn_concs)

        # Temperature window to extract data
        self.window = window
        # Thermofluor data
        self.which = 'Thermofluor'
        # Load csv file
        table = pd.read_csv(fn, comment='*', delimiter='\t', header=1, thousands=',')
        positions = np.unique(table['Well Position'])
        # Initialize lists
        fluo, deriv, sample_comment, temps = [], [], [], []
        # Sort positions
        positions = np.array(natsort.os_sorted(positions))
        # Fill lists for each position
        for position in positions:
            # # Check that temperatures are OK
            # if len(temps) <1:
            #     temps = table['Temperature']
            # else:
            #     if table['Temperature'] != temps:
            #         print('Temperature mismatch starting at %s' % position)
            # # Append well name as sample_comment
            sample_comment.append(position)
            inds = table['Well Position']==position
            fluo.append(table[inds]['Fluorescence'])
            deriv.append(table[inds]['Derivative'])
            temps = np.array(table[inds]['Temperature'])
        print("Attention: No interpolation implemented yet! Temperatures might differ!")
        # Load data into instance
        self.concs = np.array(concs).astype(float)
        self.temps = np.array(temps).astype(float)
        self.fluo = np.array(fluo).astype(float).T
        self.deriv = np.array(deriv).astype(float).T
        self.sample_comment = np.array(sample_comment)
       
        # Indices for extracting only desired temp window
        if len(self.window) == 2:
            # First three rows are comments
            ind_lower = np.argmin(np.abs(temps - self.window[0]))
            ind_upper = np.argmin(np.abs(temps - self.window[1]))
        else:
            ind_lower = 0
            ind_upper = -1
        # Create sample_concs.txt
        sort_ind = np.argsort(self.concs)
        # Take out outliers
        sort_ind = list(sort_ind)
        for out in outliers:
            sort_ind.remove(out)

        # Extract temperatures
        self.temps = temps[ind_lower:ind_upper]
        # and modify sort index for saving the fluorescence data
        sort_fluo = list(np.array(sort_ind))
        # Save concentrations
        self.concs = np.array(self.concs)
        # Order
        self.concs = self.concs[sort_ind]
        # Extract fluo
        self.fluo = self.fluo[ind_lower:ind_upper, sort_ind]
        # In case there are spaces in self.which, remove them now
        self.which = self.which.replace(' ','')
        # Set up sample_ID and comment
        self.sample_ID = positions[sort_ind]
        self.sample_comment = positions[sort_ind]
        # Extract capillary numbers
        self.capillary_no = positions[sort_fluo]
        # Save signal and concs
        self.save_signal()
        # Also export full data
        # self.fluo_full = np.array(dat.iloc[3:,sort_fluo])
        print("Loaded csv file")
        # Try to load previous fit parameters if desired
        if load_fit:
            print("Try to load fit parameters")
            self.load_fit_fluo()
        # Initialize outlier indices
        self.inds_in = np.ones(self.fluo.shape[1])==1
        return None

    def load_csv(self, fn=''):
        '''
        This function loads csv data with
        temperatures in every 2nd column
        '''
        # Load file
        data = pd.read_csv(fn, header=0, delimiter=',')
        # Check that all temperature columns are identical
        temps = data.iloc[:,::2]
        for i in range(temps.shape[1]):
            if not temps.iloc[:,0].equals(temps.iloc[:,i]):
                print("Mismatch in temperatures detected! Aborting loading")
                return None
        # Only read the fluorescence data
        data_f = data.iloc[:,1::2]
        data_f.insert(0,'Temperatures', temps.iloc[:,0])

        # Get concentrations from headers
        protein_conc, ligand_conc = [], []
        for header in data.columns[::2]:
            header = header.split()
            # Ligand concentration
            if header[1] == 'ligand':
                lig = header[0]
            elif header[0] == 'ligand':
                lig = header[1]
            # Convert to number
            ligand_conc.append(float(lig[:-2])*unit_converter(lig[-2:]))
            # Protein conc
            if header[3] == 'protein':
                prot = header[2]
            elif header[2] == 'protein':
                prot = header[3]
            protein_conc.append(float(prot[:-2])*unit_converter(prot[-2:]))
        # Check that protein conc. is constant
        if np.sum(np.diff(protein_conc)) < np.finfo('float').eps:
            pconc = protein_conc[0]
            print("Protein concentration: %.eM" % pconc)
        else:
            pconc = None
            print("Protein concentrations do not match!!!")
        # Load data into instance
        self.temps = np.array(data_f['Temperatures'])
        self.fluo = np.array(data_f.iloc[:,1:])
        self.concs = np.array(ligand_conc)
        self.sample_ID = np.arange(len(self.concs))
        self.sample_comment = self.capillary_no = self.sample_ID
        # Protein concentration
        self.pconc = pconc
        return None
            
    def interpolate_data(self, factor=10):
        '''
        Interpolate data to reduce number of points, e.g for datasets 
        with many points the fittings may take long time
        '''
        # Backup temps and fluo
        self.temps_orig = copy.deepcopy(self.temps)
        self.fluo_orig = copy.deepcopy(self.fluo)
        # Determine new temps
        self.temps = self.temps[::factor]
        # Interpolate fluo for new temps
        self.fluo = np.zeros((len(self.temps), self.fluo.shape[1]))
        for i in range(self.fluo.shape[1]):
            self.fluo[:,i] =  np.interp(self.temps, self.temps_orig, self.fluo_orig[:,i])
        return None
        

    def sort_signal(self, criterium='concs'):
        '''
        Sort signal based on one criterium
        'concs', 'sample_ID', 'sample_comment' or 'capillary_no'
        
        Required:
        criterium: Sort criterium (see above) 
        '''
        # Look for criterium
        if criterium == 'concs':
            sort_crit = self.concs
        elif criterium == 'sample_ID':
            sort_crit = self.sample_ID
        elif criterium == 'sample_comment':
            sort_crit = self.sample_comment
        elif criterium == 'capillary_no':
            sort_crit = self.capillary_no
        else:
            print("No valid selection for criterium!")
            print("Choose one: 'concs', 'sample_ID', 'sample_comment' or 'capillary_no'")
        # Get sort ind
        sort_ind = natsort.index_natsorted(sort_crit)
        # Sort all entries
        self.fluo           = self.fluo[:, sort_ind]
        self.concs          = self.concs[sort_ind]
        if hasattr(self, 'capillary_no'):
            self.capillary_no = self.capillary_no[sort_ind]
        if hasattr(self, 'sample_ID'):
            self.sample_ID = self.sample_ID[sort_ind]
        if hasattr(self, 'sample_comment'):
            self.sample_comment = self.sample_comment[sort_ind]
        if hasattr(self, 'fluo_deriv'):
            self.fluo_deriv = self.fluo_deriv[:,sort_ind]
        if hasattr(self, 'local_fit_params'):
            self.local_fit_params = self.local_fit_params[:, sort_ind]
        if hasattr(self, 'local_fit_errors'):
            self.local_fit_errors = self.local_fit_errors[:, sort_ind]
        if hasattr(self, 'global_fit_params'):
            self.global_fit_params = self.global_fit_params[:, sort_ind]
        if hasattr(self, 'global_fit_errors'):
            self.global_fit_errors = self.global_fit_errors[:, sort_ind]
        if hasattr(self, 'global_fit_cp_params'):
            self.global_fit_cp_params = self.global_fit_cp_params[:, sort_ind]
        if hasattr(self, 'global_fit_cp_errors'):
            self.global_fit_cp_errors = self.global_fit_cp_errors[:, sort_ind]
        return None

    def merge_datasets(self, dataset):
        '''
        This function incorporates another class instance 
        into the current one
        '''
        # Check if dataset is DSF_Fit module
        if dataset.__module__.split('.')[-1] != 'DSF_fit':
            print("Dataset is no DSF_fit instance! Will not merge")
            return None
        # Check that dimensions are the same
        if len(self.temps) != len(dataset.temps):
            print("Temperature dimensions do not fit!")
            print("Will need to interpolate data")
            # New list for interpolated
            fluo_interp = np.zeros((len(self.temps), dataset.fluo.shape[1]))
            for i in range(dataset.fluo.shape[1]):
                fluo_interp[:,i] =  np.interp(self.temps, dataset.temps, dataset.fluo[:,i])
            dataset.fluo = fluo_interp
        # Merge all data
        self.fluo           = np.hstack((self.fluo, dataset.fluo))
        self.concs          = np.hstack((self.concs, dataset.concs))
        self.sample_ID      = np.hstack((self.sample_ID, dataset.sample_ID))
        self.sample_comment = np.hstack((self.sample_comment, dataset.sample_comment))
        self.capillary_no   = np.hstack((self.capillary_no, dataset.capillary_no))
        # Set filename to merged.xlsx
        self.fn = 'merged_data.xlsx'
        return None

    def remove_data(self, inds=[]):
        '''
        Remove entries from the instance
        '''
        # First sort indices in descending order
        inds = natsort.natsorted(inds, reverse=True)
        # Now loop through list
        for ind in inds:
            print("Remove sample ID: %s" % self.sample_ID[ind])
            self.fluo = np.delete(self.fluo, ind, axis=1)
            if hasattr(self, 'fluo_deriv'):
                self.fluo_deriv = np.delete(self.fluo_deriv, ind, axis=1)
            self.concs = np.delete(self.concs, ind)
            self.sample_ID = np.delete(self.sample_ID, ind)
            self.sample_comment = np.delete(self.sample_comment, ind)
            self.capillary_no = np.delete(self.capillary_no, ind)
        return None

    def extract_data(self, inds=[]):
        '''
        Extract entries from the instance and remove the rest
        '''
        # Go through entries in instance and print which are used
        # and which are removed
        for i in range(len(self.concs)):
            if i in inds:
                print("Will keep sample: %s" % self.sample_comment[i])
            else:
                print("Will remove sample: %s" % self.sample_comment[i])
        # Extract subsection
        self.fluo = self.fluo[:,inds]
        self.concs = self.concs[inds]
        if hasattr(self, 'sample_ID'):
            self.sample_ID = self.sample_ID[inds]
        if hasattr(self, 'sample_comment'):
            self.sample_comment = self.sample_comment[inds]
        if hasattr(self, 'capillary_no'):
            self.capillary_no = self.capillary_no[inds]
        if hasattr(self, 'deriv'):
            self.deriv = self.deriv[:,inds]
        return None
            
        
    def calc_fluo_deriv(self, window_width=5):
        '''
        Calculate first derivative and write it to self.fluo_deriv
        
        Requires:
        window_width: window size in °C used for the savitzky_golay filter

        Writes first derivative into self.fluo_deriv
        Writes max of 1st deriv into self.deriv_max
        Writes min of 1st deriv into self.deriv_min
        '''
        # Determine window length in points
        sg_window = int(window_width//np.median(np.diff(self.temps)))
        # Window length has to be odd
        if sg_window % 2 == 0:
            sg_window -= 1
        # Calculate first derivative
        self.fluo_deriv = savgol_filter(self.fluo, sg_window, 2, deriv=1, axis=0)
        # Determine min and max values
        max_ind = np.argmax(self.fluo_deriv, axis=0)
        self.deriv_max = self.temps[max_ind]
        min_ind = np.argmin(self.fluo_deriv, axis=0)
        self.deriv_min = self.temps[min_ind]
        return None
        
    def save_signal(self, no_sort_concs=False):
        '''
        This function writes a text file with the fluorescence
        signal vs. temperature (The first column contains the temperatures)
        and another text file with the ligand concentrations. 
        You can use sort_signal to sort the signal before.
        '''
        if no_sort_concs:
            sort_ind = np.arange(len(self.concs))
        else:
            sort_ind = np.argsort(self.concs)
        if len(self.concs) > 0:
            # Save concentrations
            np.savetxt('%s/sample_concs_%s.txt' % (self.folder, self.which), self.concs[sort_ind])
            print("Concentrations saved in: %s" % ('%s/sample_concs_%s.txt' % (self.folder , self.which)))
        # Save fluo
        np.savetxt(self.folder + '/sample_fluo_%s.txt' % self.which.replace(' ',''), np.hstack((self.temps.reshape(-1, 1), self.fluo[:, sort_ind])),
                   delimiter='\t')
        print("Data saved in: %s/sample_fluo_%s.txt" % (self.folder, self.which))
        return None
    
    def load_txt(self, fn_fluo='', fn_concs='', outliers=[], load_fit=True):
        '''
        This directly loads in fluorescence text file
        Args:
            fn_fluo: fluorescence (text file)
            fn_conc: concentrations (text file)
            outliers: List with outliers
            load_fit: Try to load previous fit parameters
        Returns:
        '''
        # Get folder and filename
        self.fn = os.path.basename(fn_fluo)
        self.folder = os.path.dirname(fn_fluo)
        # Try to find sample_concs.txt
        if len(fn_concs) == 0:
            fn_concs = self.folder + '/' +  self.fn.replace('fluo','concs')
            print(fn_concs)
            # Check if file exists
            if not os.path.isfile(fn_concs):
                print("Could not find %s in folder %s" % ('/' + self.fn.replace('fluo','concs'), self.folder))
                print("Please specify concentration file with fn_concs")
                return None
        # Determine which kind of data it is ('Ratio', '330nm', '350nm' or 'Scattering')
        self.which = self.fn.split('_')[-1].replace('.txt', '')
        # Load data
        self.fluo = np.genfromtxt(fn_fluo)[:, 1:]
        self.temps = np.genfromtxt(fn_fluo)[:, 0]
        print("Loaded fluorescence from %s" % fn_fluo)
        if not fn_concs == '':
            self.concs = np.genfromtxt(fn_concs)
            print("Loaded concentrations from %s" % fn_concs)
        # Try to load previous fit parameters if desired
        if load_fit:
            print("Try to load fit parameters")
            self.load_fit_fluo()
        # Sort according to concs
        sort_ind = np.argsort(self.concs)
        # Take out outliers
        sort_ind = list(sort_ind)
        for out in outliers:
            sort_ind.remove(out)
        self.concs = self.concs[sort_ind]
        self.fluo = self.fluo[:, sort_ind]
        # Initialize outlier indices
        self.inds_in = np.ones(self.fluo.shape[1])==1
        return None

    def simulate_curves_fixed_KD(self, KD=1E-6, dHU=140, dCpU=8, Tm=45, mu=20, mf=4, bu=1, bf=2, T=np.linspace(20,80,101), noise_perc=5):
        '''
        Instead of loading data this simulates
        the fluorescence signal for a given Kd
        The protein (pconc) and ligand concentrations (concs)
        have to be set in the class instance

        Required:
        KD:   Dissociation constant in M
        dHU:  Unfolding enthalpy in kcal/mol
        dCpU: Heat capacity change upon unfolding in kcal/molK
        Tm:   Melting temperature in °C
        mu:   Fluorescence slope for unfolded state
        mf:   Fluorescence slope for folded state
        bu:   Fluorescence intercept for unfolded state
        bf:   Fluorescence intercept for folded state
        T:    Temperatures for calculation of fluorescence signal in °C
        noise_perc: Noise level in %
        '''
        from dsf_simulation_helpers import ku, dG_total, get_fluo_from_fu
        # Calculate KU(T)
        KU = ku(T, Tm, dHU, dCpU)
        # Now we need to loop through the initial ligand concentrations
        LT = []
        fuT = []
        dGT = []
        KU_new_T = []
        YT = []
        # Check if ligand concentrations are set
        if len(self.concs) ==0:
            self.concs = np.logspace(0,22, num=23, base=0.5) * 5E-3
            self.concs[-1] = 0
        # Check if pconc is set
        if np.isnan(self.pconc):
            self.pconc = 1E-5
        # Check if isothermal_Ts are set
        if len(self.isothermal_ts) == 0:
            self.isothermal_ts = np.arange(Tm-5, Tm+6)
        # Loop through ligand concentrations and calculate fluorescence
        for L in self.concs:
            LT.append(self.calculate_L_free(L, KU, KD, self.pconc))
            dG, KU_new = dG_total(T, KU, L, KD)
            dGT.append(dG)
            KU_new_T.append(KU_new)
            # Fluo without noise
            fluo_temp = get_fluo_from_fu(T, KU_new/(1+KU_new), mu, mf, bu, bf)
            # Maximal amplitude
            amp = np.max(fluo_temp) - np.min(fluo_temp)
            # Noise with random numbers
            noise = np.random.uniform(1-noise_perc/100, 1+noise_perc/100, len(T)) * amp
            YT.append(fluo_temp+noise)
        # Create dictionary with results
        self.simulation = {'LT': np.array(LT).T,
                           'dGT':  np.array(dGT).T,
                           'KU_new': np.array(KU_new_T).T,
                           'Tm': Tm, 'dHU': dHU,'dCpU': dCpU, 'noise': noise_perc, 
                           'mu': mu, 'mf': mf, 'bu': bu, 'bf': bf}
        # Write fluorescence to instance
        self.fluo = np.array(YT).T
        self.temps = T
        # Create folder for files
        folder = self.folder + '/sim_kd_%.0E_pconc_%.0E_dCpu_%.0E_noiseperc_%i/' % (KD, self.pconc, dCpU, noise_perc)
        if not os.path.isdir(folder):
            os.makedirs(folder)
            print("Created folder %s" % folder)
        # Change to subfolder
        self.folder = folder
        # Save signal
        self.save_signal()
        return None

    ## Fitting functions
    def fit_fluo_local(self, bound=True, fit_algorithm='trf'):
        '''
        This is the first step of fitting the melting curves
        It uses the deltaCp specified before or otherwise 
        assumes a deltaCp of 0
        
        args:
            bound: Set Limits for Tm (min and max temp of used data)
                   and DeltaH (0 - inf)
        '''
        
        # Check if Cp is set, otherwise set it to 0
        if not hasattr(self, 'cp'):
            self.cp = 0
        num_datasets = len(self.concs)
        local_fit_params = np.empty([7, num_datasets])
        pcovs = []
        errors = np.empty([7, num_datasets])
           
        print("\nStarting local fit of thermal unfolding")
        # Time it
        start_time = time.time()
        for index in range(num_datasets):
            fit_params, err = self.fit_single_thermal_curve(self.temps, self.fluo[:, index], self.cp, bound=bound, fit_algorithm=fit_algorithm)

            # Fill parameters
            local_fit_params[0, index] = fit_params[0]
            local_fit_params[1, index] = fit_params[1]
            local_fit_params[2, index] = fit_params[2]
            local_fit_params[3, index] = fit_params[3]
            local_fit_params[4, index] = fit_params[4]
            local_fit_params[5, index] = fit_params[5]
            local_fit_params[6, index] = self.cp
            # Fill errors
            errors[0, index] = err[0]
            errors[1, index] = err[1]
            errors[2, index] = err[2]
            errors[3, index] = err[3]
            errors[4, index] = err[4]
            errors[5, index] = err[5]
            errors[6, index] = np.nan # Dummy entry
            # print('For ligand conc %.3f uM, Tm is %.2f oC.' % (self.concs[index], local_fit_params[0, index]))
        self.local_fit_params = local_fit_params
        self.local_fit_errors = errors
        # Time it
        calc_time = time.time() - start_time
        print('Finished local fit in %02.0fmin:%02.0fs' % (int(calc_time // 60), int(calc_time % 60)))
        # Save parameters
        self.save_fit_fluo(local_fit_params, errors, 'local')
        return None

    def fit_fluo_global(self): #, correct_local_fit=True
        '''
        This is the second step of fitting the melting curves:
        Now the thermal curves are fitted with the global slopes
        All other parameters are local
        This function requires the local fit as input (starting values)
        '''
        # Get existing parameters
        old_params, _, _ = self.select_fitting_params()
        # End function if no previous parameters could be loaded
        if len(old_params)==0:
            print("Could not load any parameters from previous fit!")
            print("Run fit_fluo_local() and/or fit_fluo_global() first.")
            return None
        num_datasets = len(self.concs)
        # num_datasets = np.sum(self.inds_in)

        # Initial parameters have to be in order: Tms, dHs, unfolded_intercepts, folded_intercepts, unfolded slope, folded_slope
        Tms = old_params[0, :]
        dHs = old_params[1, :]
        unfolded_intercepts = old_params[2, :]
        folded_intercepts = old_params[3, :]
        # For the shared slopes we could also use the average
        # For now will stick to what was done in the original code
        folded_slope = np.mean(old_params[4, :])
        unfolded_slope = np.mean(old_params[5, :])
        last_temp = len(self.temps) - 1
        # Instead of looping through the datasets, we will concatenate all data
        temp_concat, fluo_concat = self.concatenate_fluorescence(self.temps, self.fluo)

        # Starting values for fit
        p0 = np.array((*Tms, *dHs, *unfolded_intercepts, *folded_intercepts, folded_slope, unfolded_slope))

        # Specify limits for fit
        low_bound = np.array([-np.inf] * (4 * num_datasets + 2))
        high_bound = np.array([np.inf] * (4 * num_datasets + 2))
        # Specify lower limits for Tms
        low_bound[:num_datasets] = [np.min(self.temps)] * num_datasets
        high_bound[:num_datasets] = [np.max(self.temps)] * num_datasets
        # Specify lower limits for dH
        low_bound[num_datasets:(2 * num_datasets)] = [0] * num_datasets
        # print(low_bound)

        # Sanity check for initial values p0
        # Check if p0 is below lower bounds
        inds = np.argwhere(p0 < low_bound).squeeze()
        if len(inds) > 0:
            print("Warning! %s starting variables below lower bounds! Adjust temperature range?" % len(inds))
        for ind in inds:
            p0[ind] = low_bound[ind]
        # Check if p0 is above high bounds
        inds = np.argwhere(p0 > high_bound).squeeze()
        if len(inds) > 0:
            print("Warning! %s starting variables above upper bounds! Adjust temperature range?" % len(inds))
        for ind in inds:
            p0[ind] = high_bound[ind] 
            

        print("\nStarting global fit of thermal unfolding")
        # Time it
        start_time = time.time()
        # Do global fit
        params, cov = curve_fit(self.global_thermal_curves_woCp(self.cp), temp_concat, fluo_concat, p0=p0, bounds=(low_bound, high_bound), maxfev=10000)
        # Now bring parameter matrix in shape needed by program
        fit_params = np.ones((7, num_datasets)) * np.nan
        fit_params[0, :] = params[:num_datasets]  # Tm
        fit_params[1, :] = params[num_datasets:2 * num_datasets]  # dH
        fit_params[2, :] = params[2 * num_datasets:3 * num_datasets]  # unfolded intercepts
        fit_params[3, :] = params[3 * num_datasets:4 * num_datasets]  # folded intercepts
        fit_params[4, :] = np.tile(params[4 * num_datasets], num_datasets)
        fit_params[5, :] = np.tile(params[4 * num_datasets + 1], num_datasets)
        fit_params[6, :] = np.tile(self.cp, num_datasets) # Cp
        # Same for errors
        errors = np.sqrt(np.diag(cov))
        fit_errors = np.ones((7, num_datasets)) * np.nan    #np.copy(old_params) * np.nan
        fit_errors[0, :] = errors[:num_datasets]  # Tm
        fit_errors[1, :] = errors[num_datasets:2 * num_datasets]  # dH
        fit_errors[2, :] = errors[2 * num_datasets:3 * num_datasets]  # unfolded intercepts
        fit_errors[3, :] = errors[3 * num_datasets:4 * num_datasets]  # folded intercepts
        fit_errors[4, :] = np.tile(errors[4 * num_datasets], num_datasets)
        fit_errors[5, :] = np.tile(errors[4 * num_datasets + 1], num_datasets)
        fit_errors[6, :] = np.tile(np.nan, num_datasets) # Dummy value for Cp
        self.global_fit_params = fit_params
        self.global_fit_errors = fit_errors
        # Time it
        calc_time = time.time() - start_time
        #print(f'Finished global fit in {int(calc_time // 60):02}min:{int(calc_time % 60):02}s')
        print('Finished global fit in %02.0fmin:%02.0fs' % (int(calc_time // 60), int(calc_time % 60)))
        # Update melting temperatures
        self.tms = fit_params[0]
        # Save parameters
        self.save_fit_fluo(fit_params, fit_errors, 'global')
        return None

    def fit_fluo_global_cp(self): # , correct_local_fit=True
        '''
        This function fits the
        Returns:

        '''

        # Get existing parameters
        old_params, _, _ = self.select_fitting_params()
        # End function if no previous parameters could be loaded
        if len(old_params)==0:
            print("Could not load any parameters from previous fit!")
            print("Run fit_fluo_local() and/or fit_fluo_global() first.")
            return None
        # # Correct local fits
        # if correct_local_fit:
        #     self.correct_local_fit()
        # Define number of datasets
        num_datasets = len(self.concs)

        # Initial parameters have to be in order: Tms, dHs, unfolded_intercepts, folded_intercepts, unfolded slope, folded_slope
        Tms = old_params[0, :]
        dHs = old_params[1, :]
        unfolded_intercepts = old_params[2, :]
        folded_intercepts = old_params[3, :]
        # For the shared slopes we use the average
        folded_slope = np.mean(old_params[4, :])
        unfolded_slope = np.mean(old_params[5, :])
        last_temp = len(self.temps) - 1
        # Instead of looping through the datasets, we will concatenate all data
        temp_concat, fluo_concat = self.concatenate_fluorescence(self.temps, self.fluo)

        # Starting values for fit
        p0 = (*Tms, *dHs, *unfolded_intercepts, *folded_intercepts, folded_slope, unfolded_slope, self.cp)

        # Specify limits for fit
        low_bound = [-np.inf] * (4 * num_datasets + 3)
        high_bound = [np.inf] * (4 * num_datasets + 3)
        # Specify lower limits for Tms
        low_bound[:num_datasets] = [np.min(self.temps)] * num_datasets
        high_bound[:num_datasets] = [np.max(self.temps)] * num_datasets
        # Specify lower limits for dH
        low_bound[num_datasets:(2 * num_datasets)] = [0] * num_datasets
        # Specify lower limit for deltaCP
        low_bound[-1] = 0

        print("\nStarting global fit of thermal unfolding (with deltaCp)")
        print("This is an experimental feature! Handle with care!")
        # Time it
        start_time = time.time()
        # Do global fit
        global_fit_params, cov = curve_fit(self.global_thermal_curves, temp_concat, fluo_concat, p0=p0,
                                           bounds=(low_bound, high_bound))
        # Now bring parameter matrix in shape needed by program
        fit_params = np.zeros((7, num_datasets)) * np.nan  # np.copy(old_params)
        fit_params[0, :] = global_fit_params[:num_datasets]  # Tm
        fit_params[1, :] = global_fit_params[num_datasets:2 * num_datasets]  # dH
        fit_params[2, :] = global_fit_params[2 * num_datasets:3 * num_datasets]  # unfolded intercepts
        fit_params[3, :] = global_fit_params[3 * num_datasets:4 * num_datasets]  # folded intercepts
        fit_params[4, :] = np.tile(global_fit_params[4 * num_datasets], num_datasets)
        fit_params[5, :] = np.tile(global_fit_params[4 * num_datasets + 1], num_datasets)
        fit_params[6, :] = np.tile(global_fit_params[4 * num_datasets + 2], num_datasets)
        # Same for errors
        errors = np.sqrt(np.diag(cov))
        fit_errors = np.copy(fit_params) * np.nan
        fit_errors[0, :] = errors[:num_datasets]  # Tm
        fit_errors[1, :] = errors[num_datasets:2 * num_datasets]  # dH
        fit_errors[2, :] = errors[2 * num_datasets:3 * num_datasets]  # unfolded intercepts
        fit_errors[3, :] = errors[3 * num_datasets:4 * num_datasets]  # folded intercepts
        fit_errors[4, :] = np.tile(errors[4 * num_datasets], num_datasets)
        fit_errors[5, :] = np.tile(errors[4 * num_datasets + 1], num_datasets)
        fit_errors[6, :] = np.tile(errors[4 * num_datasets + 2], num_datasets)
        # Send to instance
        self.global_fit_cp_params = fit_params
        self.global_fit_cp_errors = fit_errors
        # Time it
        calc_time = time.time() - start_time
        print(f'Finished global fit in {int(calc_time // 60):02}min:{int(calc_time % 60):02}s')
        # Update deltaCp
        self.cp = fit_params[6, 0]
        print("Updated deltaCp to %.1f" % self.cp)
        # Update melting temperatures
        self.tms = fit_params[0]
        # Save parameters
        self.save_fit_fluo(fit_params, errors, 'global_cp_fit')
        return None

    def fit_isothermal(self, fit_alg='trf'):  # , Kus=[], Kds=[], fit_with_shared_slopes=0, Cp=0, protein_conc=0, fit_melting='local', figsize=[12, 8], lucas=False):
        '''
        This is an alternative Kd fit directly using
        to fit the fus.

        Input
        fit_alg:  Fit algorithm ('trf' or 'dogbox')

        Output
        It outputs a fit with error from the covariance matrix
        self.bind_params: Ku and Kd (latter in M)
        self.bind_errors: Ku and Kd errors
        '''
        print("\nStart isothermal analysis to fit Kd")
        # Check if protein concentration was set
        if np.isnan(self.pconc):
            print("Protein concentration not defined!")
            print("Set with pconc first.")
            return None
        # Check if isothermal temperatures are defined
        if len(self.isothermal_ts) == 0:
            print("You need to specify isothermal_ts first!")
            return None
        if self.pconc == None:
            print("Protein concentration not defined! Set it with pconc before doing fit_isothermal.")
            return None
        # Initialize isothermal data and fits
        self.isothermal_data = np.empty([len(self.concs), len(self.isothermal_ts)])
        self.unique_concs = np.unique(self.concs)
        self.isothermal_fits = np.empty([len(self.unique_concs), len(self.isothermal_ts)])
        bind_params, bind_errors = [], []
        # Select fitting parameters
        fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
        if len(fitting_params) == 0:
            print("No fitting parameters, no isothermal analysis!")
            return None
        # Loop through temperatures and fit Ku and Kd
        for i, binding_temp in enumerate(self.isothermal_ts):
            # Select fu
            self.isothermal_data[:, i] = self.calculate_fraction_unfolded(binding_temp,
                                                                          self.concs,
                                                                          fitting_params,
                                                                          self.cp).transpose()
            fu = self.isothermal_data[:, i]
            # In case there are nans in fu: remove them
            if np.sum(np.isnan(fu)) > 0:
                print("Remove %i nan values from fu for temp. %.0f" % (np.sum(np.isnan(fu)), binding_temp))
                inds = ~np.isnan(fu)
                concs_temp = copy.deepcopy(self.concs[inds])
                fu = fu[inds]
            else:
                concs_temp = self.concs
            # Determine start values for Ku and Kd
            start_Ku = np.max(fu) / (1.01 - np.max(fu))  # Based on maximum fu since ligand stabilizes folded, 1.01 to avoid division by zero
            half_pos = np.argmin(np.abs(fu - .5 * (np.max(fu) - np.min(fu))))
            start_Kd = concs_temp[half_pos]
            # Do curve fitting
            kd_function = self.calculate_fitted_isothermal_simple(self.pconc)
            params, pcov = curve_fit(kd_function, concs_temp, fu, maxfev=10000, method=fit_alg,
                                     p0=[start_Ku, start_Kd], bounds=((0, 0), (np.inf, np.inf)))
            bind_params.append(params)
            bind_errors.append(np.sqrt(np.diag(pcov)))
            print(f"T={binding_temp:.0f}C: Ku={params[0]:.2E}; Kd={params[1]:.2E}M")
            # Write to isothermal_fits
            self.isothermal_fits[:, i] = kd_function(self.unique_concs, *params)
            # print(start_Ku, start_Kd)
        print('Done with isothermal fit.')
        # Write to instance
        self.bind_params = np.array(bind_params)
        self.bind_errors = np.array(bind_errors)
        #if vanthoff:
        #    fig, ax = plt.subplots()
            
        return None

    def fit_tms(self, fit='alt'):
        '''
        Fit melting temperatures to get apparent Kd

        Args:
            fit: fit type; either 'single' binding site, 'hill' or 'alt'ernative

        Returns:

        '''
        print('\nFit melting temperatures')
        # Check if pconc is set
        if fit in ['single', 'hill']:
            if np.isnan(self.pconc):
                print("Protein concentration not defined!")
                print("Set with pconc first.")
                return None
        # Check if tms are already there
        if hasattr(self, 'tms'):
            print("Melting temperatures found! Will use these")
        else:
            print("No melting temperatures found! Will try to obtain them from fitting parameters")
            # Get fitting parameters
            fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
            if len(fitting_params) == 0:
                print("Did not find any fitting parameters! Run fit_fluo_local() or fit_fluo_global() first.")
                return None
            # Load melting temperatures
            self.tms = fitting_params[0]
        # Figure out starting value for apparent kd
        tms_middle = np.average(self.tms)
        kd0 = self.concs[np.argmin(np.abs(self.tms - tms_middle))]
        if fit == 'single':
            kd_func = self.single_site_kd(self.pconc)
            # Starting values
            p0 = (kd0, np.min(self.tms), np.max(self.tms))
        elif fit == 'hill':
            kd_func = self.hill_model
            p0 = (kd0, 1, np.min(self.tms), np.max(self.tms))
        elif fit == 'alt':
            print("Will do alternative fitting")
            # Starting value for dH
            dH0 = 100 # kcal/molK
            # Determine Tm for zero ligand conc
            if np.sum(self.concs == 0) < 1:
                min_conc = np.min(self.concs)
                print("No zero concentrations found for Tm0! Will use the lowest concentration of %.0EM instead." % min_conc)
                Tm0 = np.average(self.tms[self.concs == min_conc])
            else:
                Tm0 = np.average(self.tms[self.concs == 0])
            # Fitting function
            kd_func = self.tm_model1_review_fixTm0(Tm0)
            p0 = (kd0, dH0)
        # Do fit
        fit_params, cov = curve_fit(kd_func, self.concs, self.tms, p0=p0, maxfev = 10000)
        model = kd_func(self.concs, *fit_params)
        # Create auxiliary function that uses only the concentration as parameter
        def model_conc(fit_params):
            def return_func(conc_lig):
                return kd_func(conc_lig, *fit_params)
            return return_func
        self.tms_fit_func = model_conc(fit_params)
        # Calculate R2
        r2 = r2_score(self.tms, model)
        # Create dictionary with results
        tms_fit = {'fit_type': fit, 'params': fit_params, 'errors': np.sqrt(np.diag(cov)), 'fitted': model, 'r2': r2}
        if fit == 'alt':
            tms_fit['Tm0'] = Tm0
        self.tms_fit = tms_fit
        print('Done fitting melting temperatures.')
        return None

    # def fit_tms_alternative(self):
    #     print('\nFit melting temperatures with alternative method')
    #     # Check if pconc is set
    #     if np.isnan(self.pconc):
    #         print("Protein concentration not defined!")
    #         print("Set with pconc first.")
    #         return None
    #     # Check if tms are already there
    #     if hasattr(self, 'tms'):
    #         print("Melting temperatures found! Will use these")
    #     else:
    #         print("No melting temperatures found! Will try to obtain them from fitting parameters")
    #         # Get fitting parameters
    #         fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
    #         if len(fitting_params) == 0:
    #             print("Did not find any fitting parameters! Run fit_fluo_local() or fit_fluo_global() first.")
    #             return None
    #         # Load melting temperatures
    #         self.tms = fitting_params[0]
    #     # Figure out starting value for apparent kd
    #     tms_middle = np.average(self.tms)
    #     Kd0 = self.concs[np.argmin(np.abs(self.tms - tms_middle))]
    #     # Starting value for dH
    #     dH0 = 100 # kcal/molK
    #     # Determine Tm for zero ligand conc
    #     if np.sum(self.concs == 0) < 1:
    #         print("No zero concentrations found for Tm0! Exiting")
    #         return None
    #     Tm0 = np.average(self.tms[self.concs == 0])
    #     # Fitting function
    #     func = self.tm_model1_review_fixTm0(Tm0)
    #     # Fit it
    #     fit_params, cov = curve_fit(func, self.concs, self.tms, p0=(Kd0, dH0))
    #     err = np.sqrt(np.diag(cov))
    #     # Calculate R2
    #     model = func(self.concs, *fit_params)
    #     r2 = r2_score(self.tms, model)
    #     # Create dictionary with results
    #     tms_fit = {'fit_type': 'alt', 'params': fit_params, 'errors': np.sqrt(np.diag(cov)), 'fitted': model, 'r2': r2, 'Tm0': Tm0}
    #     self.tms_fit = tms_fit
    #     print('Done fitting melting temperatures with alternative method')
    #     return None

    ## Plotting functions
    def plot_fluo(self, fig='', ax='', save_fig=True):
        '''
        This functions generates a plot with
        the fluorescence/scattering data with the respective ligand
        concentration in a color map
        fig: figure handle (e.g. when using inside GUI)
        ax: axis handle, plot will be placed there
        save_fig: Save figure as pdf and png
        Returns: None
        '''
        print("\nPlot fluorescence or scattering")
        # Plot fluorescence
        if fig=='':
            # Use matplotlib.pyplot
            fig, axs = plt.subplots(1)
            new_plot=True
        else:
            # Use existing
            if ax=='':
                axs = fig.add_subplot(111)
            else:
                print('Specified axis')
                axs = ax
            new_plot=False
        ax = axs
        # Make sure that concs and fluo are sorted
        # This can not be the case if the data is directly
        # copied to the instance
        # sort_ind = np.argsort(self.concs)
        # self.concs = self.concs[sort_ind]
        # self.fluo = self.fluo[:, sort_ind]
        # if hasattr(self, 'sample_ID'):
        #     self.sample_ID = self.sample_ID[sort_ind]
        # if hasattr(self, 'sample_comment'):
        #     self.sample_comment = self.sample_comment[sort_ind]
        self.sort_signal()
        # Define color map
        if np.sum(self.concs==0)>0:
            print("Contains 0 concentrations")
            cmap = iter(plt.cm.jet(np.linspace(0, 1, len(np.unique(self.concs)))))
        else:
            cmap = iter(plt.cm.jet(np.linspace(0, 1, len(np.unique(self.concs)))))
        # Plot
        # Make sure that each conc. only has one color
        prev_conc = -1
        lh = []  # Line handles
        for i in range(len(self.concs)):
            if self.concs[i] == 0:
                temp, = ax.plot(self.temps, self.fluo[:, i], label=r"%.2f $\mu$M" % (self.concs[i]),
                                alpha=self.plot_alpha, lw=self.plot_lw, color='k')
                continue
            if prev_conc != self.concs[i]:
                temp, = ax.plot(self.temps, self.fluo[:, i], label=r"%.2f $\mu$M" % (self.concs[i]), alpha=self.plot_alpha, lw=self.plot_lw, color=cmap.__next__())
                lh.append(temp)
                prev_conc = self.concs[i]
                # print("New conc %.2f" % self.concs[i])
            else:
                ax.plot(self.temps, self.fluo[:, i], alpha=self.plot_alpha, color=temp.get_color(), lw=self.plot_lw)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=lh, title="Ligand concentration")
        ax.set_xlim([self.temps[0], self.temps[-1]])
        ax.set_xlabel(r'Temperature / $^\circ$C')
        if 'Ratio' in self.which:
            ax.set_ylabel('Fluorescence ratio (350/330)')
        elif '350nm' in self.which:
            ax.set_ylabel('Fluorescence intensity 350 nm')
        elif '330nm' in self.which:
            ax.set_ylabel('Fluorescence intensity 330 nm')
        elif 'Scattering' in self.which:
            ax.set_ylabel('Scattering / a.u.')
        elif 'RFU' in self.which:
            ax.set_ylabel('RFU / a.u.')
        elif 'Simulated' in self.which:
            ax.set_ylabel('Simulated fluorescence intensity / a.u')
            ax.yaxis.set_ticks([])
            ax.text(np.min(self.temps)+2, np.max(self.fluo), '%s%% noise' % self.noise, va='top', ha='left', bbox=dict(facecolor='gray', alpha=0.5), fontsize=20)
        # # Add box
        # boxmin = .98*np.min(self.fluo)
        # boxmax = 1.02*np.max(self.fluo)
        # frame1 = Rectangle((self.window[0],boxmin), self.window[1]-self.window[0], boxmax-boxmin, self.plot_lw=2, edgecolor='r', zorder=20, fill=0)
        # frame2 = Rectangle((self.window[0],boxmin), self.window[1]-self.window[0], boxmax-boxmin, self.plot_lw=2, edgecolor=None, zorder=-20, fill=1, self.plot_alpha=.5)
        # ax.add_patch(frame1)
        # ax.add_patch(frame2)

        # Set up color bar
        normalize = mcolors.LogNorm(vmin=np.min(self.concs[self.concs > 0]), vmax=np.max(self.concs))  # Or Normalize
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=plt.cm.jet)
        scalarmappaple.set_array(self.concs[self.concs > 0])
        cbar = plt.colorbar(scalarmappaple, ax=ax)
        cbar.set_label('Ligand conc. / M', rotation=270)
        cbar.ax.get_yaxis().labelpad = 15
        # Save figure
        if save_fig:
            fig.tight_layout()
            # For simulations, include kd and noise in fn
            if hasattr(self, 'kd') and hasattr(self, 'noise'):
                fn_plot = '%s/signal_%s_Cp_%.0f_noise_%.0f_kd_%.0E_pconc_%.0E' % (self.folder, self.which.replace(' ',''), self.cp, self.noise, self.kd, self.pconc) 
            else:
                fn_plot = '%s/signal_%s_Cp_%.0f_pconc_%.0E' % (self.folder, self.which.replace(' ',''), self.cp, self.pconc) 
            for fn_suffix in ['.pdf']: #, '.png']:
                fig.savefig(fn_plot + fn_suffix)
                print("Saved figure as %s" % fn_plot)
        #fig.show()
        return fig, ax

    def plot_fit_fluo(self, fig='', save_fig=True, figsize=[12,10], show_params=True):
        '''
        This plots the fits of thermal unfolding
        
        Requires
        fig: Figure handled (for GUI)
        save_fig: Save figure as pdf and png
        show_params: Show fitting parameters in plot

        Returns:
        '''
        print("\nPlot thermal unfolding fits")
        # Get fitting parameters
        fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
        if len(fitting_params) == 0:
            print("Did not find any fitting parameters! Run fit_fluo_local() or fit_fluo_global() first.")
            return None
        # ipdb.set_trace()
        num_datasets = len(self.concs)
        # figure out good dimensions for the subplot step (in case we're using it)
        subplot_horiz = math.ceil(math.sqrt(num_datasets))
        subplot_vert = math.ceil(num_datasets / subplot_horiz)
        if fig=='':
            # Use matplotlib.pyplot
            fig, axs = plt.subplots(subplot_vert, subplot_horiz, figsize=figsize)
            print("New fig")
        else:
            # Use existing
            axs = fig.subplots(subplot_vert,subplot_horiz)
            print('Existing fig')
        # First hide all
        for ax_sub in axs:
            for ax in ax_sub:
                ax.set_axis_off()
        # fig = plt.figure(5, figsize=(20, 14))  # Mod by SN
        # axs = []
        for c_index in range(num_datasets):
            # Experimental curve
            fluo = self.fluo[:, c_index]
            # Fit errors and params
            err = fitting_errors[:, c_index]
            par = fitting_params[:, c_index]
            fit  = self.single_thermal_curve(self.temps, *par)

            # Plot
            ax = axs[c_index // subplot_horiz, c_index % subplot_horiz]
            ax.set_axis_on()
            ax.plot(self.temps, fluo, 'bs')
            ax.plot(self.temps, fit, 'r', lw=4)
            # Show Tm (Added by SN)
            ax.axvline(fitting_params[0, c_index], color='k', linestyle=':')
            if len(fitting_errors) == 0:
                if show_params:
                    ax.text(Tm, .5 * (np.max(fluo) + np.min(fluo)), r"%.0f$^\circ$C   " % Tm, ha='right')
            else:
                # ax.text(Tm, .5*(np.max(fluo) + np.min(fluo)), r"%.0f$\pm$%.0f$^\circ$C   " % (Tm, fitting_errors[0,i]), ha='right')
                # ax.text(Tm, .5*(np.max(fluo) + np.min(fluo)), r"%.0f$\pm$%.0f$^\circ$C   " % (Tm, fitting_errors[0,i]), ha='right')
                text_str = 'T$_m$: %.1f$\pm$%.1f\n' % (par[0], err[0]) + \
                           '$\Delta$H: %.1f$\pm$%.1f\n' % (par[1], err[1]) + \
                           'interc$_u$: %.1f$\pm$%.0f%%\n' % (par[2], 100 * np.abs(err[2] / par[2])) + \
                           'interc$_f$: %.1f$\pm$%.0f%%\n' % (par[3], 100 * np.abs(err[3] / par[3])) + \
                           'slope$_u$: %.2e$\pm$%.0f%%\n' % (par[4], 100 * np.abs(err[4] / par[4])) + \
                           'slope$_f$: %.2e$\pm$%.0f%%' % (par[5], 100 * np.abs(err[5] / par[5]))
                # ax.text(Tm*1.08, .504*(np.max(fluo) + np.min(fluo)), text_str, ha='left', va='top')
                # Add Cp value and error
                if not np.isnan(err[6]):
                    text_str += r'\n$\Delta C_p$: %.1f$\pm$%.1f' % (par[6], err[6])
                # Add R2 to label
                r2 = r2_score(fluo, fit)
                text_str += '\n1-R$^2$: %.1e' % (1-r2)
                if show_params:
                    ax.text(.5 * (np.min(self.temps) + np.max(self.temps)),\
                            .5 * (np.max(fluo) + np.min(fluo)),\
                            text_str, ha='center', va='center',\
                            bbox=dict(facecolor='white', alpha=0.5))
            if ((c_index % subplot_horiz) == 0):
                # print the y-axis label only on the left-most column
                ax.set_ylabel('Fluorescence')
            # take the numbers off all columns (not just the middle ones)
            # ax.set_yticklabels([])
            if ((c_index + subplot_horiz >= num_datasets)):
                # print the x-axis label only on the bottom row
                ax.set_xlabel(r'Temperature ($^\circ$C)')
            else:
                # take the numbers off all other rows
                ax.set_xticklabels([])
            ax.set_title(r'Ligand conc. %.2f $\mu$M' % (self.concs[c_index] * 1E6))
            ax.set_title(rf'Ligand conc. {self.concs[c_index] * 1E6:.2f} $\mu$M')
        if save_fig:
            fig.tight_layout()
            # For simulations, include kd and noise in fn
            if hasattr(self, 'kd') and hasattr(self, 'noise'):
                fn_plot = r'%s/thermal_unfolding_curves_%s_Cp_%.0f_noise_%.0f_kd_%.0E_pconc_%.0E' % (self.folder, self.which.replace(' ',''), self.cp, self.noise, self.kd, self.pconc) 
            else:
                fn_plot = r'%s/thermal_unfolding_curves_%s_Cp_%.0f_pconc_%.0E' % (self.folder, self.which.replace(' ',''), self.cp, self.pconc) 
            # Save figure
            for fn_suffix in ['.pdf']: #, '.png']:
                fig.savefig(fn_plot + fn_suffix)
                print("Saved figure as %s" % fn_plot)
            #fig.show()
        return fig, axs

    def plot_fit_isothermal(self, fig='', figsize=[12, 8], save_fig=True, show_only_kd=True, rows=0, leg_loc='lower left'):
        '''
        This function plots the binding fits
        fig: Figure handled (for GUI)
        figsize: figure size
        save_fig: Save figure as pdf and png
        show_only_kd: Do not show Ku
        rows: number of rows, if set to 0, will determine automatically
        leg_loc: location of legend
        '''
        print("\nPlot isothermal fit")
        if not hasattr(self, 'bind_params'):
            print("No binding fitting data available. Call fit_isothermal() first.")
            return None, None
        # Determine fit_melting
        fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
        if len(fitting_params) == 0:
            print('No fitting parameters, no isothermal plot!')
            return None, None
        # Initialize plot
        if rows==0:
            subplot_horiz = math.ceil(math.sqrt(len(self.isothermal_ts)))
            subplot_vert = math.ceil(len(self.isothermal_ts) / subplot_horiz)
        else:
            subplot_vert = rows
            subplot_horiz = math.ceil(len(self.isothermal_ts) / subplot_vert)
        if fig=='':
            # Use matplotlib.pyplot
            fig, axs = plt.subplots(subplot_vert, subplot_horiz, figsize=figsize)
            print("New fig")
        else:
            # Use existing
            axs = fig.subplots(subplot_vert,subplot_horiz)
            print('Existing fig')
        # First hide all
        for ax in fig.get_axes():
            ax.set_axis_off()
        for t_index in range(len(self.isothermal_ts)):
            # Choose right parameters and errors
            params = self.bind_params[t_index]
            err = self.bind_errors[t_index]
            fu = self.isothermal_data[:, t_index]
            # In case there are nans in fu: remove them
            if np.sum(np.isnan(fu)) > 0:
                print("Remove %i nan values from fu for temp. %.0f" % (np.sum(np.isnan(fu)), binding_temp))
                inds = ~np.isnan(fu)
                concs_temp = copy.deepcopy(self.concs[inds])
                fu = fu[inds]
            else:
                concs_temp = self.concs
            ku, kd = params[0], params[1]
            ku_err, kd_err = err[0], err[1]

            # Plot
            if type(axs) == np.ndarray:
                if len(axs.shape) == 1:
                    ax = axs[t_index]
                else:
                    ax = axs[t_index// subplot_horiz, t_index % subplot_horiz]
            else:
                ax = axs
            ax.set_axis_on()
            ax.semilogx(concs_temp, fu, 'o', color='k', markerfacecolor='C0', alpha=self.plot_alpha,
                        linewidth=self.plot_lw)  # label='Experiment',
            if ((t_index + subplot_horiz >= len(self.bind_params))):
                # print the x-axis label only on the bottom row
                ax.set_xlabel('Ligand concentration / M')
            if ((t_index % subplot_horiz) == 0):
                # print the y-axis label only on the left-most column
                ax.set_ylabel('Fraction unfolded')
            # if not stop_it:
            model = self.calculate_fitted_isothermal_simple(self.pconc)
            temp_conc = np.unique(self.concs)
            ext_ligand_conc = 10 ** (np.linspace(np.log10(temp_conc[1] * .5), np.log10(temp_conc[-1] * 2), 1000))
            # Adjust labels to mM, muM or nM
            kd_label = kd_unit(kd)
            # Create label
            if show_only_kd:
                label = r"$K_d$=%s$\pm%.0f$" % (kd_label, kd_err / kd * 100)+ '%'
            else:
                label = r"$K_d$=%s$\pm%.0f$" % (kd_label, kd_err / kd * 100) +'%' + r'\n$K_U$=%.1E$\pm$%.0f' % (ku, ku_err / ku * 100)+ '%'
            hp, = ax.semilogx(ext_ligand_conc, model(ext_ligand_conc, *params), '-',\
                              label=label, zorder=-10, color='C1')
            # Plot errors
            # Correct values < 0 for
            params_lower = params - err
            params_lower[params_lower< 0] = 1E-20
            model_upper = model(ext_ligand_conc, *(params + err))
            model_lower = model(ext_ligand_conc, *(params_lower))
            ax.fill_between(ext_ligand_conc, model_upper, model_lower, facecolor=hp.get_color(), alpha=.5,
                            zorder=-20)
            # ax.legend(loc='lower left')
            ax.set_xlim([np.min(ext_ligand_conc), np.max(ext_ligand_conc)])
            ax.set_ylim([np.min(fu) - 0.05 * (np.max(fu) - np.min(fu)), np.max(fu) + 0.05 * (np.max(fu) - np.min(fu))])
            # ax.text(ax.get_xlim()[0], ax.get_ylim()[0], r" K$_D$=%.0f$\mu\mathrm{M}\pm%.0f$" % (params[1]/1E-6,err[1]/params[1]*100) +'%', ha='left', va='bottom', fontsize=18)
            # ax.set_title(r"K$_D$=%.0f$\mu\mathrm{M}\pm%.0f$%%" % (params[1]/1E-6,err[1]/params[1]*100))
            ax.set_title(r"T=%s$^\circ$C" % self.isothermal_ts[t_index])
            # ax.text(ax.get_xlim()[0], ax.get_ylim()[0], r" T=%s$^\circ$C" % self.isothermal_ts[i], ha='left', va='bottom', fontsize=18, fontweight='bold')
            # ax.text(ax.get_xlim()[0], ax.get_ylim()[0],
            #        r" K$_D$=%.1e$\mu\mathrm{M}\pm%.0f$" % (kd , kd_err / kd * 100) + '%',
            #        ha='left', va='bottom', fontsize=18)
            ax.legend(loc=leg_loc)
            fig.tight_layout()
        # Save data
        # np.savetxt(new_folder + '/own_fits_data.npz', fus=fus, kus=Kus, kds=kds, concs=concs)
        if save_fig:
            # For simulations, include kd and noise in fn
            if hasattr(self, 'kd') and hasattr(self, 'noise'):
                fn_plot = r'%s/isothermal_fits_%s_Cp_%.0f_noise_%.0f_kd_%.0E_pconc_%.0E' % (self.folder, self.which.replace(' ',''), self.cp, self.noise, self.kd, self.pconc) 
            else:
                fn_plot = r'%s/isothermal_fits_%s_Cp_%.0f_pconc_%.0E' % (self.folder, self.which.replace(' ',''), self.cp, self.pconc) 
            # Save figure
            for fn_suffix in ['.pdf']: #, '.png']:
                fig.savefig(fn_plot + fn_suffix)
                print("Saved figure as %s" % fn_plot)
            #fig.show()
        return fig, axs

    ## Start of Functions for shiny server

    def shiny_export_fit_fluo(self):
        '''
        This is a helper function to export fluorescence fits
        that can then be plotted in the web app

        It writes the following arrays to the instance:
        self.fit_fluo_pred   
        self.fit_fluo_params 
        self.fit_fluo_errs   
        '''    

        if not hasattr(self, 'select_fitting_params'):
            print("No  data available. Call fit_fluo_local() or fit_fluo_global() first.")
            return None, None

        fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
        num_datasets = len(self.concs)

        fit_fluo = np.empty((self.fluo).shape)

        for c_index in range(num_datasets):

            Tm = fitting_params[0, c_index]  # Tm
            dH = fitting_params[1, c_index]  # dH
            ti = fitting_params[2, c_index]  # unfolded intercept
            bi = fitting_params[3, c_index]  # folded intercept
            ts = fitting_params[4, c_index]  # unfolded slope
            bs = fitting_params[5, c_index]  # folded slope

            for t_index in range(len(self.temps)):
                T = self.temps[t_index] + 273.15
                R = 1.987 / 1000
                dG = dH * (1 - T / (Tm + 273.15)) - self.cp * (Tm + 273.15 - T + T * np.log(T / (Tm + 273.15)))
                try:
                    Ku = np.exp(-dG / (R * T))
                except RuntimeWarning:
                    print("Caught runtime warning for Ku = np.exp(-dG / (R * T))")
                    print(dG, -dG / (R*T))

                fit_fluo[t_index,c_index] = (Ku / (1 + Ku)) * (ts * T + ti) + (1 / (1 + Ku)) * (bs * T + bi)
        
        self.fit_fluo_pred   =  fit_fluo     
        self.fit_fluo_params =  fitting_params
        self.fit_fluo_errs   =  fitting_errors

    def pre_fit_isothermal(self):
        """
        Fill self.isothermal_data with the predicted fraction unfolded
        """

        # Initialize isothermal data and fits
        self.isothermal_data = np.empty([len(self.concs), len(self.isothermal_ts)])
        fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
        # Loop through temperatures and fill self.isothermal_data
        for i, binding_temp in enumerate(self.isothermal_ts):
            # Select fu
            self.isothermal_data[:, i] = self.calculate_fraction_unfolded(binding_temp,
                self.concs,fitting_params,self.cp).transpose()

    def shiny_export_isothermal(self):
        '''
        This is a helper function to export isothermal fits
        that can then be plotted in the web app

        It write the following arrays to the instance:
        self.kd_model_conc: fine conc grid for plotting
        self.kd_models: Fitted binding curves
        self.kd_models_lower: Lower limits based on fit errors
        self.kd_models_upper: Upper limits based on fit errors
        '''
        print("\nPlot isothermal fit")
        if not hasattr(self, 'bind_params'):
            print("No binding fitting data available. Call fit_isothermal() first.")
            return None, None
        # Determine fit_melting
        fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
        if len(fitting_params) == 0:
            print('No fitting parameters, no isothermal plot!')
            return None, None
        # Initialize arrays
        kd_fit_conc = np.unique(self.concs)
        # Finer grid for lines
        kd_model_conc = 10 ** (np.linspace(np.log10(kd_fit_conc[1] * .5), np.log10(kd_fit_conc[-1] * 2), 1000))
        kd_models = np.ones((len(kd_model_conc), len(self.isothermal_ts))) * np.nan
        kd_models_lower = copy.deepcopy(kd_models)
        kd_models_upper = copy.deepcopy(kd_models)
        # Loop through isothermal ts and fill arrays
        for t_index in range(len(self.isothermal_ts)):
            # Choose right parameters and errors
            params = self.bind_params[t_index]
            err = self.bind_errors[t_index]
            fu = self.isothermal_data[:, t_index]
            ku, kd = params[0], params[1]
            ku_err, kd_err = err[0], err[1]
            # Model
            model = self.calculate_fitted_isothermal_simple(self.pconc)
            kd_models[:, t_index] = model(kd_model_conc, *params)
            # Errors
            params_lower = params - err
            params_lower[params_lower< 0] = 1E-20
            kd_models_upper[:, t_index] = model(kd_model_conc, *(params + err))
            kd_models_lower[:, t_index] = model(kd_model_conc, *(params_lower))
        # Write to class instance
        self.kd_model_conc = kd_model_conc
        self.kd_models = kd_models
        self.kd_models_lower, self.kd_models_upper = kd_models_lower, kd_models_upper

    ## End of Functions for shiny server

    def tms_from_derivatives(self, which='max'):
        '''
        This function determines melting temperatures from
        derivatives and writes them into self.tms for later fitting
        
        which: Whether to use maximum ('max') or minimum ('min') of 
               first derivative for determination of tm
        '''
        # Calculate derivatives
        self.calc_fluo_deriv()
        # Write min or max values to tms
        if which=='max':
            self.tms = self.deriv_max
        elif which=='min':
            self.tms = self.deriv_min
        else:
            print("'which' has to be either 'min' or 'max'!")
            return None
        return self.tms
        

    def plot_tms(self, fig='', save_fig=True, simple_legend=False):
        '''
        Plot melting temperatures vs. ligand concentrations
        If global fit data is present, it will use this
        Otherwise it takes local fit

        fig: Figure handled (for GUI)
        save_fig: Save figure as pdf and png
        simple_legend: only show KDapp

        Returns: fig and axs handles
        '''
        print("Plot melting temperatures")
        # Check if fig parameter was defined
        if fig=='':
            # Use matplotlib.pyplot
            fig, axs = plt.subplots(1, figsize=[6.4, 4.8])
            new_plot=True
        else:
            # Use existing
            axs = fig.add_subplot(111)
            new_plot=False
            print("Use exisiting.")
        ax = axs
        # Check that there are melting temperatures
        if not hasattr(self, 'tms'):
            print("No melting temperatures found! Will use the ones from fluo fitting")
            # Get fitting parameters
            fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
            if len(fitting_params) == 0:
                print("Load fitting parameters or run fit_fluo_local()/global().")
                return None
            self.tms = fitting_params[0]
        # Check if fit was done
        if not hasattr(self, 'tms_fit'):
            print("No tms fitting results have been found!")
            print("Run tms_fit() first.")
            print("Will only plot melting temperatures.")
            plot_fit = False
        else:
            plot_fit = True
            # Log equal spaced concs for plot
            concs_temp = np.unique(self.concs)
            if concs_temp[0] == 0:
                concs_int = 10 ** (np.linspace(np.log10(concs_temp[1]), np.log10(concs_temp[-1]), 1000))
            else:
                concs_int = 10 ** (np.linspace(np.log10(concs_temp[0]), np.log10(concs_temp[-1]), 1000))
        

        # Plot
        ax.semilogx(self.concs, self.tms, 'o', color='k', markerfacecolor='red')  # , label='Exp. melting temperatures')
        ax.set_ylabel(r'T$_\mathrm{m}$ / $^\circ$C')
        ax.set_xlabel(r'Ligand conc. / M')
        if plot_fit:
            fit_params = self.tms_fit['params']
            err = self.tms_fit['errors'] / self.tms_fit['params'] * 100
            if self.tms_fit['fit_type'] == 'single':
                leg_label = 'Single-site model: \nK$_{d,\mathrm{app}}$=%.2EM$\pm$%.0f%% \n1-R$^2$=%.1E' % (fit_params[0], err[0], 1-self.tms_fit['r2'])
                kd_func = self.single_site_kd(self.pconc)
            elif self.tms_fit['fit_type'] == 'hill':
                leg_label = 'Cooperative model (Hill): \nK$_{d,\mathrm{app}}$=%.2EM$\pm$%.0f%% \nn=%.2f$\pm$%.0f%% \n1-R$^2$=%.1E' % (fit_params[0], err[0], fit_params[1], err[1], 1-self.tms_fit['r2'])
                kd_func = self.hill_model
            elif self.tms_fit['fit_type'] == 'alt':
                print("Alternative method")
                leg_label = 'T$_m$ model: \nK$_{d,\mathrm{app}}$=%.2EM$\pm$%.0f%% \n$\Delta H_U$=%.1fkcal/molK$\pm$%.0f%% \n1-R$^2$=%.1E' % (fit_params[0], err[0], fit_params[1], err[1], 1-self.tms_fit['r2'])
                kd_func = self.tm_model1_review_fixTm0(self.tms_fit['Tm0'])
            if simple_legend:
                kd_label = kd_unit(fit_params[0])
                leg_label = r'K$_{d,\mathrm{app}}=$%s$\pm$%.0f%%' % (kd_label, err[0])
            ax.semilogx(concs_int, kd_func(concs_int, *fit_params), \
                        label=leg_label, zorder=-20, lw=self.plot_lw)
            ax.legend()
            # Write results to instance
            #self.tms_fit['model_fine'] = np.vstack((concs_int, kd_func(concs_int, *fit_params))).T
            if new_plot:
                fig.tight_layout()
        if save_fig:
            # For simulations, include kd and noise in fn
            if hasattr(self, 'kd') and hasattr(self, 'noise'):
                fn_plot = '%s/tms_%s_Cp_%.0f_noise_%.0f_kd_%.0E_pconc_%.0E' % (self.folder, self.which.replace(' ',''), self.cp, self.noise, self.kd, self.pconc) 
            else:
                fn_plot = '%s/tms_%s_Cp_%.0f_pconc_%.0E' % (self.folder, self.which.replace(' ',''), self.cp, self.pconc) 
            for fn_suffix in ['.pdf']: #, '.png']:
                if plot_fit:
                    fn_plot += '_%s' % self.tms_fit['fit_type']
                #if plot_fit:
                #    fn_plot = '%s/melting_temps_%s_Cp_%.0f_pconc_%.0f_%s_%s%s' % (self.folder, self.which.replace(' ',''), self.cp, self.pconc, fit_melting, self.tms_fit['fit_type'], fn_suffix)
                #else:
                #    fn_plot = '%s/melting_temps_%s_Cp_%.0f_pconc_%.0f_%s%s' % (
                #    self.folder, self.which.replace(' ',''), self.cp, self.pconc*1E6, fit_melting, fn_suffix)
                fn_plot = fn_plot + fn_suffix
                fig.savefig(fn_plot)
                print("Saved figure as %s" % fn_plot)
        #ipdb.set_trace()
        #fig.show()
        return fig, ax

    ## Saving functions
    def save_fit_fluo(self, params, errors, fn_prefix):
        '''
        This function uses parameters and errors and saves
        these in two files text files

        Args:
            parameters: fitting parameters
            errors: fitting errors
            fn_prefix: Name, e.g. 'local' or 'global'
        Returns: None
        '''
        # File names
        fn_param = "%s/fluo_fit_%s_%s_cp_%.0f_params.txt" % (self.folder, self.which.replace(' ',''), fn_prefix, self.cp)
        fn_errors = "%s/fluo_fit_%s_%s_cp_%.0f_errors.txt" % (self.folder, self.which.replace(' ',''), fn_prefix, self.cp)
        # Header
        if len(params) == 6:
            header = "  Tm (oC)     dH (kcal/mol)       intercept_U       intercept_F           slope_U           slope_F"
        elif len(params) == 7: # deltaCp fit
            header = "  Tm (oC)     dH (kcal/mol)       intercept_U       intercept_F           slope_U           slope_F              deltaCp"
        else:
            print("Need to define save_fit_fluo for number of parameters = %i" % len(params))
            return None
        np.savetxt(fn_param, params.T, fmt='%15.5f', header='Params' + header, delimiter="   ", comments="")
        np.savetxt(fn_errors, errors.T, fmt='%15.5f', header='Errors' + header, delimiter="   ", comments="")
        print("Wrote parameters to %s" % fn_param)
        print("Wrote errors to %s" % fn_errors)
        return None

    ### Loading functions
    def load_fit_fluo(self):
        '''
        Searches folder for existing text files and loads them

        Returns: None
        '''
        # Check for global
        fn_prefix = 'global'
        fn_params = "%s/fluo_fit_%s_%s_cp_%.0f_params.txt" % (self.folder, self.which.replace(' ',''), fn_prefix, self.cp)
        fn_errors = "%s/fluo_fit_%s_%s_cp_%.0f_errors.txt" % (self.folder, self.which.replace(' ',''), fn_prefix, self.cp)
        print(fn_params, fn_errors)
        if os.path.exists(fn_params) and os.path.exists(fn_errors):
            self.global_fit_params = np.genfromtxt(fn_params, skip_header=1).T  # , delimiter="   ")
            print("Loaded fit parameters from %s" % fn_params)
            self.global_fit_errors = np.genfromtxt(fn_errors, skip_header=1).T  # , delimiter="   ")
            print("Loaded fit errors from %s" % fn_errors)
            # Update melting temperatures
            self.tms = self.global_fit_params[0]
        # Check for local
        fn_prefix = 'local'
        fn_params = "%s/fluo_fit_%s_%s_cp_%.0f_params.txt" % (self.folder, self.which.replace(' ',''), fn_prefix, self.cp)
        fn_errors = "%s/fluo_fit_%s_%s_cp_%.0f_errors.txt" % (self.folder, self.which.replace(' ',''), fn_prefix, self.cp)
        print(fn_params, fn_errors)
        if os.path.exists(fn_params) and os.path.exists(fn_errors):
            self.local_fit_params = np.genfromtxt(fn_params, skip_header=1).T  # , delimiter="   ")
            print("Loaded fit parameters from %s" % fn_params)
            self.local_fit_errors = np.genfromtxt(fn_errors, skip_header=1).T  # , delimiter="   ")
            print("Loaded fit errors from %s" % fn_errors)
            # Update melting temperatures
            self.tms = self.local_fit_params[0]
            #self.reject_outliers()
        else:
            # If nothing was found
            print("No parameters were found in folder %s for deltaCp=%.0f!" % (self.folder, self.cp))
            print("Run fit_fluo_local() and fit_fluo_global() first.")
            return None
        return None

    ### Helper functions
    def ku(self, T, Tm, dH, Cp):
        '''
        Unfolding constant KU vs. T
        This approximation is only valid around Tm

        Input:
        T: Temperatures in °C
        Tm: Melting temperature of protein in °C
        dH: Enthalpy change in kcal/mol unfolding
        Cp: Change of heat capacity in kcal/molK for unfolding
  
        Returns: equilibrium constant KU
        '''
        # Convert to K
        T_K = T +  273.15
        Tm_K = Tm + 273.15
        # Gas constant in kcal/molK
        R = 1.987E-3 
        # Free enthalpy
        dG = dH*(1-T_K/Tm_K) - Cp*(Tm_K -T_K + T_K * np.log(T_K/Tm_K))
        # Equilibrium constant
        Ku = np.exp(-dG/(R*T_K))
        return Ku
            
    
    def calculate_fraction_unfolded(self, binding_temperature, concs, fitting_params, Cp):
        '''
        Build the (isothermal) fraction folded curve at a specified temperature
        Args:
            binding_temperature: temperature to calculate fu
            concs: ligand concentrations
            fitting_params: results from local or global fit
            Cp: deltaCp of unfolding
        Returns:
        '''
        num_datasets = len(concs)
        fraction_unfolded = np.empty((concs).shape)
        for c_index in range(num_datasets):
            Tm = fitting_params[0, c_index]  # Tm
            dH = fitting_params[1, c_index]  # dH
            Ku = self.ku(binding_temperature, Tm, dH, Cp)
            fraction_unfolded[c_index] = (Ku / (1 + Ku))
        return fraction_unfolded

    def fit_single_thermal_curve(self, temps, fluo, cp, bound=True, fit_algorithm='trf'):
        '''
        Local fit of thermal curve

        Args:
            temps: Temperatures
            fluo:  Fluorescence intensities
            cp:    DeltaCp
            fit_algorithm: 'trf' or 'lm', 'lm' only possible if bound=False
            bound: Set Limits for Tm (min and max temp of used data)
                   and DeltaH (0 - inf)

        Returns: None
        '''
        init_dH = 150
        window_size = int(len(temps) / 10)
        (init_bottom_slope, init_bottom_intercept, junk1, junk2, junk3) = scipy.stats.linregress(
            temps[0:window_size] + 273.15, fluo[0:window_size])
        tse = len(temps) - 1
        (init_top_slope, init_top_intercept, junk1, junk2, junk3) = scipy.stats.linregress(
            temps[tse - window_size:tse] + 273.15, fluo[tse - window_size:tse])
        fluo_midpoint = (np.max(fluo) + np.min(fluo)) / 2

        # Determine initial melting temperatures
        init_Tm = temps[np.argmin(np.abs(fluo - fluo_midpoint))].squeeze()

        # Starting values for fit
        p0 = (init_Tm, init_dH, init_top_intercept, init_bottom_intercept, init_top_slope, init_bottom_slope)

        # Define boundaries for fit
        low_bound = [-np.inf] * 6
        high_bound = [np.inf] * 6
        # Define temps
        low_bound[0] = np.min(self.temps)
        high_bound[0] = np.max(self.temps)
        # Define DeltaH
        low_bound[1] = 0        

        # Function with fixed Cp
        model = self.single_thermal_curve_woCp(cp)
        # Fit
        #par, cov = curve_fit(model, temps, fluo, p0=p0, method=fit_algorithm, maxfev=1E5)
        if bound:
            par, cov = curve_fit(model, temps, fluo, p0=p0, bounds=(low_bound, high_bound), max_nfev=1E6, method=fit_algorithm)
        else:
            par, cov = curve_fit(model, temps, fluo, p0=p0, method=fit_algorithm)
            #print("Local fit without boundaries")
       
        # if fit_algorithm == 'trf':
        #     par, cov = curve_fit(model, temps, fluo, p0=(
        #     init_Tm, init_dH, init_top_intercept, init_bottom_intercept, init_top_slope, init_bottom_slope),
        #                          method='trf', maxfev=1E5)  # method='trf', bounds=(bounds_min, bounds_max))
        # elif fit_algorithm == 'lm':
        #     par, cov = curve_fit(model, temps, fluo, p0=(
        #     init_Tm, init_dH, init_top_intercept, init_bottom_intercept, init_top_slope, init_bottom_slope),
        #                          method='lm', maxfev=1E5)
        errors = np.sqrt(np.diag(cov))
        return par, errors

    def single_thermal_curve(self, temperatures, Tm, dH, unfolded_intercept, folded_intercept, unfolded_slope, folded_slope, Cp):
        '''
        Uses parameters from local/global fit and outputs thermal unfolding curve
        Args:
            temperatures:
            Tm: melting temperatures
            dH: unfolding enthalpy in kcal/mol
            unfolded_intercept:
            folded_intercept:
            unfolded_slope:
            folded_slope:
            Cp: deltaCp of thermal unfolding in kcal/(mol*K)
        Returns: None
        '''
        # Tm, dH, unfolded_intercept, folded_intercept, unfolded_slope, folded_slope, Cp = parameters
        T = temperatures + 273.15
        # Gas constant in kcal/(mol*K)
        R = 1.987E-3
        # dG and dH in kcal/mol; Cp in kcal/(mol*K)
        dG = dH * (1 - T / (Tm + 273.15)) - Cp * (Tm + 273.15 - T + T * np.log(T / (Tm + 273.15)))
        Ku = np.exp(-dG / (R * T))
        Y = (Ku / (1 + Ku)) * (unfolded_slope * T + unfolded_intercept) + (1 / (1 + Ku)) * (
                    folded_slope * T + folded_intercept)
        return Y

    def single_thermal_curve_woCp(self, Cp):
        '''
        Wrapper script that returns single_thermal_curve function with a fixed Cp
        Args:
            Cp: deltaCp of thermal unfolding
        Returns: function
        '''

        # This returns of single_thermal_curve function with a fixed Cp
        def return_func(temperatures, Tm, dH, unfolded_intercept, folded_intercept, unfolded_slope, folded_slope):
            return self.single_thermal_curve(temperatures, Tm, dH, unfolded_intercept, folded_intercept, unfolded_slope,
                                             folded_slope, Cp)

        return return_func

    def single_thermal_curve_Kd(self, temperatures, Tm, dH0, unfolded_intercept, folded_intercept, unfolded_slope, folded_slope, Cp, Kd, conc):
        '''
        Uses parameters from local/global fit and outputs thermal unfolding curve
        for a particular Kd and Ku0
        Args:
            temperatures:
            Tm: melting temperatures
            dH: unfolding enthalpy in kcal/mol
            unfolded_intercept:
            folded_intercept:
            unfolded_slope:
            folded_slope:
            Cp: deltaCp of thermal unfolding in kcal/(mol*K)
            Kd: dissociation constant in M^-1
            dH0: unfolding enthalpy for pure protein (no ligand)
            
        Returns: None
        '''
        # Tm, dH, unfolded_intercept, folded_intercept, unfolded_slope, folded_slope, Cp = parameters
        T = temperatures + 273.15
        # Gas constant in kcal/(mol*K)
        R = 1.987E-3
        # dG and dH in kcal/mol; Cp in kcal/(mol*K)
        dG0 = dH0 * (1 - T / (Tm + 273.15)) - Cp * (Tm + 273.15 - T + T * np.log(T / (Tm + 273.15)))
        Ku0 = np.exp(-dG0 / (R * T))
        Y = (Ku0 / (Ku0 + conc/Kd  + 1)) * (unfolded_slope * T + unfolded_intercept) + \
            (1 - Ku0 / (Ku0 + conc/Kd  + 1)) * (folded_slope * T + folded_intercept)
        return Y

    
    def concatenate_fluorescence(self, temperatures, fluorescence):
        '''
        Concatenate fluorescence matrix in single column matrix
        for global fit
        '''
        reps = fluorescence.shape[1]
        temps = np.tile(temperatures, reps)
        fluors = np.hstack(fluorescence.T)
        return temps, fluors

    def global_thermal_curves(self, temperatures, *args):  # Tms, dHs, unfolded_intercepts, folded_intercepts, unfolded_slopes, folded_slopes, Cp):
        '''
        This is a function to calculate all curves at once
        It is used for the global fit
        The arguments have to be in the following order:
        Single Tms
        Single dHs
        Single unfolded_intercepts
        Single folder_intercepts
        Global unfolded_slope
        Global folded_slope
        '''
        # Number of "repeats"
        num_datasets = np.sum(temperatures == temperatures[0])
        Tms = list(args[:num_datasets])
        dHs = args[num_datasets:2 * num_datasets]
        unfolded_intercepts = args[2 * num_datasets:3 * num_datasets]
        folded_intercepts = args[3 * num_datasets:4 * num_datasets]
        unfolded_slope = args[4 * num_datasets]
        folded_slope = args[4 * num_datasets + 1]
        Cp = args[4 * num_datasets + 2]
        # New temperature
        single_temp = temperatures[:len(temperatures) // num_datasets]
        # Data is appended to a list
        all_data = []
        for i in range(num_datasets):
            T = single_temp + 273.15
            R = 1.987 / 1000
            if Tms[i] < 0:
                Tms[i] = 0
                #print(i)
                #print(Tms[i])
            dG = dHs[i] * (1 - T / (Tms[i] + 273.15)) - Cp * (Tms[i] + 273.15 - T + T * np.log(T / (Tms[i] + 273.15)))
            # except RuntimeWarning:
            #     print("\n Runtime warning detected in log expression log(T/(Tms[i] + 273.15))!!")
            #     print(Tms[i])
            Ku = np.exp(-dG / (R * T))
            Y = (Ku / (1 + Ku)) * (unfolded_slope * T + unfolded_intercepts[i]) + (1 / (1 + Ku)) * (
                        folded_slope * T + folded_intercepts[i])
            all_data.append(Y)
        return np.hstack(all_data)

    def global_thermal_curves_woCp(self, Cp):
        '''
        Wrapper script that returns global_thermal_curves function with a fixed Cp
        Args:
            Cp: deltaCp of thermal unfolding
        Returns: function
        '''

        def return_func(temperatures, *args):
            args = (*args, Cp)
            return self.global_thermal_curves(temperatures, *args)

        return return_func

    def calculate_fitted_isothermal_old(self, ligand_conc, Ku, Kd, protein_conc):
        '''
        Calculate binding curve vs. ligand concentration
        Args:
            ligand_conc: ligand concentrations
            Ku: equilibrium constant of thermal unfolding
            Kd: dissociation constant
            protein_conc:
        Returns:
        '''
        Kd = abs(Kd)
        Ku = abs(Ku)
        b = protein_conc + Kd * (1 + Ku) - ligand_conc
        c = -1.0 * ligand_conc * Kd * (1 + Ku)
        L_free = (-b + np.sqrt(b ** 2 - 4 * c)) / 2
        #sqrt_expr = np.sqrt(0.25*(ligand_conc + protein_conc/(Ku+1) - Kd)**2 - ligand_conc * protein_conc / (Ku + 1))
        #L_free = .5*ligand_conc - .5*protein_conc/(Ku+1) + .5*Kd + sqrt_expr
        # assume that L_free = L_tot (if we don't want to use the quadratic equation above)
        # L_free = concs[index]
        # then use L_free to get fraction unfolded
        fit_fraction_unfolded = 1 / (1 + (1 / Ku) * (1 + L_free / Kd))
        return fit_fraction_unfolded

    def calculate_fitted_isothermal(self, ligand_conc, Ku, Kd, pconc):
        '''
        Calculate binding curve vs. ligand concentration
        Args:
            ligand_conc: ligand concentrations
            Ku: equilibrium constant of thermal unfolding
            Kd: dissociation constant
            protein_conc:
        Returns:
        '''
        # Calculate L_free
        L_free = self.calculate_L_free(ligand_conc, Ku, Kd, pconc)
        # Calculate fraction unfolded
        fu = 1 / (1 + 1/Ku *(1+L_free/Kd))
        return fu

    def calculate_L_free(self, ligand_conc, Ku, Kd, pconc):
        '''
        Calculate the equilibrium concentration of free ligand
        in the equation: U + F <=>[Ku] L + F <=>[Kd] LF
        Args: 
           ligand_conc: initial ligand concentrations
           pconc: total protein concentration
           Ku: eqilibrum constant of unfolding
           Kd: dissociation constant in M
        '''
        # First calculate complex concentration by solving quadratic expression
        p = pconc/(Ku+1) + ligand_conc + Kd
        q = pconc*ligand_conc/(Ku+1)
        # if pconc > Kd:
        #     complex_conc = .5*p + np.sqrt(0.25*p**2 - q)
        # else:
        complex_conc = .5*p - np.sqrt(0.25*p**2 - q)
        # Calculate Lfree
        L_free = ligand_conc - complex_conc
        return L_free
        
    
    def calculate_fitted_isothermal_simple_old(self, protein_conc):
        '''
        This is a wrapper function that simplifies the isothermal curve
        to two parameters (Ku and Kd)
        This function can be used to fit Kd and Ku
        '''
        def return_func(ligand_conc, Ku, Kd):
            return self.calculate_fitted_isothermal_old(ligand_conc, Ku, Kd, protein_conc)
        return return_func

    def calculate_fitted_isothermal_simple(self, protein_conc):
        '''
        This is a wrapper function that simplifies the isothermal curve
        to two parameters (Ku and Kd)
        This function can be used to fit Kd and Ku
        '''
        def return_func(ligand_conc, Ku, Kd):
            return self.calculate_fitted_isothermal(ligand_conc, Ku, Kd, protein_conc)
        return return_func
    
    def calculate_fitted_isothermal_onlyKd(self, protein_conc, Ku):
        '''
        This is a wrapper function that simplifies the isothermal curve
        to one parameter (Kd)
        This function can be used to fit Kd
        '''
        def return_func(ligand_conc, Kd):
            return self.calculate_fitted_isothermal(ligand_conc, Ku, Kd, protein_conc)
        return return_func        

    def select_fitting_params(self, verbose=True):
        '''
        Checks if temperature fit was done and selects respective fitting parameters
        If global fit exists, it will choose these parameters
        Otherwise it selects local fitting parameters
        Args:
            verbose: print output
        Returns:
        '''
        # Check if global fit with variable Cp was done
        if hasattr(self, 'global_fit_cp_params'):
            if verbose:
                print("Found global fit parameters with fitted Cp")
            fit_params = self.global_fit_cp_params
            fit_errors = self.global_fit_cp_errors
            fit_melting = 'global_cp'
        elif hasattr(self, 'global_fit_params'):
            if verbose:
                print("Found global fit parameters")
            fit_params = self.global_fit_params
            fit_errors = self.global_fit_errors
            fit_melting = 'global'
        elif hasattr(self, 'local_fit_params'):
            if verbose:
                print("Did not find global fitting parameters. Will use local ones")
            fit_params = self.local_fit_params
            fit_errors = self.local_fit_errors
            fit_melting = 'local'
        else:
            if verbose:
                print("Did not find any fitting parameters! Run fit_fluo_local and/or fit_fluo_global first.")
            return [], [], []
        # Update melting temperatures
        self.tms = fit_params[0]
        return fit_params, fit_errors, fit_melting

    def single_site(self, conc_lig, kd, conc_prot, t_bottom, t_top):
        '''
        Single-site model to fit apparent Kd from melting temperatures
        '''
        expr_sqrt = (conc_prot - kd - conc_lig + np.sqrt(
            ((conc_prot + conc_lig + kd) ** 2) - (4 * conc_prot * conc_lig))) / (2 * conc_prot)
        return t_bottom + ((t_top - t_bottom) * (1 - expr_sqrt))

    def single_site_kd(self, conc_prot):
        '''
        This is a wrapper function to obtain a single_site function with a
        fixed protein concentration for fitting the apparent Kd from melting
        temperatures

        Args:
            conc_prot: Protein concentration

        Returns: function
        '''

        def return_func(conc_lig, kd, t_bottom, t_top):
            return self.single_site(conc_lig, kd, conc_prot, t_bottom, t_top)

        return return_func

    def fluo_fit_r2(self):
        '''
        Calculate r2 between fitted fluorescence curve
        and experimental one
        This writes an entry fluo_r2 to the class instance
        '''
        # Obtain parameters
        fitting_params, fitting_errors, fit_melting = self.select_fitting_params()
        if len(fitting_params) == 0:
            print("Did not find any fitting parameters! Run fit_fluo_local() or fit_fluo_global() first.")
            return None
        # Initialize R2
        r2s = []
        # Loop through datasets and fill r2s
        num_datasets = len(self.concs)
        model = self.single_thermal_curve
        for c_index in range(num_datasets):
            fit = model(self.temps, *fitting_params[:, c_index])
            fluo = self.fluo[:, c_index]
            # Calculate R2
            r2 = r2_score(fluo, fit)
            r2s.append(r2)
        # Convert to array and save to instance
        r2s = np.array(r2s)
        self.fluo_r2 = r2s
        return r2s
            
    # New function to calculate whole binding curve not just one point
    def calculate_fitted_isothermal_2kds(self, conc_lig, Ku, Kd1, Kd2,  conc_prot):
        #Kd1 = Kd2 = Kd
        # Cooperativity (experimental)
        c = 1 
        B = Kd1 + Kd2 + (2*conc_prot - conc_lig) / c
        C = (conc_prot - conc_lig) * (Kd1+Kd2) + Kd1*Kd2 * (1+Ku)
        D = -conc_lig*Kd1*Kd2 * (1+Ku)
        # More variables
        p = B * c
        q = C * c
        r = D * c
        # Solutions
        m = (3*q - p**2)/3
        n = (2*p**3 - 9*p*q + 27*r)/27
        k = 0
        tk = 2 * np.sqrt(-m/3) * np.cos(1/3*np.arccos(3*n/(2*m)*np.sqrt(-3/m)) - 2*np.pi*k/3)
        A_free = tk - p/3
        B_free = Kd1*Kd2*(conc_lig - A_free) / ( Kd1+Kd2+2*A_free/c) / A_free
        BA = A_free*B_free / Kd2
        AB = A_free*B_free / Kd1
        ABA = AB * A_free / ( Kd2*c)
        # then use L_free to get fraction unfolded
        fit_fraction_unfolded = Ku*B_free / (Ku*B_free + B_free + BA + AB + ABA)
        return fit_fraction_unfolded

    def calculate_fitted_isothermal_2kds_simple(self, conc_prot):
        '''
        This is a wrapper function that simplifies the isothermal curve
        to only one parameter (Kd)
        This function can be used to fit Kd
        '''
        def return_func(conc_lig, Ku, Kd1, Kd2):
            return self.calculate_fitted_isothermal_2kds(conc_lig, Ku, Kd1, Kd2, conc_prot)
        return return_func

    def calculate_fitted_isothermal_2kds_simple2(self, conc_prot):
        '''
        This is a wrapper function that simplifies the isothermal curve
        to only one parameter (Kd)
        This function can be used to fit Kd
        '''
        def return_func(conc_lig, Ku, Kd):
            return self.calculate_fitted_isothermal_2kds(conc_lig, Ku, Kd, Kd, conc_prot)
        return return_func

    def hill_model(self, conc_lig, kd, n, t_bottom, t_top):
        '''
        Hill model to fit apparent Kd from melting temperatures
        Derived from Hill equation
        theta = ((conc_lig)**n)/ (kd+conc_lig**n)
        '''
        return t_bottom + (t_top - t_bottom) * ((conc_lig) ** n) / (kd + conc_lig ** n)

    def tm_model1_review(self, lig, Tm0, Kd, dH):
        '''
        This function uses the melting temperature in the absence of ligand
        dH of unfolding and the dissociation constant to calculate the Tm at
        different ligand concentrations
        '''
        # Constants
        R = 1.987E-3 # R in kcal/molK
        T0 = 273.15
        # Convert temperatures from °C to K
        Tm0_K = Tm0 + T0
        # Calculate expression for (Tm-Tm0)/Tm
        diffratio = R*Tm0_K / dH * np.log(1+lig/Kd)
        # Calculate Tm
        Tm = 1 / (1- diffratio) * Tm0_K - T0
        return Tm

    def tm_model1_review_fixTm0(self,Tm0):
        '''
        Helper function for fitting
        tm_model1_review with fixed Tm0
        '''
        def return_func(lig, Kd, dH):
            return self.tm_model1_review(lig, Tm0, Kd, dH)
        return return_func

    def correct_local_fit(self, fact=2):
        '''
        Correct outliers based on (local) fitting parameters
        This can be done before the global fits to have better
        starting values

        fact: Number of allowed median absolute deviations from median

        Returns:

        '''
        # # Get existing parameters
        # old_params, _, _ = self.select_fitting_params()
        # # End function if no previous parameters could be loaded
        # if len(old_params)==0:
        #     print("Could not load any parameters from previous fit!")
        #     print("Run fit_fluo_local() and/or fit_fluo_global() first.")
        #     return None
        # Use local parameters
        if hasattr(self, 'local_fit_params'):
            old_params = self.local_fit_params
        else:
            print("Could not find local fit parameters!")
            print("Run fit_fluo_local() first")
        # Use temperatures as measure
        values = old_params[0]
        # Determine median and mad
        med = np.median(values)
        mad = scipy.stats.median_absolute_deviation(values)
        print("Will use %.1f allowed median absolute deviations as outlier criterium" % fact)
        print("%.1f < Tm < %.1f C" % (med - fact*mad, med + fact*mad))

        # Check which values are in between fact *mads
        self.inds_in = (values > med - fact*mad) * (values < med + fact*mad)
        inds_out = np.invert(self.inds_in)
        self.outliers = np.argwhere(inds_out).squeeze().reshape((1,-1))
        print("Detected %i outliers in local fit" % (np.sum(inds_out)))
        if len(self.outliers[0]) == 0:
            return None
        else:
            # print(self.outliers)
            # Correct
            print("Will exchange these outliers in local fit with closest ligand concentration")
            for ind_out in np.nditer(self.outliers):
                diffs = np.abs(self.concs - self.concs[ind_out])
                # Change diff value to inf for the outliers (so it does not select any outlier)
                for j in self.outliers:
                    diffs[j] = np.inf
                # Get first minimum
                ind_in = np.argmin(diffs)
                print("Will exchange entry %i (conc=%.1E M, Tm=%.1f C) with entry %i (conc=%.1E M, Tm=%.1f C)" % (ind_out, self.concs[ind_out], values[ind_out], ind_in, self.concs[ind_in], values[ind_in]))
                # Exchange columns
                self.local_fit_params[:,ind_out] = self.local_fit_params[:, ind_in]
            # Adjust
            #self.fluo = self.fluo[:, self.inds_in]
            #self.concs = self.concs[self.inds_in]
            #self.local_fit_params = self.local_fit_params[:, self.inds_in]
            #self.local_fit_errorsg = self.local_fit_errors[:, self.inds_in]
            return None      
        
    def plot_derivative(self, save_fig=True, legend=False, colormap=False, legend_out=True, show_tms=False, linestyle='-', no_deriv=False, axs=[], map='viridis'):
        '''
        This function creates a plot with the intensities (first panel) and the
        first derivative (second panel)
        legend: show legend instead of color bar
        colormap: Use continous colormap, otherwise use discrete color cycle 
        legend_out: If true place legend next to plot (right side)
        no_deriv: do not plot second panel with derivatives
        axs: List of axes for plotting derivative
        map: colormap ('viridis' or 'jet')
        '''
        print("\nPlot fluorescence or scattering with derivatives")
        if len(axs)==0:
            if no_deriv:
                fig, axs = plt.subplots(1, sharex=True, figsize=[8,5])
                ax = axs
            else:
                fig, axs = plt.subplots(2, sharex=True, figsize=[8,5])
                ax = axs[0]
                ax2 = axs[1]
        else:
            if no_deriv:
                ax = axs[0]
            else:
                ax = axs[0]
                ax2 = axs[1]
            fig = axs[0].axes.get_figure()
        # Make sure that concs and fluo are sorted
        # This can not be the case if the data is directly
        # copied to the instance
        sort_ind = np.argsort(self.concs)
        self.concs = self.concs[sort_ind]
        self.fluo = self.fluo[:, sort_ind]
        if hasattr(self, 'sample_ID'):
            self.sample_ID = self.sample_ID[sort_ind]
        if hasattr(self, 'sample_comment'):
            self.sample_comment = self.sample_comment[sort_ind]
        # Calculate derivative and tms
        #self.calc_fluo_deriv()
        if not hasattr(self,'tms'):
            self.calc_fluo_deriv()
            self.tms_from_derivatives()
        # Define color map
        if map=='jet':
            cm = plt.cm.jet
        elif map=='viridis':
            cm = plt.cm.viridis
        else:
            cm = plt.cm.viridis
        cmap = iter(cm(np.linspace(0, 1, 1+len(np.unique(self.concs)))))
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=lh, title="Ligand concentration")
        ax.set_xlim([self.temps[0], self.temps[-1]])
        # Plot
        # Make sure that each conc. only has one color
        if self.which == 'Ratio':
            fact = 1000
        else:
            fact = 1
        prev_conc = -1
        lh = []  # Line handles
        colors, IDs, temp_temps = [], [], []
        sort_concs = np.sort(np.unique(self.concs))
        for i in range(len(self.concs)):
            # If sample_comment was already used before use same color
            if self.sample_comment[i] in IDs:
                # Get index of sample_comment
                j = int(np.argwhere(np.array(IDs) == self.sample_comment[i]).squeeze())
                color = colors[j]
                temp_temps.append(self.tms[i])
                ax.plot(self.temps, self.fluo[:, i], alpha=self.plot_alpha, lw=self.plot_lw, color=color, linestyle=linestyle)
                if not no_deriv:
                    ax2.plot(self.temps, self.fluo_deriv[:, i]*fact, alpha=self.plot_alpha, lw=self.plot_lw, color=color, linestyle=linestyle)
                #print(temp_temps)
                # Set temps back to empty list
                temp_temps = []
            # If sample_comment was not used before use new color
            else:
                # Define color from map
                if colormap:
                    color = cmap.__next__()
                    temp, = ax.plot(self.temps, self.fluo[:, i], label=self.sample_comment[i], alpha=self.plot_alpha, lw=self.plot_lw, color=color, linestyle=linestyle)
                    if not no_deriv:
                        temp, = ax2.plot(self.temps, self.fluo_deriv[:, i]*fact, label=self.sample_comment[i], alpha=self.plot_alpha, lw=self.plot_lw, color=color, linestyle=linestyle)
                # Or use normal color cycle
                else:
                    temp, = ax.plot(self.temps, self.fluo[:, i], label=self.sample_comment[i], alpha=self.plot_alpha, lw=self.plot_lw, linestyle=linestyle) #, color=color)
                    if not no_deriv:
                        temp, = ax2.plot(self.temps, self.fluo_deriv[:, i]*fact, label=self.sample_comment[i], alpha=self.plot_alpha, lw=self.plot_lw, linestyle=linestyle) #, color=color)
                IDs.append(self.sample_comment[i])
                colors.append(temp.get_color())
                temp_temps.append(self.tms[i])
                lh.append(temp)
                prev_conc = self.concs[i]
                # print("New conc %.2f" % self.concs[i])
            # Plot dotted lines for Tm if desired
            if show_tms:
                ax.axvline(self.tms[i], color=temp.get_color(), linestyle='--', zorder=-20, lw=self.plot_lw)
                if not no_deriv:
                    ax2.axvline(self.tms[i], color=temp.get_color(), linestyle='--', zorder=-20, lw=self.plot_lw)
            if not no_deriv:
                ax2.set_xlabel(r'Temperature / $^\circ$C')
            else:
                ax.set_xlabel(r'Temperature / $^\circ$C')
        if 'Ratio' in self.which:
            ax.set_ylabel('Fluo. ratio (350/330)')
        elif '350nm' in self.which:
            ax.set_ylabel('Fluo. intensity 350 nm')
        elif '330nm' in self.which:
            ax.set_ylabel('Fluo. intensity 330 nm')
        elif 'Scattering' in self.which:
            ax.set_ylabel('Scattering / a.u.')
        elif 'RFU' in self.which:
            ax.set_ylabel('RFU / a.u.')
        elif 'Cumulant Radius' in self.which:
            ax.set_ylabel('Cumulant Radius / nm')
            ax.set_yscale('log')
        elif 'Turbidity' in self.which:
            ax.set_ylabel('Turbidity counts')
        elif 'Simulated' in self.which:
            ax.set_ylabel('Simulated fluorescence intensity / a.u')
            ax.yaxis.set_ticks([])
            ax.text(np.min(self.temps)+2, np.max(self.fluo), '%s%% noise' % self.noise, va='top', ha='left', bbox=dict(facecolor='gray', alpha=0.5), fontsize=20)
        if not no_deriv:
            if fact==1:
                ax2.set_ylabel('1st deriv.')
            else:
                ax2.set_ylabel('1st deriv. * %i' % fact)
        # # Add box
        # boxmin = .98*np.min(self.fluo)
        # boxmax = 1.02*np.max(self.fluo)
        # frame1 = Rectangle((self.window[0],boxmin), self.window[1]-self.window[0], boxmax-boxmin, self.plot_lw=2, edgecolor='r', zorder=20, fill=0)
        # frame2 = Rectangle((self.window[0],boxmin), self.window[1]-self.window[0], boxmax-boxmin, self.plot_lw=2, edgecolor=None, zorder=-20, fill=1, self.plot_alpha=.5)
        # ax.add_patch(frame1)
        # ax.add_patch(frame2)

        if legend:
            if legend_out:
                ax.legend(bbox_to_anchor=(1.05, 1))
                if not no_deriv:
                    ax2.legend(bbox_to_anchor=(1.05, 1))
            else:
                ax.legend()
                if not no_deriv:
                    ax2.legend()
            #axs[0].legend(loc = 'lower center', bbox_to_anchor = (0,-0.1,1,1), bbox_transform = fig.transFigure )
            #axs[1].legend(loc = 'lower center', bbox_to_anchor = (0,-0.1,1,1), bbox_transform = fig.transFigure ))
        else:
            if colormap:
                # Set up color bar
                normalize = mcolors.LogNorm(vmin=np.min(self.concs[self.concs > 0]), vmax=np.max(self.concs))  # Or Normalize
                scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=plt.cm.viridis)
                scalarmappaple.set_array(self.concs[self.concs > 0])
                # Show color bar
                cbar = plt.colorbar(scalarmappaple, ax=ax)
                cbar.set_label('Ligand conc. / M', rotation=270)
                cbar.ax.get_yaxis().labelpad = 15
                if not no_deriv:
                    cbar2 = plt.colorbar(scalarmappaple, ax=ax2)
                    cbar2.set_label('Ligand conc. / M', rotation=270)
                    cbar2.ax.get_yaxis().labelpad = 15
        # Set window title
        fig.canvas.manager.set_window_title(self.which)
        # Save figure
        if save_fig:
            fig.tight_layout()
            # For simulations, include kd and noise in fn
            if hasattr(self, 'kd') and hasattr(self, 'noise'):
                fn_plot = '%s/signal_%s_Cp_%.0f_noise_%.0f_kd_%.0E_pconc_%.0E' % (self.folder, self.which.replace(' ',''), self.cp, self.noise, self.kd, self.pconc) 
            else:
                fn_plot = '%s/derivative_%s' % (self.folder, self.which.replace(' ',''), ) 
            for fn_suffix in ['.pdf']: #, '.png']:
                fig.savefig(fn_plot + fn_suffix)
                print("Saved figure as %s" % fn_plot)
        #fig.show()
        return fig, axs

if __name__ == '__main__':
    # This here is an example on how the program can be used

    # Parameters, adjust!
    folder = '/home/stephan/work/kd_dsf/200310_MHR1-2/01cmin'
    fn = '/200310_mhr_adpr_01cmin.xlsx'
    which = 'Ratio'
    window = [35, 68]
    # Create sample_concs.txt
    concs = []
    for i in range(14):
        concs.append(2000 * 0.5 ** i)
    concs = concs * 2
    concs += [0, 0]
    concs = np.array(concs) * 1E-6

    # Load into class
    test = DSF_binding()
    # test.load_xlsx(fn=folder+fn, concs=concs, which=which, window=window)
    test.load_txt(fn_fluo=folder + '/sample_fluo_Ratio.txt', fn_concs=folder + '/sample_concs.txt')
    test.isothermal_ts = [46, 48, 50, 52, 54, 56, 58, 60, 62]
    test.pconc = 5E-6
    test.cp = 0
    #test.plot_fluo()
    #test.fit_fluo_local()
    #test.fit_fluo_global()
    # test.fit_fluo_global_cp()
    #test.fit_isothermal()
    #test.plot_fit_isothermal()
    test.fit_tms()
    test.plot_tms()

    # %qtconsole
    # test.fluo_fit_local()

    # %qtconsole
