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
import os, glob, sys
import time
from scipy.optimize import curve_fit
# import shutil
from sklearn.metrics import r2_score
import pandas as pd
from matplotlib.patches import Rectangle
import ipdb
  
'''
This is a collection of functions for simulating data,
e.g. for testing the robustness of the method
'''
#print(os.path.abspath(__file__))
this_dir = os.path.dirname(os.path.abspath(__file__)) #.replace('/simulations_helpers.py','')
sys.path.insert(0, this_dir)
from DSF_fit import DSF_binding

def log_norm(x, vmin=1E-3, vmax=1E3):
    '''
    Log10 normalization between vmin and vmax
    vmin, vmax: Normalization range
    Return: Normalized value between 0 and 1 
    '''
    if (x < vmin) or (x > vmax):
        return None
    else:
        return np.log10(x/vmin) / np.log10(vmax/vmin) 
    
def lin_norm(x, vmin=1E-3, vmax=1E3):
    '''
    Linear normalization between vmin and vmax
    vmin, vmax: Normalization range
    Return: Normalized value between 0 and 1 
    '''
    if (x < vmin) or (x > vmax):
        return None
    else:
        return (x - vmin) / (vmax - vmin) 

def ku(T, Tm, dH, Cp):
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
    K = np.exp(-dG/(R*T_K))
    return K

def equilibr_const(T, Tr, dH, dS, dCp):
    '''
    General expression for the temperature 
    dependent equilibrium constant 
    This can be for instance be used to obtain KD(T)

    Input:
    T: Temperatures in °C
    Tr: Reference temperature in °C
    dH: Enthalpy change in kcal/mol at ref. temp
    dS: Entropy change in kcal/molK at ref. temp

    Returns: Equilibrium constant
    '''
    # Convert to K
    T_K = T +  273.15
    # Convert Tr to K
    Tr_K = Tr + 273.15
    # Gas constant in kcal/molK
    R = 1.987E-3 
    # Free enthalpy
    dG = dH + dCp*(T_K - Tr_K) - T_K * (dS + dCp * np.log(T_K/Tr_K))
    # Equilibrium constant
    K = np.exp(-dG/(R*T_K))
    return K

def get_fluo_from_fu(T, fu, mu, mf, bu, bf):
    '''
    Simulates fluorescence curve with given fu(T)

    Input:
    T:  temperatures
    fu: fraction unfolded vs. T
    mu: slope unfolded
    mf: slope folded
    bu: intercept unfolded
    bf: intercept folded

    Returns: Fluorescence signal vs. T
    '''
    T_K = T  + 273.15
    Y = fu * (mu*T_K + bu) + (1-fu) * (mf*T_K+bf)
    return Y


# # Use self.calculate_fitted_isothermal instead
# # if there is only one Kd (no temp dep)
def get_fu(Ku, Kd, L, pconc):
    '''
    Obtains fraction unfolded vs. temperatures

    Input:
    T: Temperatures
    KU0(T): Unfolding constant for pure protein (zero ligand conc.) vs. T
    Kd(T): Dissociation constant in M vs. T
    L: Ligand concentration in M

    Returns: fu(T)
    '''
    # First calculate complex concentration by solving quadratic expression
    p = pconc/(Ku+1) + L + Kd
    q = pconc*L/(Ku+1)
    complex_conc = .5*p - np.sqrt(0.25*p**2 - q)
    # Calculate Lfree
    L_free = L - complex_conc
    # Calculate fraction unfolded
    fu = 1 / (1 + 1/Ku *(1+L_free/Kd))   
    # Fraction unfolded
    fu = Ku/(Ku+L_free/Kd+1)
    return fu

def dG_total(T, KU, L, KD):
    '''
    Calculate total free enthalpy for unfolding
    This takes into account binding of ligand
      
    Requires:
    T: Temperatures in °C
    KU: Unfolding constant vs. T
    L: Free ligand conc. vs. T
    KD: Dissociation constant
    '''
    # Gas constant in kcal/molK
    R = 1.987E-3 
    # Convert to K
    T_K = T +  273.15
    # Free enthalpy
    dG = -R*T_K*np.log(KU) + R*T_K* np.log(1+L/KD)
    # Get pseudo KU
    KU_new = np.exp(-dG/(R*T_K))
    return dG, KU_new 


class KD_test_suite():
    '''
    This is a class to benchmark isothermal analysis
    for a combination of different initial KDs and
    protein concentrations
    This will loop over all combinations and do a 
    isothermal analysis with and without dCp optimization
    '''

    def __init__(self, folder='./', prefix='sim_kd*', cp=0):
        '''
        Requires
          folder: Folder that contains subfolders with sim data 
        '''
        self.fit_cp = False
        self.folder = folder
        self.prefix = prefix
        self.cp = cp
        # Create folder if not existing
        if not os.path.isdir(self.folder):
            print("Did not find folder %s! Will create it")
            os.makedirs(self.folder)
        return None

    def generate_data_fixed_Kd(self, Kds=np.logspace(-1,-10, num=10, base=10), pconcs=np.logspace(-1,-10, num=10, base=10), dHU=140, dCpU=8, Tm=45, mu=20, mf=4, bu=1, bf=2, T=np.linspace(20,80,101), noise_perc=5, plot_fluo=False):
        '''
        Generate data using a fixed Kd

        Required:
        Kds:  List of dissociation constants in M
        pconcs: List of protein concentrations
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
        # Load parameters into instance
        self.noise = noise_perc
        self.dHU = dHU
        self.dCpU = dCpU
        self.tm = Tm
        # Create data 
        for KD in Kds:
            for pconc in pconcs:
                print("Simulating KD=%.1EM and pconc=%.1EM" % (KD, pconc))
                test = DSF_binding(folder=self.folder)
                # Specify protein concentration
                test.pconc = pconc
                test.which = 'simulated'
                # Simulate curve
                test.simulate_curves_fixed_KD(KD=KD, dHU=dHU, dCpU=dCpU, Tm=Tm, mu=mu,
                                              mf=mf, bu=bu, bf=bf, T=T, noise_perc=noise_perc)
                if plot_fluo:
                    test.plot_fluo()
                    plt.close('all')
        print("Done simulating data")
        # Write meta data into file
        fs = "{:<30s} {:>10.1E}\n"
        with open(self.folder + '/simulation_parameters.txt', 'w') as f:
            f.write(fs.format('dHU / kcal/mol', dHU))
            f.write(fs.format('dCpU / kcal/molK', dCpU))
            f.write(fs.format('KD / M', KD))
            f.write(fs.format('Tm / °C', Tm))
            f.write(fs.format('mu (Slope unfolded)', mu))
            f.write(fs.format('mf (Slope folded)', mf))
            f.write(fs.format('bu (Intercept unfolded)', bu))
            f.write(fs.format('bf (Intercept folded)', bf))
            f.write(fs.format('noise_perc / %', noise_perc))
        return None

    def plot_fits(self, plot_fluo=False, plot_fluo_fit=False, plot_iso=True, isothermal_ts=[40,45,50]):
        '''

        '''
        # Search for subfolders in folder and extract
        # kd and protein concentration from folder name
        subf = glob.glob(self.folder + '/' + self.prefix)
        # Extract KDs and pconcs
        kds, pconcs = [], []
        for sub in subf:
            kd = sub.split('/')[-1].split('_')[2]
            pconc = sub.split('/')[-1].split('_')[4]
            #print("Pconc=%.1E" % float(pconc))
            kds.append(kd)
            pconcs.append(pconc)
        self.subf = subf
        self.kds = np.array(kds, dtype='float')
        self.pconcs = np.array(pconcs, dtype='float')

        # Only for this particular case
        self.isothermal_ts = isothermal_ts

        # Loop over subfolders 
        for i, folder in enumerate(self.subf):
            print("\nStarting dataset %i/%i: %s\n" % (i, len(self.subf), folder))
            # Initialize/reset variable to detect failed fit
            failed = False
            # Find files with fluo and concs
            fn_fluo = glob.glob(folder + '/sample_fluo*.txt')[0]
            fn_concs = glob.glob(folder + '/sample_conc*.txt')[0]
            # Initialize instance of DSF_binding
            test = DSF_binding()
            test.cp = self.cp
            test.pconc = self.pconcs[i]
            test.isothermal_ts = self.isothermal_ts
            test.load_txt(fn_fluo=fn_fluo, fn_concs=fn_concs, load_fit=True)
            if plot_fluo:
                test.plot_fluo()
                plt.close('all')
            # Do isothermal analysis
            test.fit_isothermal()
            # Plot isothermal
            if plot_iso:
                test.plot_fit_isothermal()
                # Only save figure
                plt.close('all') 
            #self.runs.append(test)
        return None
        
        
    def run(self, plot_fluo=False, plot_fluo_fit= False, plot_iso=False, fit_cp=False, isothermal_ts=[40, 45, 50]):
        '''
        Loop through self.subf and run DSF_fit in each of them
        '''
        self.fit_cp = fit_cp
        # Search for subfolders in folder and extract
        # kd and protein concentration from folder name
        subf = glob.glob(self.folder + '/' + self.prefix)
        # Extract KDs and pconcs
        kds, pconcs = [], []
        for sub in subf:
            kd = sub.split('/')[-1].split('_')[2]
            pconc = sub.split('/')[-1].split('_')[4]
            #print("Pconc=%.1E" % float(pconc))
            kds.append(kd)
            pconcs.append(pconc)
        self.subf = subf
        self.kds = np.array(kds, dtype='float')
        self.pconcs = np.array(pconcs, dtype='float')

        # Only for this particular case
        self.isothermal_ts = isothermal_ts
        # Start with this dCp
        cp = 0
        # Data collection via dictionary of dictionaries
        runs = {}
        for temp in self.isothermal_ts:
            runs[temp] = {'kd': [], 'ku': [], 'cp': [], 'fit_type': []}
        # Loop over subfolders 
        for i, folder in enumerate(self.subf):
            print("\nStarting dataset %i/%i: %s\n" % (i, len(self.subf), folder))
            # Initialize/reset variable to detect failed fit
            failed = False
            # Find files with fluo and concs
            fn_fluo = glob.glob(folder + '/sample_fluo*.txt')[0]
            fn_concs = glob.glob(folder + '/sample_conc*.txt')[0]
            # Initialize instance of DSF_binding
            test = DSF_binding()
            test.cp = cp
            test.pconc = self.pconcs[i]
            test.isothermal_ts = self.isothermal_ts
            test.load_txt(fn_fluo=fn_fluo, fn_concs=fn_concs, load_fit=False)
            if plot_fluo:
                test.plot_fluo()
                plt.close('all')
            test.fit_fluo_local()
            try:
                test.fit_fluo_global()
            except:
                print("Was not able to do global fit!! Will use local fit instead")
                failed = True
            # Fit dCp if desired
            if fit_cp:
                try:
                    test.fit_fluo_global_cp()
                except:
                    print("Was not able to do global fit!! Will use local fit instead")
                    failed = True
            # Plot fluo fit if desired
            if plot_fluo_fit:
                test.plot_fit_fluo()
                plt.close('all')
            # Do isothermal analysis
            test.fit_isothermal()
            # Plot isothermal
            if plot_iso:
                test.plot_fit_isothermal()
                # Only save figure
                plt.close('all') 
            #self.runs.append(test)
            # Collect data
            for j, temp in enumerate(isothermal_ts):
                runs[temp]['ku'].append(test.bind_params[j,0])
                runs[temp]['kd'].append(test.bind_params[j,1])
                runs[temp]['cp'].append(test.cp)
                # Mark local and global fits
                if failed:
                    runs[temp]['fit_type'].append('local')
                else:
                    runs[temp]['fit_type'].append('global')
        self.runs = runs
        # Save file in folder
        for j, temp in enumerate(self.isothermal_ts):
            if fit_cp:
                fn = '%s/results_%idegrees_fit_cp.npz' % (self.folder, temp)
            else:
                fn = '%s/results_%idegrees_fixed_cp_%.1f.npz' % (self.folder, temp, self.cp)
            np.savez(fn, ku=np.array(runs[temp]['ku']), kd=np.array(runs[temp]['kd']), cp=runs[temp]['cp'], fit_type=runs[temp]['fit_type'], real_kd=self.kds, pconcs = self.pconcs, folders = self.subf)
            print('Written output to %s' % fn)
        return None

    def load(self, fit_cp=False):
        '''
        This function loads the fitting data into 
        the instance
        
        fit_cp: Use results with fitted Cp
        '''
        self.fit_cp = fit_cp
        # Search for npz files
        if fit_cp:
            fns = glob.glob(self.folder + '/results*_fit_cp.npz')
        else:
            fns = glob.glob(self.folder + '/results*_fixed_cp_%.1f.npz' % self.cp)
        print("Found %i files" % len(fns))
        if len(fns) < 1:
            return None
        # Load files and fill instance
        runs = {}
        for fn in fns:
            print("Will now load %s" % fn)
            # Extract iso temp from fn
            T_iso = fn.split('/')[-1].split('_')[1].replace('degrees','')
            T_iso = int(T_iso)
            # Load
            res = np.load(fn)
            # Create dictionary entry
            runs[T_iso] = {'kd': res['kd'], 'ku': res['ku'], 'cp': res['cp'], 'fit_type': res['fit_type']}

        # Try to load metadata file
        fn_meta = self.folder + '/simulation_parameters.txt'
        if os.path.isfile(fn_meta):
            print("Found metadata file: %s" % fn_meta)
            self.simulation_parameters = {}
            with open(fn_meta, 'r') as f:
                for line in f:
                    key = line.split(' ')[0].strip()
                    value = float(line.split(' ')[-1].strip())
                    self.simulation_parameters[key] = value
            self.cp = self.simulation_parameters['dCpU']
            self.noise_perc = self.simulation_parameters['noise_perc']
            self.tm = self.simulation_parameters['Tm']
        else:
            print("Did not find metadata file %s" % fn_meta)
                                
        # Generate entries
        self.T_iso = np.array(runs.keys())
        self.pconcs = res['pconcs']
        self.kds = res['real_kd']
        self.runs = runs
        self.folders = res['folders']
           
    def plot_heatmap(self, show_cp=False, plot_borders=[], isothermal_ts=[40,45, 50], cmap=plt.cm.jet):
        '''
        This plots a heatmap with the deviations of KD
        To do: also add dCp 
        plot_borders:  Exclude data with deviations outside borders
        isothermal_ts: Which temperatures to show in analysis
        show_cp: Plot DeltaCp deviation instead of Kd deviation
        cmap: Colormap (e.g. jet, viridis, cividis, magma)
        '''

        # All iso_ts
        all_iso_ts = self.runs.keys()
       # Only show one temperature, since the fitted dCp is the same for all
        if show_cp:
            ts = [isothermal_ts[0]]
        else:
            ts = np.sort(isothermal_ts)
        
        # Create figure
        fig, axs = plt.subplots(len(ts) , figsize=[4, 8/3*len(ts)]) #, sharex=True)
        # If only one axis is created, create a list
        if len(ts) == 1:
            axs = [axs]

        # # Borders for plotting
        # if len(plot_borders) != 2:
        #     if show_cp:
        #         plot_borders = [self.cp-8, self.cp+8]
        #     else:
        #         plot_borders = [1E-3, 1E3]
        #         #plot_borders = [-100, 100]

        # Create colormap
        # shades = 201
        # color_map = plt.cm.jet
        # cmap = plt.cm.viridis
        #if show_cp:
        #    #cmap = color_map(np.linspace(0, 1, shades))
        #else:
        #    #cmap = color_map(np.linspace(0, 1, shades))
                
        # Determine xticks
        xmin, xmax = np.min(np.log10(self.kds)), np.max(np.log10(self.kds))
        xticks = np.logspace(xmin, xmax, 4)
        # Determine yticks
        ymin, ymax = np.min(np.log10(self.pconcs)), np.max(np.log10(self.pconcs))
        yticks = np.logspace(ymin, ymax, 4)
        
        # Loop through runs
        for i, iso_t in enumerate(ts):
            ax = axs[i]
            run = self.runs[iso_t]
            if not show_cp:
                deviations = run['kd']/self.kds #(run['kd']-self.kds)/self.kds*100
                if len(plot_borders) != 2:
                    plot_borders = [self.cp-8, self.cp+8]
            else:
                deviations = run['cp']
                if len(plot_borders) != 2:
                    plot_borders = [np.min(deviations), np.max(deviations)]
            cps = run['cp']
            for j in range(len(deviations)):
                if (deviations[j] < np.min(plot_borders)) or (deviations[j] > np.max(plot_borders)):
                    color = 'white' #'grey' #(0, 0, 1)
                    alpha = .5
                else:
                    #ind = int((deviations[j] - np.min(plot_borders))/np.abs(np.diff(plot_borders))*shades)
                    if show_cp:
                        color_ind = lin_norm(deviations[j], *plot_borders)
                    else:
                        color_ind = log_norm(deviations[j], vmin=plot_borders[0], vmax=plot_borders[1])
                    color = cmap(color_ind)
                    alpha = 1
                ax.loglog(self.kds[j], self.pconcs[j], 'o', color=color, alpha=alpha, markeredgecolor='k', markeredgewidth=1)
                # Plot values as text (for debugging)
                #ax.text(self.kds[j], self.pconcs[j], "%.0f" % deviations[j], ha='center', va='center')
            ax.loglog([np.min(self.kds), np.max(self.kds)], [np.min(self.pconcs), np.max(self.pconcs)], ':', zorder=-20)
            # Labels
            if iso_t == self.tm:
                ax.set_title('T$_\mathrm{iso}$=T$_m$=%.0f$^\circ$C' % iso_t)
            else:
                ax.set_title('T$_\mathrm{iso}$=%.0f$^\circ$C' % iso_t)
            # No title if deltaCp is shown
            if show_cp:
                ax.set_title('')
            ax.set_ylabel('[P]$_0$ / M')
            ax.set_xlabel('$K_d$ / M')
            # Set ticks
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            
        # Set up color bar
        if show_cp:
            normalize = mcolors.Normalize(vmin=np.min(plot_borders), vmax=np.max(plot_borders))  # Or Normalize or LogNorm
        else:
            normalize = mcolors.LogNorm(vmin=np.min(plot_borders), vmax=np.max(plot_borders))
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        #scalarmappaple.set_array(np.ones(len(deviations)))
        for ax in axs:
            cbar = plt.colorbar(scalarmappaple, ax=ax)
            if show_cp:
                cbar.set_label('$\Delta C_p$ / kcal/molK', rotation=270)
            else:
                cbar.set_label('$K_\mathrm{d,fitted}/K_\mathrm{d,true}$', rotation=270)
            cbar.ax.get_yaxis().labelpad = 15
            # Change ticks
            #cbar.set_ticks([-1, 0, 1])
            #cbar.set_ticklabels(['$3 \\times 10^{-1}$', '$1 \\times 10^{0}$', '$3 \\times 10^{0}$'])
            #ipdb.set_trace()
            #cbar.ax.set_yticklabels([])
            #cbar.set_ticklabels([]) #'.3',  '1', '3'])

        # Tight layout, take into account suptitle with rect
        if hasattr(self, 'simulation_parameters'):
            # Figure super title
            fig.suptitle('$\Delta C_p=$%.1f kcal/molK, %.0f%% noise' % (self.simulation_parameters['dCpU'], self.simulation_parameters['noise_perc']))
            fig.tight_layout(rect=[0, 0, 1, .95])
        else:
            fig.tight_layout()

        # Save figure
        if not show_cp:
            fig.savefig(self.folder + '/dCp_%.0f_noise_%i_fit_cp_%i_kd.pdf' % (self.simulation_parameters['dCpU'], self.simulation_parameters['noise_perc'], self.fit_cp))
        else:
            fig.savefig(self.folder +'/dCp_%.0f_noise_%i_fit_cp_%i_cp.pdf' % (self.simulation_parameters['dCpU'], self.simulation_parameters['noise_perc'], self.fit_cp))
        return cbar

    
    
