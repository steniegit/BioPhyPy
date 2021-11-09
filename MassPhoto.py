'''
Mass photometry class
'''

import h5py
import numpy as np
import pandas as pd
import scipy.signal as ssi
from scipy.optimize import curve_fit
from .helpers import *

class MP_data:
    '''
    Simple class to load refeyn eventsFitted.h5 files from Refeyn mass photometer
    '''
    def __init__(self, fn=''):
        self.fn = fn
        # Load data
        if fn != '':
            data = h5py.File(self.fn, 'r')
            # Initialize variables, Squeeze necessary for older datasets
            self.masses_kDa = np.array(data['masses_kDa']).squeeze()
            # Get number of counts
            self.n_counts = len(self.masses_kDa)
            self.n_binding = np.sum(self.masses_kDa>0)
            self.n_unbinding = np.sum(self.masses_kDa < 0)
        else:
            # Empty instance if no file name is given
            self.masses_kDa = np.empty(1)
            self.n_counts, self.n_binding, self.n_unbinding = 0, 0, 0
        # Set default processing/plotting parameters
        self.proc_pars = {'window': [0, 2000], 'bin_width': 4}
        self.plot_pars = {'xlim': [0, 2000], 'show_labels': True}
        self.fit_pars  = {'max_width': 50,
                          'weighted_width': 50,
                          'tol': 50,
                          'guess_pos': [66, 148, 480]}
        # Mock table
        self.fit_table = pd.DataFrame()
        return None

    def write_parameters(self, fn=''):
        '''
        Function to export processing/plotting parameters
        fn: Filename for text file
        '''
        ### Use json
        # Go through parameters and write to text file
        with open(fn, 'w') as f:
            f.write("### Processing and plotting parameters for %s\n" % self.fn)
            # Processing parameters
            self.write_dictionary(f, self.proc_pars, "Processing parameters")
            self.write_dictionary(f, self.plot_pars, "Plot parameters")
            self.write_dictionary(f, self.fit_pars, "Fitting parameters")
        print("Wrote parameters to %s" % fn)
        return None
            
    def write_dictionary(self, f, par_dict, header):
        '''
        Helper function to write dictionary to file f
        f: open file
        par_dict: dictionary of parameters
        header: Header in file (name of parameters)
        '''
        # Write header
        f.write('# ' + header + '\n\n')
        for key in par_dict.keys():
            f.write(key + '\t' + str(par_dict[key]) + '\n')
        f.write('\n')
        return None

    def load_parameters(self, fn):
        '''
        Load processing/plotting parameters from file
        
        '''
        return None
        
    def create_histo(self, bin_width=4):
        '''
        Creates histogram of masses
        '''
        # Get min and maximum value
        window = [np.floor(np.min(self.masses_kDa)), np.ceil(np.max(self.masses_kDa))]
        # Determine number of bins based on bin_width
        nbins = int((window[1] - window[0]) // bin_width)
        # Create histogram
        self.hist_counts, self.hist_bins = np.histogram(self.masses_kDa, range=window, bins=nbins)
        self.hist_mass = (self.hist_bins[1:] + self.hist_bins[:-1]) / 2.0
        # Write parameters to instance
        self.hist_centers = 0.5 * (self.hist_bins[:-1] + self.hist_bins[1:])
        self.hist_binwidth = bin_width
        self.hist_window = window
        self.hist_nbins = nbins
        self.hist_window = window
        return None

    def create_fit_table(self):
        '''
        Uses info in self.fit to generate a 
        pandas DataFrame that summarizes fit results
        '''
        if hasattr(self, 'fit'):
            # Create lists with fitting parameters
            # These are later used to create a pandas DataFrame
            list_pos, list_sigma, list_counts = [], [], []
            # Loop over entries in optimized parameters
            for i in range(int(len(self.popt)/3)):
                list_pos.append(self.popt[3*i])
                list_sigma.append(self.popt[3*i+2]/2/np.sqrt(2*np.log(2)))
                list_counts.append(np.trapz(self.fit[:,i+1], x=self.fit[:,0]) / np.diff(self.hist_mass)[0])
            # Create Pandas Dataframe
            self.fit_table = pd.DataFrame(data={'Position / kDa': list_pos,
                                                'Sigma / kDa': list_sigma,
                                                'Counts' : list_counts,
                                                'Counts / %': np.round(np.array(list_counts)/self.n_binding*100)}
                                          ) #.astype({'Position / kDa': int,
                                            #       'Sigma / kDa': int,
                                            #       'Counts' : int,
                                            #       'Counts / %': int})
        else:
            print('No fit results available')
        return None
    
    def plot_histo(self, plot_weights=False, xlim=[0, 2000], ylim=[], ax=None, show_labels=True):
        '''
        Plot histogram of data
        plot_weights: plot weights used for gaussian fits
        xlim: list with lower and upper x limit for plot
        show_label: Shows gaussian paramters for each component 
        '''
        # Create fig
        if ax==None:
            fig, ax = plt.subplots(1)
        else:
            fig = plt.gcf()
        # Plot it
        ax.bar(self.hist_centers, self.hist_counts, alpha=.5, width=self.hist_binwidth)
        ax.set_xlabel('Mass / kDa')
        ax.set_ylabel('Counts')
        # Plot fit if there
        if hasattr(self, 'fit'):
            #ax.plot(self.fit[:,0], self.fit[:,1:-1], linestyle='--', color='C1')
            ax.plot(self.fit[:,0], self.fit[:,-1], linestyle='-', color='C1', alpha=1)
            for i in range(int(len(self.popt)/3)):
                pos = self.popt[3*i]
                # Check if band is inside xlim, otherwise go to next loop
                if (pos < xlim[0]) or (pos > xlim[1]):
                    print("Yes: %i" % pos)
                    continue
                pos_err = self.fit_error[3*i]
                width = self.popt[3*i + 2 ]
                height = self.popt[3*i + 1 ]
                # ax.plot([self.fit[pos,-1], self.spec[pos,-1]], [self.spec[peak,1]+0.01*ylim[1], self.spec[peak,1]+0.05*ylim[1]], color='k')
                # Plot individual gaussians
                ax.plot(self.fit[:,0], self.fit[:,i+1] , color='C1', alpha=1, linestyle='--')
                # Determine area under curve
                auc = np.trapz(self.fit[:,i+1], x=self.fit[:,0]) / np.diff(self.hist_mass)[0]
                # Add label
                if show_labels:
                    ax.text(pos, height+0.05*np.max(self.hist_counts), "%.0f kDa\n$\sigma=%.0f\,$kDa\n%.0f$\,$counts \n(%.0f%%)" % (pos, width/2/np.sqrt(2*np.log(2)), auc, auc/self.n_binding*100), ha='center', va='bottom')
            if plot_weights:
                ax.plot(self.hist_mass, self.weights * np.max(self.hist_counts), color='k')

        # Set limits
        if len(xlim)==0:
            ax.set_xlim([0, np.max(self.hist_mass)])
        else:
            ax.set_xlim(xlim)
        if hasattr(self, 'fit'):
            # Increase ylim to have space for labels
            if show_labels:
                ax.set_ylim([0, np.max(self.hist_counts)*1.35])
        else:
            ax.set_ylim([0, np.max(self.hist_counts)*1.1])
        fig.tight_layout()
        # Get axis dimension
        x_border = ax.get_xlim()[1]
        y_border = ax.get_ylim()[1]
        # Print number of counts
        ax.text(x_border*.99, y_border*.99, "Total counts: %i\nBinding: %.0f%%\nUnbinding: %.0f%%" % (self.n_counts, self.n_binding/self.n_counts*100, self.n_unbinding/self.n_counts*100), va='top', ha='right')
        return fig
    
    def fit_histo(self, guess_pos=[], tol=100, max_width=200, weighted=False, weighted_width=200):
        '''
        Fit gaussians to histogram
        guess: list with guessed centers, defines the number of gaussians to be used, 
               if empty, it will use the maximum of the histogram as guess
        weighted: Use weights around start positions, this can be used if there is a broad background
        max_width: Maximum FWHM for fitted gaussians
        weighted_width: FWHM for weights
        '''
        # If no guess are taken, only fit one gaussian and use maximum in histogram as guess
        if len(guess_pos) == 0:
            #guess_pos = self.hist_mass[np.argmax(self.hist_counts)]
            #guess_amp = np.max(self.hist_counts)
            #fit_guess = (guess_pos, guess_amp, 50)
            #bounds = ((guess_pos-tol, 0, 0), (guess_pos+tol, np.max(self.hist_counts), max_width))
            print("No guess positions given (guess_pos)! Will try to find maxima.")
            peaks, info = ssi.find_peaks(self.hist_counts, prominence=0.1*np.max(self.hist_counts))
            guess_pos = self.hist_centers[peaks]
            guess_amp = self.hist_counts[peaks]
            print(guess_pos)
            print(guess_amp)
            if len(guess_pos) == 0:
                print("No starting values found with peak picker")
                return None
        else:
            # Get amplitude for each guess position
            guess_amp = []
            for pos in guess_pos:
                ind = np.argmin(np.abs(self.hist_mass - pos))
                guess_amp.append(self.hist_counts[ind])
        fit_guess = np.column_stack((np.array(guess_pos), np.array(guess_amp), np.array([.5*max_width]*len(guess_pos)))).flatten()
        lower_bounds = np.column_stack((np.array(guess_pos) - tol , np.array([0]*len(guess_pos)), np.array([0]*len(guess_pos)))).flatten()
        upper_bounds = np.column_stack((np.array(guess_pos) + tol , np.array([np.max(self.hist_counts)]*len(guess_pos)), np.array([max_width]*len(guess_pos)))).flatten()
        bounds = (tuple(lower_bounds), tuple(upper_bounds))
        # Determine function
        func = multi_gauss
        # Set weights
        if weighted:
            print("Will do weighted fit")
            #sigma_params = np.column_stack((np.array(guess_pos), np.array([1]*len(guess_pos)), np.array([weighted_width]*len(guess_pos)))).flatten()
            #sigma = func(self.hist_mass, *sigma_params)
            #sigma = (np.max(sigma) - sigma) + np.finfo(float).eps
            #self.weights = sigma
            sigma = delta(self.hist_mass, guess_pos, np.array([max_width]*len(guess_pos))) * np.max(self.hist_counts) + np.finfo(float).eps
        else:
            sigma = np.ones((len(self.hist_mass)))*np.finfo(float).eps
        # Write sigma to instance
        self.weights = sigma
        # Do fit
        self.popt, self.pcov = curve_fit(func, self.hist_mass, self.hist_counts, p0=fit_guess, bounds=bounds, sigma=sigma)  #, method='dogbox', maxfev=1E5)
        print(self.popt)
        # Create fit and individual gaussians for plotting
        # Finer grid
        x = np.linspace(np.min(self.hist_mass), np.max(self.hist_mass), 1000)
        single_gauss = []
        for i in range(0, len(self.popt), 3):
            ctr = self.popt[i]
            amp = self.popt[i+1]
            wid = self.popt[i+2]
            single_gauss.append(func(x, ctr, amp, wid))
        # Sum of all
        fit_sum = func(x, *self.popt)
        # Create one array for all
        self.fit = np.column_stack((x, np.array(single_gauss).T, fit_sum))
        # Errors
        self.fit_error = np.sqrt(np.diag(self.pcov))
        # Create fit table
        self.create_fit_table()
        return None
