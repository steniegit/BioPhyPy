'''
Mass photometry class
'''

import h5py
import numpy as np
import pandas as pd
import scipy.signal as ssi
from scipy.optimize import curve_fit
from .helpers import *

# To do
# Catch exception if masses kDa are not in file


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
            self.contrasts = np.array(data['contrasts']).squeeze()
            # Get number of counts
            self.n_counts = len(self.contrasts)
            self.n_binding = np.sum(self.contrasts<0)
            self.n_unbinding = np.sum(self.contrasts>0)
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
        self.fit_type  = 'None'
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
        
    def create_histo(self, bin_width=4, bin_width_contrasts=0.002):
        '''
        Creates histogram of masses
        '''
        # If masses_kDa exists
        if hasattr(self, 'masses_kDa'):           
            # Get min and maximum value
            window = [np.floor(np.min(self.masses_kDa)), np.ceil(np.max(self.masses_kDa))]
            # Determine number of bins based on bin_width
            nbins = int((window[1] - window[0]) // bin_width)
            # Create histogram
            self.hist_counts, self.hist_bins = np.histogram(self.masses_kDa, range=window, bins=nbins)
            # Write parameters to instance
            self.hist_centers = 0.5 * (self.hist_bins[:-1] + self.hist_bins[1:])
            self.hist_binwidth = bin_width
            self.hist_window = window
            self.hist_nbins = nbins
            self.hist_window = window
        # For contrast values do the same
        window_contrasts = [np.floor(np.min(self.contrasts)), np.ceil(np.max(self.contrasts))]
        nbins_contrasts = int((window_contrasts[1] - window_contrasts[0]) // bin_width_contrasts)
        # Create histogram for contrasts
        self.hist_counts_contrasts, self.hist_bins_contrasts = np.histogram(self.contrasts, range=window_contrasts, bins=nbins_contrasts)
        # Write parameters to instance
        self.hist_centers_contrasts = 0.5 * (self.hist_bins_contrasts[:-1] + self.hist_bins_contrasts[1:])
        self.hist_binwidth_contrasts = bin_width_contrasts
        self.hist_window_contrasts = window_contrasts
        self.hist_nbins_contrasts = nbins_contrasts
        self.hist_window_contrasts = window_contrasts
        return None

    def create_fit_table(self):
        '''
        Uses info in self.fit to generate a 
        pandas DataFrame that summarizes fit results
        '''
        # Check if fit data is available
        if not (hasattr(self, 'fit_contrasts') or hasattr(self, 'fit_masses')):
            print('No fit results available')
        # Load data from fits
        if hasattr(self, 'fit_contrasts'):
            fit, error, popt, pcov, weights = self.fit_contrasts.values()
            hist_centers = self.hist_centers_contrasts
            print(popt)
            # Create lists with fitting parameters
            # These are later used to create a pandas DataFrame
            list_pos, list_sigma, list_counts = [], [], []
            # Loop over entries in optimized parameters
            for i in range(int(len(popt)/3)):
                list_pos.append(popt[3*i])
                list_sigma.append(popt[3*i+2]/2/np.sqrt(2*np.log(2)))
                list_counts.append(np.trapz(fit[:,i+1], x=fit[:,0]) / np.diff(hist_centers)[0])
            self.fit_table_contrasts = pd.DataFrame(data={'Position': list_pos,
                                                          'Sigma': list_sigma,
                                                          'Counts' : list_counts,
                                                          'Counts / %': np.round(np.array(list_counts)/self.n_binding*100)})
            print("Created fit_table_contrasts")           
        if hasattr(self, 'fit_masses'):
            fit, error, popt, pcov, weights = self.fit_masses.values()
            hist_centers = self.hist_centers
            # Create lists with fitting parameters
            # These are later used to create a pandas DataFrame
            list_pos, list_sigma, list_counts = [], [], []
            # Loop over entries in optimized parameters
            for i in range(int(len(popt)/3)):
                list_pos.append(popt[3*i])
                list_sigma.append(popt[3*i+2]/2/np.sqrt(2*np.log(2)))
                list_counts.append(np.trapz(fit[:,i+1], x=fit[:,0]) / np.diff(self.hist_centers)[0])
            self.fit_table_masses = pd.DataFrame(data={'Position / kDa': list_pos,
                                                       'Sigma / kDa': list_sigma,
                                                       'Counts' : list_counts,
                                                       'Counts / %': np.round(np.array(list_counts)/self.n_binding*100)})
        return None
    
    def plot_histo(self, plot_weights=False, xlim=[0, 2000], ylim=[], ax=None, show_labels=True, contrasts=False):
        '''
        Plot histogram of data
        plot_weights: plot weights used for gaussian fits
        xlim: list with lower and upper x limit for plot
        show_label: Shows gaussian paramters for each component 
        contrast: Show contrast instead of masses in kDa
        '''
        # Set right quantity based on choice in contrasts
        if contrasts:
            counts = self.hist_counts_contrasts
            centers = self.hist_centers_contrasts
            binwidth = self.hist_binwidth_contrasts
            # Check if fit results are available
            if hasattr(self, 'fit_contrasts'):
                fit, error, popt, pcov, weights = self.fit_contrasts.values()
                plot_fit = True
            else:
                plot_fit = False
        else:
            counts = self.hist_counts
            centers = self.hist_centers
            binwidth = self.hist_binwidth
            # Check if fit results are available
            if hasattr(self, 'fit_masses'):
                fit, error, popt, pcov, weights = self.fit_masses.values()
                plot_fit = True
            else:
                plot_fit = False
        # Create fig
        if ax==None:
            fig, ax = plt.subplots(1)
        else:
            fig = plt.gcf()
        # Plot it
        if contrasts:
            ax.set_xlabel('Contrast')
        else:
            ax.set_xlabel('Mass / kDa')
        ax.set_ylabel('Counts')
        # Plot histogram
        ax.bar(centers, counts, alpha=.5, width=binwidth)

        # Check and set xlim
        # Sanity check for xlim if contrasts are used
        xlim = np.sort(xlim)
        if contrasts:
            # Set xlim after sanity check            
            if np.max(xlim) > 1:
                # Contrast cannot be beyond 2
                xlim = [np.min(centers), np.max(centers)]
            ax.set_xlim(xlim)                    
            # Invert xaxis
            ax.invert_xaxis()
        else:
            ax.set_xlim(xlim)       
        # Set xlim

        # Determine indices
        lower_ind = np.argmin(np.abs(centers-xlim[0]))
        upper_ind = np.argmin(np.abs(centers-xlim[1]))
        inds = np.sort([lower_ind, upper_ind])
        # Determine ylim
        if (hasattr(self, 'fit_masses') or hasattr(self, 'fit_contrasts')):
            # Increase ylim to have space for labels
            if show_labels:
                ax.set_ylim([0, np.max(counts[inds[0]:inds[1]])*1.35])
            else:
                ax.set_ylim([0, np.max(counts[inds[0]:inds[1]])*1.1])
        # And also fits (if available)
        if plot_fit:
            #ax.plot(fit[:,0], fit[:,1:-1], linestyle='--', color='C1')
            ax.plot(fit[:,0], fit[:,-1], linestyle='-', color='C1', alpha=1)
            for i in range(int(len(popt)/3)):
                pos = popt[3*i]
                # Check if band is inside xlim, otherwise go to next loop
                if (pos < xlim[0]) or (pos > xlim[1]):
                    continue
                pos_err = fit[3*i]
                width = popt[3*i + 2 ]
                height = popt[3*i + 1 ]
                # ax.plot([self.fit[pos,-1], self.spec[pos,-1]], [self.spec[peak,1]+0.01*ylim[1], self.spec[peak,1]+0.05*ylim[1]], color='k')
                # Plot individual gaussians
                ax.plot(fit[:,0], fit[:,i+1] , color='C1', alpha=1, linestyle='--')
                # Determine area under curve
                auc = np.trapz(fit[:,i+1], x=fit[:,0]) / np.diff(centers)[0]
                # Add label
                if show_labels:
                    if contrasts:
                        text_label = "%.2E \n$\sigma=%.2E\,$\n%.0f$\,$counts \n(%.0f%%)" % (pos, width/2/np.sqrt(2*np.log(2)), auc, auc/self.n_binding*100)
                    else:
                        text_label = "%.0f kDa\n$\sigma=%.0f\,$kDa\n%.0f$\,$counts \n(%.0f%%)" % (pos, width/2/np.sqrt(2*np.log(2)), auc, auc/self.n_binding*100)
                    ax.text(pos, height+0.05*np.max(counts), text_label , ha='center', va='bottom')
            if plot_weights:
                ax.plot(hist_centers, weights * np.max(counts), color='k')
        fig.tight_layout()
        # Get axis dimension
        x_border = ax.get_xlim()[1]
        y_border = ax.get_ylim()[1]
        # Print number of counts
        ax.text(x_border*.99, y_border*.99, "Total counts: %i\nBinding: %.0f%%\nUnbinding: %.0f%%" % (self.n_counts, self.n_binding/self.n_counts*100, self.n_unbinding/self.n_counts*100), va='top', ha='right')
        return fig, ax
    
    def fit_histo(self, xlim=[], guess_pos=[], tol=100, tol_contrasts = 0.05, max_width=200, max_width_contrasts=0.005, weighted=False, weighted_width=200, weighted_width_contrasts=0.005, contrasts=False, cutoff=0):
        '''
        Fit gaussians to histogram
        xlim: fit range
        guess: list with guessed centers, defines the number of gaussians to be used, 
               if empty, it will use the maximum of the histogram as guess
        weighted: Use weights around start positions, this can be used if there is a broad background
        max_width: Maximum FWHM for fitted gaussians
        weighted_width: FWHM for weights
        contrasts: fit contrasts instead of masses
        cutoff: lower cutoff for gaussian function. E.g. for 'RefeynOne' the lower cutoff is 40 kDa (for masses), for 'RefeynTwo' it is 30 kDa. If not sure select 0
        '''
        # Depending on contrasts select the right quantity
        if contrasts:
            centers = self.hist_centers_contrasts
            counts = self.hist_counts_contrasts
            # Use tolerance for contrasts
            tol = tol_contrasts
            max_width = max_width_contrasts
            weighted_width = weighted_width_contrasts
        else:
            # Load data for masses/kDa
            centers = self.hist_centers
            counts = self.hist_counts
        # Determine indices for xlim
        if len(xlim) == 0:
            ind_range = [0, len(centers)]
        else:
            xlim = np.sort(xlim)
            ind_range = []
            ind_range.append(np.argmin(np.abs(centers - xlim[0])))
            ind_range.append(np.argmin(np.abs(centers - xlim[1])))
            ind_range = np.sort(ind_range)
        # Adjust centers and contrasts
        centers = centers[ind_range[0]:ind_range[1]]
        counts  = counts[ind_range[0]:ind_range[1]] 
        # If no guess are taken, only fit one gaussian and use maximum in histogram as guess
        if len(guess_pos) == 0:
            print("No guess positions given (guess_pos)! Will try to find maxima.")
            peaks, info = ssi.find_peaks(counts, prominence=0.1*np.max(counts))
            guess_pos = centers[peaks]
            guess_amp = counts[peaks]
            print(guess_pos)
            print(guess_amp)
            if len(guess_pos) == 0:
                print("No starting values found with peak picker")
                return None
        else:
            # Get amplitude for each guess position
            guess_amp = []
            for pos in guess_pos:
                ind = np.argmin(np.abs(centers - pos))
                guess_amp.append(counts[ind])
        fit_guess = np.column_stack((np.array(guess_pos), np.array(guess_amp), np.array([.5*max_width]*len(guess_pos)))).flatten()
        lower_bounds = np.column_stack((np.array(guess_pos) - tol , np.array([0]*len(guess_pos)), np.array([0]*len(guess_pos)))).flatten()
        upper_bounds = np.column_stack((np.array(guess_pos) + tol , np.array([np.max(counts)]*len(guess_pos)), np.array([max_width]*len(guess_pos)))).flatten()
        bounds = (tuple(lower_bounds), tuple(upper_bounds))
        # Determine function
        #func = multi_gauss
        func = trunc_gauss_fixed(cutoff)
        # Set weights
        if weighted:
            print("Will do weighted fit")
            sigma = delta(centers, guess_pos, np.array([max_width]*len(guess_pos))) * np.max(counts) + np.finfo(float).eps
        else:
            sigma = np.ones((len(centers)))*np.finfo(float).eps
        # Write sigma to instance
        weights = sigma
        # Do fit
        popt, pcov = curve_fit(func, centers, counts, p0=fit_guess, bounds=bounds, sigma=sigma)  #, method='dogbox', maxfev=1E5)
        # Create fit and individual gaussians for plotting
        # Finer grid
        x = np.linspace(np.min(centers), np.max(centers), 1000)
        single_gauss = []
        for i in range(0, len(popt), 3):
            ctr = popt[i]
            amp = popt[i+1]
            wid = popt[i+2]
            single_gauss.append(func(x, ctr, amp, wid))
        # Sum of all
        fit_sum = func(x, *popt)
        # Create one array for all
        fit = np.column_stack((x, np.array(single_gauss).T, fit_sum))
        # Errors
        fit_error = np.sqrt(np.diag(pcov))
        # Collect everything in a dictionary
        fit_results = {'fit': np.column_stack((x, np.array(single_gauss).T, fit_sum)),
                       'error': np.sqrt(np.diag(pcov)),
                       'popt': popt,
                       'pcov': pcov,
                       'weights': weights}
        # Set fit type
        if contrasts:
            #self.fit_type = 'contrasts'
            self.fit_contrasts = fit_results
        else:
            #self.fit_type = 'masses'
            self.fit_masses = fit_results
        # Create fit table
        self.create_fit_table()
        return None
