'''
Mass photometry class
'''

import h5py
import numpy as np
import pandas as pd
import scipy.signal as ssi
from scipy.optimize import curve_fit
from matplotlib import gridspec
from matplotlib.patches import Circle
from .helpers import *

# To do
# Catch exception if masses kDa are not in file


class MP_data:
    '''
    Simple class to load refeyn eventsFitted.h5 files from Refeyn mass photometer
    '''
    def __init__(self, fn='', mp_fn=''):
        self.fn = fn
        # Filename for movie, this is optional
        self.mp_fn = mp_fn
        # Load data
        if fn != '':
            data = h5py.File(self.fn, 'r')
            # Initialize variables, Squeeze necessary for older datasets
            if 'masses_kDa' in data.keys():
                self.masses_kDa = np.array(data['masses_kDa']).squeeze()
            elif 'calibrated_values' in data.keys():
                self.masses_kDa = np.array(data['calibrated_values']).squeeze()
            else:
                print("Could neither find calibrated_values nor masses_kDa! Will only load contrasts.")
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

    def analyze_movie(self, frame='most', threshold_big=1000, ratiometric_size=10, frame_range=2, image_scale=1, image_xoffset=0, image_yoffset=0):
        '''
        It is optional to use the original movie file
        This can be used to show an inlet in the plots
        frame :        frame with 'most' counts, 'largest' counts, or frame number (int)
        threshold_big: threshold for frame='largest'
        frame_range:   Number of frames around target frame are used for plotting events
        '''
        # Check if filename is specified
        if self.mp_fn == '':
            print("No mp_fn defined!")
            return None
        # Check if file exists
        if not os.path.isfile(self.mp_fn):
            print("Could not find file %s" % self.mp_fn)
            return None
        # Save parameters in instance
        self.image_scale = image_scale
        self.image_xoffset = image_xoffset
        self.image_yoffset = image_yoffset
        # Load video file
        video = h5py.File(self.mp_fn)
        video = np.array(video['movie']['frame']).astype('int16')
        # Change sign
        video = video[:]*-1
        # Load fitted events to obtain more parameters
        data = h5py.File(self.fn, 'r')
        # Create dataframe
        events = pd.DataFrame({'frame_ind': data['frame_indices'], 
                               'contrasts': data['contrasts'], 
                               'kDa': self.masses_kDa, # used to be  test['masses_kDa']
                               'x_coords': data['x_coords'], 
                               'y_coords': data['y_coords']})
        self.events_temp = events
        # Detect frame with most or largest masses
        frames, counts, big_counts = [], [], []
        frame_nos = np.unique(events['frame_ind'])
        for frame_no in frame_nos:
            frames.append(frame_no)
            temp_events = events[events['frame_ind']==frame_no]
            counts.append(len(temp_events))
            # Only show big particles
            temp_events = temp_events[temp_events['kDa'] > threshold_big]
            big_counts.append(len(temp_events))
        # Select frame number
        if frame == 'most':
            self.frame_no = frames[np.argmax(counts)]
        elif frame == 'largest':
            self.frame_no = frames[np.argmax(big_counts)]
        else:
            self.frame_no = frame
        self.frame_no = int(self.frame_no)
        # Obtain ratiometric contrast
        self.dra = np.mean(video[self.frame_no+1:self.frame_no+1+ratiometric_size//2], axis=0) / np.mean(video[self.frame_no-ratiometric_size//2:self.frame_no], axis=0) - 1
        # Only obtain events in frame range
        print(self.frame_no)
        self.events = events[events['frame_ind'].between(self.frame_no-frame_range, self.frame_no+frame_range)]
        print(self.events)
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
        
    def create_histo(self, bin_width=4, bin_width_contrasts=0.0002, only_masses=False):
        '''
        Creates histogram of masses
        '''
        # Write parameters to instance
        self.hist_binwidth= bin_width
        # If masses_kDa exists
        if hasattr(self, 'masses_kDa'):
            # Also check if it contains nans
            if np.isnan(self.masses_kDa).any():
                print("Detected nans in masses_kDa")
            else:
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
                self.hist_binwidth_contrasts = bin_width_contrasts
        # Do also for contrasts
        if not only_masses:
            # For contrast values do the same
            window_contrasts = [np.floor(np.min(self.contrasts)), np.ceil(np.max(self.contrasts))]
            nbins_contrasts = int((window_contrasts[1] - window_contrasts[0]) // bin_width_contrasts)
            # Create histogram for contrasts
            self.hist_counts_contrasts, self.hist_bins_contrasts = np.histogram(self.contrasts, range=window_contrasts, bins=nbins_contrasts)
            # Write parameters to instance
            self.hist_centers_contrasts = 0.5 * (self.hist_bins_contrasts[:-1] + self.hist_bins_contrasts[1:])
            self.hist_window_contrasts = window_contrasts
            self.hist_nbins_contrasts = nbins_contrasts
            self.hist_window_contrasts = window_contrasts
            self.hist_binwidth_contrasts = bin_width_contrasts
        return None

    def calibrate(self, calib_stand=['NM1','NM2','NM3'], plot=False):
        ''' 
        Calibration based on contrasts histogram
        You need to have run create_histo and fit_histo(contrasts=True) before
        calib_standard: List of calibration standards
                        Masses can be defined by strings: 'NM1', 'NM2', 'NM3', 'NM4'
                        Or it can be defined by floats: e.g. 66, 146, 480
        '''
        # Dictionary with masses
        mass_transl = {'NM1': 66, 'NM2': 146, 'NM3': 480, 'NM4': 1048}
        # Convert calib_standard in list of floats
        calib_floats = []
        for stand in calib_stand:
            if isinstance(stand, float):
                calib_floats.append(stand)
            else:
                try:
                    calib_floats.append(mass_transl[stand])
                except:
                    print("Could not find %s in list of calibrants" % stand)
                    return None
        # Convert to array
        calib_stand = np.array(calib_floats)
        # Sort in reversed order
        calib_stand.sort()
        # Check if fitted contrast histo is available
        if not hasattr(self, 'fit_contrasts'):
            print("No fitted contrasts available")
            print("Please run fit_histo(contrasts=True) first")
            return None
        # Calibration points from fitting
        calib_points = self.fit_table_contrasts['Position']
        # Check if number of fitted gaussians fits to number of calibrants
        if not (len(calib_points) == len(calib_stand)):
            print("%i calibration standards given, but %i gaussians fitted to contrasts!" % (len(calib_stand), len(calib_points)))
        # Now use these to calibrate system
        # First order polynomial fit
        params = np.polyfit(calib_points, calib_stand, 1)
        # Calculate R2
        calc = np.polyval(params, calib_points)
        r2 = r_sq(calib_stand, calc)
        print("R^2=%.4f" % r2)
        # Calculate max error
        max_error = np.max((np.abs(calc-calib_stand))/calib_stand*100)
        #print(max_error)
        #print("Max error: %f.2" % max_error)
        # Write info to instance
        self.calib = {'standards':  calib_stand,
                      'exp_points': calib_points,
                      'fit_params': params,
                      'fit_r2': r2,
                      'calib_maxerror': max_error}
        self.calib_stand = calib_stand
        self.calib_points = calib_points
        self.calib_params = params
        self.calib_r2 = r2
        self.calib_maxerror = max_error
        # Plot calibration
        if plot:
            self.plot_calibration()
        # Calibrate masses
        self.calibrate_masses()
        self.create_histo(only_masses=True)
        return None

    def plot_calibration(self, ax=None):
        '''
        Plot calibration line
        '''
        # Check if calibration data is available
        if not hasattr(self, 'calib'):
            print("No calibration data available. Run calibrate first.")
            return None
        # Read calib data from instance
        calib_stand = self.calib['standards']
        calib_points = self.calib['exp_points']
        max_error = self.calib['calib_maxerror']
        r2 = self.calib['fit_r2']
        
        # Create fig if no axis is given
        if ax==None:
            fig, ax = plt.subplots(1)
        else:
            fig = plt.gcf()

        # Do reverse fit for function of contrast
        params_rev = np.polyfit(calib_stand, calib_points, 1)
        # Plot points
        ax.plot(calib_stand, calib_points, 'o', label='Calibration standard')
        # Plot calibration line
        two_masses = np.array([np.min(calib_stand)-50, np.max(calib_stand)+50])
        cal_label = 'Gradient: %.1E kDa\nIntercept: %.1E\nMax error: %.1f%%' % (params_rev[0], params_rev[1], max_error)
        ax.plot(two_masses, np.polyval(params_rev,two_masses), '--', zorder=-20, label=cal_label)
        # Lables
        ax.set_xlabel('Molecular mass / kDa')
        ax.set_ylabel('Contrast')
        ax.set_title('Calibration (R$^2=%.4f$)' % r2)
        ax.legend()
        fig.tight_layout()
        return ax
    
    def calibrate_masses(self):
        '''
        Function to calibrate masses from
        contrasts after changing/setting the 
        calibration parameters self.calib_para
        '''
        if hasattr(self, 'calib_params'):
            # Convert contrasts to masses
            self.masses_kDa = np.polyval(self.calib_params, self.contrasts)
            print("Successfully calibrated masses/kDa and created mass histogram")
        else:
            print("No calibration parameters found!")
            print("Run calibrate first")
        return None

    def calibration_export(self, fn='calib.txt'):
        '''
        Export calibration parameters and R2 to text file
        fn: File name of exported file, will be copied to subfolder (location of h5 file)
        '''
        if hasattr(self, 'calib_params'):
            #fn_out = self.fn.replace('eventsFitted.h5','') + fn
            # Concatenate arrays with parameters with R2
            out_array = np.concatenate((self.calib_params, np.array(self.calib_r2).reshape(1)))
            np.savetxt(fn, out_array)
            print("Saved parameters and R2 in %s" % fn)
        return None

    def calibration_import(self, fn='calib.txt'):
        '''
        Reads calibration parameters and R2 from text file
        fn: file name
        '''
        # Check if file exists
        if not os.path.isfile(fn):
            print("File %s not found!" % fn)
            print("No data imported.")
            return None
        # Read file
        in_array = np.genfromtxt(fn)
        # Fill instance
        # First two are calibration parameters
        self.calib_params = in_array[:2]
        # Last one is R2
        self.calib_r2 = in_array[-1]
        print("Calibration with R^2=%.4f successfully imported" % self.calib_r2)
        self.calibrate_masses()
        
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
            # Create lists with fitting parameters
            # These are later used to create a pandas DataFrame
            list_pos, list_sigma, list_counts = [], [], []
            # Loop over entries in optimized parameters
            for i in range(int(len(popt)/3)):
                list_pos.append(popt[3*i])
                list_sigma.append(popt[3*i+2])
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
                list_sigma.append(popt[3*i+2])
                list_counts.append(np.trapz(fit[:,i+1], x=fit[:,0]) / np.diff(self.hist_centers)[0])
            self.fit_table_masses = pd.DataFrame(data={'Position / kDa': list_pos,
                                                       'Sigma / kDa': list_sigma,
                                                       'Counts' : list_counts,
                                                       'Counts / %': np.round(np.array(list_counts)/self.n_binding*100)})
            print("Created fit_table_masses")
        return None
    
    def plot_histo(self, plot_weights=False, xlim=[0, 2000], ylim=[], ax=None, show_labels=True, contrasts=False, short_labels=False, show_counts=True, counts_pos='right', hist_fc='', hist_ec='', hist_label=''):
        '''
        Plot histogram of data
        plot_weights: plot weights used for gaussian fits
        xlim: list with lower and upper x limit for plot
        show_label: Shows gaussian paramters for each component 
        contrast: Show contrast instead of masses in kDa
        show_counts: Print number of binding/unbinding events
        counts_pos: 'left' or 'right' position for count numbers
        '''
        # Set right quantity based on choice in contrasts
        if contrasts:
            counts = self.hist_counts_contrasts
            centers = self.hist_centers_contrasts
            binwidth = self.hist_binwidth_contrasts
            print(binwidth)
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
        # Compile plot parameters in dictionary
        hist_params ={}
        if len(hist_ec) > 0:
            hist_params['ec'] =  hist_ec
        if len(hist_fc) > 0:
            hist_params['ec'] =  hist_fc
        if len(hist_label) > 0:
            hist_params['label'] = hist_label
        # Plot histogram
        if len(hist_label) > 0:
            ax.bar(centers, counts, alpha=.5, width=binwidth, label=hist_label)
        else:
            ax.bar(centers, counts, alpha=.5, width=binwidth)

        # Check and set xlim
        # Sanity check for xlim if contrasts are used
        xlim = np.sort(xlim)
        if contrasts:
            # Set xlim after sanity check            
            if np.max(xlim) > 1:
                # Contrast cannot be beyond 2
                xlim = [np.min(centers), np.max(centers)]
                xlim = [-.1, 0.05] 
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
        # Override if ylim is set
        if len(ylim)==2:
            ax.set_ylim(ylim)
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
                        if short_labels:
                            text_label = "%.2E" % pos
                        else:
                            text_label = "%.2E \n$\sigma=%.2E\,$\n%.0f$\,$counts \n(%.0f%%)" % (pos, width, auc, auc/self.n_binding*100)
                    else:
                        if short_labels:
                            text_label = "%.0f kDa" % pos
                        else:
                            text_label = "%.0f kDa\n$\sigma=%.0f\,$kDa\n%.0f$\,$counts \n(%.0f%%)" % (pos, width, auc, auc/self.n_binding*100)
                    ax.text(pos, height+0.05*np.max(counts), text_label , ha='center', va='bottom')
            if plot_weights:
                ax.plot(hist_centers, weights * np.max(counts), color='k')
        fig.tight_layout()
        # Get axis dimension
        x_borders = ax.get_xlim()
        y_borders = ax.get_ylim()
        # Print number of counts
        if show_counts:
            if counts_pos == 'left':
                ax.text(x_borders[0] + 0.01*(x_borders[1] - x_borders[0]), y_borders[1]*.99, "Total counts: %i\nBinding: %.0f%%\nUnbinding: %.0f%%" % (self.n_counts, self.n_binding/self.n_counts*100, self.n_unbinding/self.n_counts*100), va='top', ha='left')
            elif counts_pos == 'right':
                ax.text(x_borders[1]*.99, y_borders[1]*.99, "Total counts: %i\nBinding: %.0f%%\nUnbinding: %.0f%%" % (self.n_counts, self.n_binding/self.n_counts*100, self.n_unbinding/self.n_counts*100), va='top', ha='right')
        # If movie file is specified, create inlet with frame picture
        # Calculate coordinates
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        x_space = xlims[1] - xlims[0]
        y_space = ylims[1] - ylims[0]
        # If counts are shown use larger space
        # 50% of xspace and 50% of yspace
        if show_counts:
            # Use 40% of space
            x0 = xlims[0] + .6*x_space + self.image_xoffset
            x_size = .9*.4*x_space * self.image_scale
            y0 = ylims[0] + 0.5*.6*y_space + self.image_yoffset
            y_size = .9*.4*y_space * self.image_scale
        # Otherwise use 60% of y-space
        else:
            x0 = xlims[0] + .4*x_space + self.image_xoffset
            x_size = .97*.6*x_space * self.image_scale
            y0 = ylims[0] + .4*y_space + self.image_yoffset
            y_size = .9*.6*y_space * self.image_scale
        # # Draw image
        axin = ax.inset_axes([x0, y0, x_size, y_size],transform=ax.transData, alpha=.5)    # create new inset axes in data coordinates
        axin.imshow(self.dra)
        axin.axis('off')
        # Create circles
        for event in self.events.iterrows():
            event = event[1]
            circ = Circle((int(event['x_coords']), int(event['y_coords'])), 5, fc='None', ec='red', lw=2)
            axin.add_patch(circ)
            axin.text(int(event['x_coords']), int(event['y_coords'])+5, int(event['kDa']), ha='center', va='top', fontsize=6)
        return fig, ax
    
    def fit_histo(self, xlim=[], guess_pos=[], tol=None, max_width=None, weighted=False, weighted_width=None, contrasts=False, cutoff=0, fit_points=1000):
                  #tol=100, tol_contrasts = 0.05, max_width=100, max_width_contrasts=0.005, weighted=False, weighted_width=200, weighted_width_contrasts=0.005, contrasts=False, cutoff=0, fit_points=1000):
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
        fit_points: How many points are used for plotted fits (the more the finer), default is 1000
        '''
        # Depending on contrasts select the right quantity
        if contrasts:
            centers = self.hist_centers_contrasts
            counts = self.hist_counts_contrasts
            # Set default values if not specified
            if tol==None:
                tol = 0.005
            if max_width==None:
                max_width = 0.005
            if weighted_width==None:
                weighted_width = 0.005
            ## Use tolerance for contrasts
            #tol = tol_contrasts
            #max_width = max_width_contrasts
            #weighted_width = weighted_width_contrasts
        # Otherwise if masses are fitted
        else:
            # Load data for masses/kDa
            centers = self.hist_centers
            counts = self.hist_counts
            # Set default values if not specified
            if tol==None:
                tol = 100
            if max_width==None:
                max_width = 100
            if weighted_width==None:
                weighted_width = 200
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
        print("Number of points to be fitted %i" % len(centers))
        # If no guess are taken, only fit one gaussian and use maximum in histogram as guess
        if len(guess_pos) == 0:
            print("No guess positions given (guess_pos)! Will try to find maxima.")
            peaks, info = ssi.find_peaks(counts, prominence=0.1*np.max(counts))
            guess_pos = centers[peaks]
            guess_amp = counts[peaks]
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
        x = np.linspace(np.min(centers), np.max(centers), fit_points)
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

def MultipleHisto(MP_list=[], markers=[], param_dict = {'xlim': [0, 2000], 'show_counts': False, 'short_labels': True}):
    '''
    Function to plot multiple histograms
    MP_list: List MP_data handles
    markers: dashed lines to mark positions
    markers_text: 
    param_dict: Dictionary with parameters for plot_histo
    '''
    # Check that all members of MP_list are actually MP_Data
    for mp_data in MP_list:
        if type(mp_data) != MP_data:
            print("%s is not MP_data! Exiting" % mp_data)
            print(type(mp_data))
            return None
    # Create grid for subplots and figure
    gs = gridspec.GridSpec(len(MP_list), 1,
                           wspace=0.0, hspace=0.0, top=0.95, bottom=0.15, left=0.17, right=0.845) 
    #fig, axs = plt.subplots(len(MP_list))
    fig = plt.figure()
    # Cycle through list and plot
    axs = []
    for i in range(len(MP_list)):
        ax = plt.subplot(gs[i, 0])
        axs.append(ax)
        MP_list[i].plot_histo(ax=ax, **param_dict)
    # Remove all but the last xtick labels
    for ax in axs[:-1]:
        ax.set_xticklabels([])
    return fig, axs
        
