class OpusData():
    '''
    Class to load and process Opus IR data
    fn: Opus filename  (extension normally .0, .1...)
    spec_lim: Extract data in this region; give empty list if all data is to be extracted
    peak_lim: Region with peaks, this is excluded when baseline is determined
    interpol_space: Spacing for interpolation
    sg_window: Window for Savitzky-Golay (SG) smoothing
    sg_pol: Polynomial order for SG smoothing
    bl_pol: Polynomial order for baseline determination
    '''
    
    def __init__(self, fn='', spec_lim=[2100,2200], peak_lim=[2145,2175], interpol_space=1, sg_window=13, sg_poly=2, bl_pol=6):
        self.folder = '/'.join(fn.split('/')[:-1])
        self.fn = fn.split('/')[-1]
        self.spec_lim = np.sort(spec_lim)
        self.peak_lim = np.sort(peak_lim)
        self.interpol_space = interpol_space
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.bl_pol = bl_pol
        return None
    
    def load(self, process=True):
        '''
        Load data with initialized filename
        '''
        full_fn = self.folder + '/' + self.fn
        # Load data
        dbs = ofc.listContents(full_fn)
        data = {}
        for db in dbs:
            data[str(db[0])] = ofc.getOpusData(full_fn, db)
        #print(data.keys())

        # Chose AB as spectrum
        spec_full = np.vstack((data['AB'].x, data['AB'].y)).T
        # Convert to mOD
        spec_full[:,1] = spec_full[:,1]*1000

        # If x values are decreasing, flip matrix
        if spec_full[0,0] > spec_full[-1,0]:
            spec_full = np.flipud(spec_full)
        # Backup original data
        self.spec_orig = spec_full

        # Process if desired
        if process:
            self.extract()
            self.smooth()
            self.bl_correction()
        return None

    def extract(self):
        '''
        Extract spectrum and interpolate
        Limits are used from self.spec_lim
        '''

        # Check if lower limit is valid
        if np.min(self.spec_orig[:,0]) > self.spec_lim[0]:
            print("Lower spec_lim is lower than smallest spectral value!\n Will use the latter as limit.")
            self.spec_lim = np.ceil(self.spec_orig[:,0])
        # Check if upper limit is valid
        if np.max(self.spec_orig[:,0]) < self.spec_lim[1]:
            print("Higher spec_lim is higher than largest spectral value!\n Will use the latter as limit.")
            self.spec_lim = np.floor(self.spec_orig[:,0])

        # x values for interpolation (need to increase higher by one so that it's included)
        self.x = np.arange(*(self.spec_lim + [0,1]), self.interpol_space)
        # Interpolate
        self.raw = np.interp(self.x, self.spec_orig[:,0], self.spec_orig[:,1])
        return None
        
    def smooth(self):
        '''
        Use Savitzky-Golay filter to smooth
        Use self.sg_window and self.sg_poly as parameters
        Writes output to self.smoothed and self.deriv1/2
        '''
        # Smooth it
        self.smoothed = np.array(ssi.savgol_filter(self.raw, self.sg_window, self.sg_poly)).transpose()
        # Get second derivative
        self.deriv1 = np.array(ssi.savgol_filter(self.raw, self.sg_window, self.sg_poly, deriv=1)).transpose()
        self.deriv2 = np.array(ssi.savgol_filter(self.raw, self.sg_window, self.sg_poly, deriv=2)).transpose()
        print("Smoothed spectrum and derived 1st derivative")
        # Get zero crossings of second derivative
        self.pos_zero1 = np.where(np.diff(np.sign(self.deriv1)))[0]
        self.x_zero1 = self.x[self.pos_zero1]
        # Get zero crossings of second derivative
        self.pos_zero2 = np.where(np.diff(np.sign(self.deriv2)))[0]
        self.x_zero2 = self.x[self.pos_zero2]
        return None
        
    def bl_correction(self):
        '''
        Do polynomial baseline correction
        Use self.bl_pol as polynomial order
        Ignores region defined by self.peak_lim 
        Writes baseline to self.bl and bl corrected spectrum to self.bl_corr
        '''

        # Check if smoothed spectrum is there, otherwise use raw spectrum
        if hasattr(self, 'smooth'):
            print("Found smoothed spectrum. Will use this")
            spec = self.smoothed
        else:
            print("Did not find smoothed spectrum. Will use raw spectrum")
            spec = self.raw
        
        # Calculate weighting matrix in range
        peak_pos = [np.argmin(np.abs(self.x - self.peak_lim[0])), np.argmin(np.abs(self.x - self.peak_lim[1]))]
        peak_pos = np.sort(peak_pos)
        w_vector = np.ones(len(spec))
        w_vector[peak_pos[0]: peak_pos[1]] = 0
        self.w_vector = w_vector

        # Polynomial fit
        p = np.polyfit(self.x, spec, self.bl_pol, w = w_vector)
        self.bl = np.polyval(p, self.x).transpose()
        self.bl_corr = np.array(spec - self.bl).transpose()
        print("Did baseline correction")
        #if norm==True:
        #    bl_corrected = bl_corrected / np.max(bl_corrected)
        #    # Determine minimum of the difference in peak_limits
        #    yshift = np.min(bl_corrected)
        # Shift the processed spectrum by yshift
        #bl_corrected = bl_corrected# - yshift

        return None

    def deconvolute(self, fit_type='gauss', fit_number=1):
        '''
        Fit either sum of gaussian or lorentzian functions to spectrum
        
        '''
        # Set fit function
        if fit_type=='gauss':
            fit_func = multi_gauss
        elif fit_type=='lorentz':
            fit_func = multi_lorentz
        else:
            print("Unknown fit_type! Please choose either 'gauss' or 'lorentz'")
            return None
        self.fit_number = fit_number
        self.fit_type = fit_type
        # Check that bl_corr spectrum is there
        if not hasattr(self, 'bl_corr'):
            print("Baseline corrected spectrum not found! Please run bl_correction first.")
            return None
        else:
            spec = self.bl_corr
        # Get maximum and intensity
        max_x = self.x[np.argmax(spec)]
        max_y = np.max(spec)
        # Set initial values for fit
        if fit_number==1:
            p0 = [max_x, max_y, 10]
            lbounds = [np.min(self.peak_lim), 0, 0 ]
            ubounds = [np.max(self.peak_lim), max_y, np.inf]
        elif fit_number==2:
            p0 = [max_x-5, .5*max_y, 10, max_x+5, .5*max_y, 10]
            lbounds = [np.min(self.peak_lim), 0, 0, np.min(self.peak_lim), 0, 0 ]
            ubounds = [np.max(self.peak_lim), max_y, np.inf, np.max(self.peak_lim), max_y, np.inf]
        else:
            print("Fit currently only implemented for 1 and 2 components (fit_number)!")
            return None
        # Do the fit
        popt, pcov = curve_fit(fit_func, self.x, spec, p0=p0, maxfev=1000000, bounds=(lbounds, ubounds))
        self.popt, self.pcov = popt, pcov
        self.fit_sum = fit_func(self.x, *popt)
        # Calculate R2
        self.r2 = r_sq(spec, self.fit_sum)
        # If 2 components were chosen, save them individually
        if fit_number == 2:
            self.fit_1 = fit_func(self.x, *popt[:3])
            self.fit_2 = fit_func(self.x, *popt[3:])
        else:
            self.fit_1 = self.fit_sum
        return None

    def plot(self, plot_raw=True, plot_deriv=True, plot_fit=True):
        '''
        Creates plot with spectra and fits
        '''
        # Determine number of subplots
        no_subplots = plot_raw + plot_deriv + plot_fit
        # Initialize plot
        fig, axs = plt.subplots(no_subplots, sharex=True)
        if no_subplots == 1:
            axs = [axs]
        for i in range(no_subplots):
            ax = axs[i]
            if plot_raw:
                spec = self.raw
                ax.plot(self.x, spec, label='Raw spectrum')
                ax.plot(self.x, self.bl, '--', color='grey', label='Baseline')
                ax.legend()
                ax.set_ylabel('Absorbance / mOD')
                plot_raw = False
                continue
            if plot_deriv:
                ax.plot(self.x, self.deriv2, label='1st derivative')
                # ax.legend()
                ax.set_ylabel('1st derivative')
                ax.set_ylabel('2nd derivative')
                ax.axhline(0, ls='--', color='grey')
                # Plot zero crossings
                for i, cross in enumerate(self.x_zero2):
                    ax.axvline(cross, ls=':', color='gray')
                    # Label zero crossings
                    if i % 2 == 0:
                        ha = 'right'
                    else:
                        ha = 'left'
                    ax.text(cross, 1.1* ax.get_ylim()[1], "%.0f" % cross, ha=ha)
                plot_deriv = False
                continue
            if plot_fit:
                # Check which fit was used for label
                if self.fit_type=='gauss':
                    label = 'Gaussian fit (R$^2$=%.4f)' % self.r2 
                elif self.fit_type=='lorentz':
                    label = 'Lorentzian fit (R$^2$=%.4f)' % self.r2 
                # Plot data
                ax.plot(self.x, self.bl_corr, label='Exp. spectrum')
                hp, = ax.plot(self.x, self.fit_sum, '--', label=label)
                # Plot individual components if more than 1
                ax.plot(self.x, self.fit_1, ':', color=hp.get_color())
                ax.axvline(self.popt[0], ls=':', color=hp.get_color())
                if len(self.popt) > 3:
                    ha = 'right'
                else:
                    ha = 'center'
                ax.text(self.popt[0], 1.1* ax.get_ylim()[1], "%.0f(%.0f)" % (self.popt[0], self.popt[2]), ha=ha)
                if len(self.popt) > 3:
                    ax.plot(self.x, self.fit_2, ':', color=hp.get_color())
                    ax.axvline(self.popt[3], ls=':', color=hp.get_color())
                    ax.text(self.popt[0], 1.1* ax.get_ylim()[1], "%.0f(%.0f)" % (self.popt[3], self.popt[5]), ha='left')
                ax.legend()
                ax.set_ylabel('Absorbance / mOD')
                plot_fit = False
            ax.set_xlabel('Wavenumber / cm$^{-1}$')
            ax.set_xlim([self.x[0], self.x[-1]])
            fig.tight_layout()
            # Save plot
            save_name = self.folder + self.fn[:-2] + '_%i%s' % (self.fit_number, self.fit_type)
            fig.savefig( save_name + '.pdf')
            fig.savefig(save_name + '.png')
            print("Plot saved as %s" % (save_name + '.png'))
        return None
