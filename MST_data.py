'''
MST data
'''

from .helpers import *
import pandas as pd
import numpy as np

class MST_data():
    '''
    This is a class to read in MST data that 
    has been exported as xlsx
    '''
    def __init__(self, fn=''):
        '''
        Initialize ms data
        fn: File name
        '''
        self.fn = fn
        #self.path = path
        # Load data
        xls = pd.ExcelFile(fn)
        #print(xls.sheet_names)
        dat = pd.read_excel(xls, 'RawData', index_col=None, header=None)
        
        # Find out where data starts
        dat_pos = np.argwhere(np.array(dat.iloc[:,0]) == 'Time [s]')[0,0] +1
        # Get concentrations and locations
        lig_pos_ver = np.argwhere(np.array(dat.iloc[:,0]) == 'Ligand Concentration:')[0,0] 
        lig_pos_hor = np.argwhere(np.array(dat.iloc[lig_pos_ver,:]) == 'Ligand Concentration:').squeeze() +1
        self.concs = np.array(dat.iloc[lig_pos_ver, lig_pos_hor]).astype(np.float32) * 1E-6
        # Get ligand name
        lig_pos_ver = np.argwhere(np.array(dat.iloc[:,0]) == 'Ligand:')[0,0] 
        lig_pos_hor = np.argwhere(np.array(dat.iloc[lig_pos_ver,:]) == 'Ligand:').squeeze() +1
        self.lig_names = np.array(dat.iloc[lig_pos_ver, lig_pos_hor]).astype(str)       
        # Get times
        self.times = np.array(dat.iloc[dat_pos:,0]).astype('float32')
        # Get decays
        self.decays = np.array(dat.iloc[dat_pos:,lig_pos_hor]).astype('float32')
        # Outliers
        self.outliers = []
        # Remove nans in time
        non_nans = ~np.isnan(self.times)
        self.times = self.times[non_nans]
        self.decays = self.decays[non_nans, :]
        # Sort
        self.sort()
        return None

    def sort(self):
        '''
        Sort ligand concentrations and fluorescence
        with ascending ligand concentration
        '''
        sort_ind = np.argsort(self.concs)
        self.concs = self.concs[sort_ind]
        self.decays = self.decays[:, sort_ind]
        self.lig_names = self.lig_names[sort_ind]
        return None

    def subset(self, inds=[], ligname=''):
        '''
        Select subset and delete other entries
        Either select by indices or ligand name
        '''
        if len(ligname) > 0:
            print('Selection based on ligand name')
            inds = np.argwhere(self.lig_names == ligname).squeeze()
            print('Found %i entries with name: %s' % (len(inds), ligname))
        if len(inds) == 0:
            print('Ligand name does not match or empty indices list! Aborting')
            return None
        self.concs = self.concs[inds]
        self.decays = self.decays[:, inds]
        self.lig_names = self.lig_names[inds]
        # Sort afterwards
        self.sort()
        return None
        
    def normalize(self):
        '''
        Normalize with values at t <= 0
        '''
        ind_neg = self.times <=0
        # This uses the maximum for negative times
        #self.decays /= np.nanmax(self.decays[ind_neg,:], axis=0)
        # This uses the mean for negative times
        self.decays /= np.mean(self.decays[ind_neg,:], axis=0)
        return None

    def calc_fnorm(self, hot=20, cold=0, no_cold=False):
        '''
        This calculates fnorm
        '''
        ind_hot = (self.times >= hot-1) * (self.times <= hot)
        ind_cold = (self.times >= cold-1) * (self.times <= cold)
        F_cold = np.mean(self.decays[ind_cold,:], axis=0)
        F_hot = np.mean(self.decays[ind_hot,:], axis=0)
        if no_cold:
            # This is if only hot region is used (e.g. because cold is 0)
            fnorm = F_hot
        else:
            fnorm = F_hot/F_cold
        self.fnorm = fnorm
        self.hot = hot
        self.cold = cold
        self.F_cold = F_cold
        self.F_hot = F_hot
        return fnorm

    def plot_init_fluo(self, fix_pconc=False, hot=20, cold=0, bleach_correct=False):
        '''
        This extracts the initial fluorescence and plots it vs conc
        '''
        # Obtain indices for times < 0
        inds = self.times <= 0
        # Do linear regression
        p_params = np.polyfit(self.times[inds], self.decays[inds], 1)
        # Initial values
        f_init = [np.polyval(p_params[:,i], self.times[0]) for i in range(p_params.shape[1])]
        f_bleach = np.array([np.polyval(p_params[:,i], self.times[inds]) for i in range(p_params.shape[1])]).T
        f_bl = np.array([np.polyval(p_params[:,i], self.times) for i in range(p_params.shape[1])]).T
        fnorm = self.calc_fnorm(hot=hot, cold=cold)
        # Calculate new fnorm with bleach corrected data
        # No cold region necessary nor possible (since it's zero)
        decays_init = self.decays
        self.decays = self.decays - f_bl
        fnorm_bl = self.calc_fnorm(hot=hot, cold=cold, no_cold=True)
        # Move back
        self.decays = decays_init
        # Fit f_init
        try:
            fit_f_init, fit_f_init_opt, fit_f_init_err = self.fit_kd(self.concs, f_init, fix_pconc=fix_pconc)
        except:
            print("Could not fit initial fluorescence data")
            fit_f_init = []
        try:
            fit_f_bleach, fit_f_bleach_opt, fit_f_bleach_err = self.fit_kd(self.concs, -p_params[0], fix_pconc=fix_pconc)
        except:
            print("Could not fit bleach")
            fit_f_bleach = [] 
        try:
            fit_fnorm, fit_fnorm_opt, fit_fnorm_err = self.fit_kd(self.concs, fnorm , fix_pconc=fix_pconc)
        except:
            print("Could not fit fnorm")
            fit_fnorm = []
        try:
            fit_fnorm_bl, fit_fnorm_bl_opt, fit_fnorm_bl_err = self.fit_kd(self.concs, fnorm_bl , fix_pconc=fix_pconc)
        except:
            print("Could not fit fnorm_bl")
            fit_fnorm_bl = []
        
        # Do fit for f_bleach
        # Create plot
        #fig, axs = plt.subplots(2,2, figsize=(10,7.5))
        fig = plt.figure(figsize=(10,7.5)) # constrained_layout=True
        axs = []
        gs = fig.add_gridspec(2,2)
        axs.append(fig.add_subplot(gs[0,:]))
        axs.append(fig.add_subplot(gs[1,0]))
        axs.append(fig.add_subplot(gs[1,1]))
        # Plot initial fluorescence
        ax = axs[0]
        ax.set_title('Non-normalized MST signal')
        self.plot_colored(ax, self.times, self.decays, self.concs)
        self.add_colorbar(ax, self.concs)
        #ax.plot(self.times, self.decays)
        if bleach_correct:
            self.plot_colored(ax, self.times, f_bl, self.concs, linestyle='--', lw=.5)
        ax.set_ylabel('Fluorescence / Counts')
        ax.set_xlabel('Time / s')
        # ax = axs[1,0]
        # ax.set_title('Bleach-corrected signal')
        # self.plot_colored(ax, self.times, self.decays - f_bl, self.concs)
        # self.add_colorbar(ax, self.concs)
        # ax.set_ylabel('Fluorescence / Counts')
        # ax.set_xlabel('Time / s')
        ax = axs[1]
        ax.set_xlabel('Ligand concentration / M')
        if bleach_correct:
            ax.set_title('Bleach-corrected MST analysis')
            self.plot_colored(ax, self.concs, fnorm_bl, self.concs)
            kd_label = kd_unit(fit_fnorm_bl_opt[0])
            label = "$K_d=$%s$\pm%.0f$" % (kd_label, fit_fnorm_bl_err[0] / fit_fnorm_bl_opt[0]  * 100) +'%'
            # Finer concs for plotting fit
            concs_fine = np.linspace(np.min(self.concs), np.max(self.concs), 300)
            if fix_pconc:
                fit_fine = single_site_kd(self.pconc)(concs_fine, *fit_fnorm_bl_opt)
            else:
                print("Fitted pconc.: %.1E" % fit_fnorm_bl_opt[1])
                print(fit_fnorm_bl_opt)
                fit_fine = single_site(concs_fine, *fit_fnorm_bl_opt)
            #ax.semilogx(self.concs, fit_fnorm_bl, '-', zorder=-20, label=label)
            ax.semilogx(concs_fine, fit_fine, '-', zorder=-20, label=label)
            ax.legend()
            ax.set_ylabel('F$_\mathrm{norm,bl}$ / Counts' )
        else:
            ax.set_title('MST analysis (after normalisation)')
            self.plot_colored(ax, self.concs, fnorm, self.concs)
            if (len(fit_fnorm) > 0 ):
                if fit_fnorm_err[0] / fit_fnorm_opt[0] < 1:
                    print("Fit error for fnorm larger than 100%. Will not plot it")
                    label = "$K_D$=%.0EM$\pm%.0f$" % (fit_fnorm_opt[0], fit_fnorm_err[0] / fit_fnorm_opt[0]  * 100) +'%' 
                    ax.semilogx(self.concs, fit_fnorm, '-', zorder=-20, label=label)
                    ax.legend()
                ax.set_ylabel('F$_\mathrm{norm}$ / ' + u'\u2030')
        ax = axs[2]
        ax.set_title('Initial fluorescence vs. ligand conc.')
        self.plot_colored(ax, self.concs, f_init, self.concs)
        print(fit_f_init)
        if (len(fit_f_init) > 0 ):
            if fit_f_init_err[0] / fit_f_init_opt[0] < 1:
                print("Fit error for f_init larger than 100%. Will not plot it")  
                label = "$K_D=$%.0E$\mathrm{M}\pm$%.0f" % (fit_f_init_opt[0], fit_f_init_err[0] / fit_f_init_opt[0]  * 100) +'%' 
                ax.semilogx(self.concs, fit_f_init, '-', zorder=-20, label=label)
                ax.legend()
        ax.set_xlabel('Ligand concentration / M')
        ax.set_ylabel('F$_\mathrm{init}$ / Counts')
        # Plot tolerance area +- 20%
        F_mean = np.mean(f_init)
        ax.axhline(F_mean, linestyle='--', color='grey', zorder=-10)
        ax.axhline(.8*F_mean, linestyle='--', color='grey', zorder=-10)
        ax.axhline(1.2*F_mean, linestyle='--', color='grey', zorder=-10)
        ax.axhspan(.8*F_mean, 1.2*F_mean, facecolor='grey', alpha=.3, zorder=-20)
        # # Plot bleach rate
        # ax = axs[1,1]
        # ax.set_title('Bleaching rate vs. ligand conc.')
        # self.plot_colored(ax, self.concs, -p_params[0], self.concs)
        # label = "$K_D$=%.0E$\pm%.0f$" % (fit_f_bleach_opt[0], fit_f_bleach_err[0] / fit_f_bleach_opt[0]  * 100) +'%' 
        # ax.semilogx(self.concs, fit_f_bleach, '--', zorder=-20, label=label)
        # ax.legend()
        # #ax.axvline(fit_f_bleach_opt[0], linestyle='--')
        # ax.set_xlabel('Ligand concentration / M')
        # ax.set_ylabel('Bleach rate / Counts/s')
        # Plot regions
        ax = axs[0]
        ax.axvspan(self.hot-1, self.hot, facecolor='red', alpha=.5)
        ax.set_xlim([np.nanmin(self.times), np.nanmax(self.times)])
        ax.axvspan(self.cold-1, self.cold, facecolor='blue', alpha=.5)
        # Change xlim
        for ax in axs[1:]:
            ax.set_xlim([np.floor(np.log10(self.concs[0])), np.ceil(np.log10(self.concs[-1]))])
            ax.set_xlim([np.floor(np.log10(self.concs[0])), np.ceil(np.log10(self.concs[-1]))])
        fig.tight_layout()
        fig.show()
        # Write results to dictionary
        self.analysis = {'f_init': f_init}
        # Save plot
        fig.savefig(self.fn.replace('.xlsx', '_mst_and_init_F.pdf'))
        print("Figure saves as %s" % self.fn.replace('.xlsx', '_mst_and_init_F.pdf'))
        fig.savefig(self.fn.replace('.xlsx', '_mst_and_init_F.png'))
        print("Figure saves as %s" % self.fn.replace('.xlsx', '_mst_and_init_F.png'))
        return fig, ax

    def fit_kd(self,concs, y, fix_pconc=False):
       # Chose fitting function
        if fix_pconc:
            func = single_site_kd(self.pconc)
            print("Will fit with fixed protein concentration of %.1E M." % self.pconc)
        else:
            func = single_site
            print("Will fit with variable protein concentration.")
        # Get starting values
        nonbound0 = y[0]
        bound0 = y[-1]
        half_bound = np.mean((nonbound0, bound0))
        kd0 = concs[np.argmin(np.abs(y - half_bound))]
        pconc0 = self.pconc
        if fix_pconc:
            bounds = ((0, -np.inf, -np.inf), (np.inf, np.inf, np.inf))
            p0 = (kd0, nonbound0, bound0)
        else:
            bounds = ((0, 0, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf))
            p0 = (kd0, pconc0, nonbound0, bound0)
        opt, cov = curve_fit(func, concs, y, p0=p0, bounds=bounds) #, p0=(1E-6, np.min(self.fnorm), np.max(self.fnorm)))
        if fix_pconc:
            print("Fitted protein conc: %.1EM" % (self.pconc)) #, opt[1]))
            #self.pconc = opt[1]
        # Print results
        err = np.sqrt(np.diag(cov))
        fit = func(concs, *opt)
        return fit, opt, err

    def get_kd(self, fix_pconc=True, use_fluo=False):
        '''
        Get Kd from fnorm
        fix_pconc: Fix protein concentration, otherwise it will be fit
        use_fluo: Use initial fluorescence instead of Fnorm
        '''
        if not hasattr(self, 'fnorm'):
            print("Fnorm has not been calculated yet!\n Will do that now")
            self.calc_fnorm()
        if not hasattr(self, 'pconc'):
            print("Protein concentration not specified yet!")
            print("This needs to be done before by setting conc_prot!")
            print("Exiting function")
            return None
        # Set fix_pconc
        self.fix_pconc = fix_pconc
        # Remove outliers
        concs_in, fnorm_in = [], []
        concs_out, fnorm_out = [], []
        for i in range(len(self.concs)):
            if i in self.outliers:
                concs_out.append(self.concs[i])
                fnorm_out.append(self.fnorm[i])
            else:
                concs_in.append(self.concs[i])
                fnorm_in.append(self.fnorm[i])                
        # Chose fitting function
        if fix_pconc:
            func = single_site_kd(self.pconc)
            print("Will fit with fixed protein concentration of %.1e." % self.pconc)
        else:
            func = single_site
            print("Will fit with variable protein concentration.")
        # Get starting values
        nonbound0 = fnorm_in[0]
        bound0 = fnorm_in[-1]
        half_bound = np.mean((nonbound0, bound0))
        kd0 = concs_in[np.argmin(np.abs(fnorm_in - half_bound))]
        pconc0 = self.pconc
        if fix_pconc:
            bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))
            p0 = (kd0, nonbound0, bound0)
        else:
            bounds = ((0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf))
            p0 = (kd0, pconc0, nonbound0, bound0)
        opt, cov = curve_fit(func, concs_in, fnorm_in, p0=p0, bounds=bounds) #, p0=(1E-6, np.min(self.fnorm), np.max(self.fnorm)))
        # Print results
        err = np.sqrt(np.diag(cov))
        print("Error for Kd: %.2e+-%.2e" % (opt[0], err[0]))
        if fix_pconc:
            print("Error for nonbound: %.2f+-%.2f%%" % (opt[1], 100*err[1]/opt[1]))
            print("Error for bound: %.2f+-%.2f%%" % (opt[2], 100*err[2]/opt[2]))
        else:
            print("Error for pconc: %.1e+-%.1f%%" % (opt[1], 100*err[1]/opt[1]))
            print("Error for nonbound: %.2f+-%.2f%%" % (opt[2], 100*err[2]/opt[2]))
            print("Error for bound: %.2f+-%.2f%%" % (opt[3], 100*err[3]/opt[3]))
            print("Updated concentration from %.1e to %.1e" % (self.pconc, opt[1])) 
            self.pconc = opt[1]
        # Calculate dense curves for plot
        concs_dense = np.exp(np.linspace(np.log(self.concs[0]), np.log(self.concs[-1]), 100))
        kd_err = np.sqrt(cov[0,0])
        if self.fix_pconc:
            func = single_site_kd(self.pconc)
            fit_upper = func(concs_dense, *(opt - np.array([kd_err, 0, 0])))
            fit_lower = func(concs_dense, *(opt + np.array([kd_err, 0, 0])))
        else:
            func = single_site
            fit_upper = func(concs_dense, *(opt - np.array([kd_err, 0, 0, 0])))
            fit_lower = func(concs_dense, *(opt + np.array([kd_err, 0, 0, 0])))
        fit = func(concs_dense, *opt)
        # Write results to instance
        self.fit_opt = opt
        self.fit_cov = cov
        self.fit_err = np.sqrt(np.diag(cov))
        self.concs_in = concs_in
        self.fnorm_in = fnorm_in
        self.func = func
        self.concs_dense = concs_dense
        self.fit = fit
        self.fit_upper = fit_upper
        self.fit_lower = fit_lower
        #self.plot()
        return opt, cov

    def plot_colored(self, ax, xs, ys, concs, outliers=[], lw=1, alpha=1, alpha_out=.2, linestyle='-'):
        '''
        Helper script to plot color coded (vs. conc) curves
        Make sure that concs are sorted in ascending order!
        '''
        # Create color map
        cmap = plt.cm.jet(np.linspace(0, 1, len(concs)))
        # This is to use it both for decays (multi-rows) and fnorms (one value)
        ys = np.array(ys).T
        # Plot curves or points
        for i in range(len(concs)):
            if i in self.outliers:
                # Check if is number or array
                if isinstance(ys[i], np.ndarray):
                    ax.plot(xs, ys[i], alpha=alpha_out, lw=lw, color=cmap[i], linestyle=linestyle)
                else:
                    ax.semilogx(xs[i], ys[i], 'o', alpha=alpha_out, lw=lw, color=cmap[i], markeredgecolor='k')
                    # if hasattr(self, 'fnorm'):
                #     axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha_out, lw=lw, color=cmap[i])
            else:
                # Check if is number or array
                if isinstance(ys[i], np.ndarray):
                    ax.plot(xs, ys[i],  alpha=alpha, lw=lw, color=cmap[i], linestyle=linestyle)
                else:
                    ax.semilogx(xs[i], ys[i], 'o', alpha=alpha, lw=lw, color=cmap[i], markeredgecolor='k')

                # if hasattr(self, 'fnorm'):
                #     axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha, lw=lw, color=cmap[i])
        return None


    def add_colorbar(self, ax, concs):
        '''
        Helper function to add colorbar
        Use in combination with plot_colored
        Make sure that concs are sorted in ascending order
        '''
        # setup the colorbar
        normalize = mcolors.LogNorm(vmin=np.min(concs), vmax=np.max(concs)) # Or Normalize 
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=plt.cm.jet) 
        scalarmappaple.set_array(concs)
        cbar = plt.colorbar(scalarmappaple, ax=ax)
        cbar.set_label('Lig. conc. / M', rotation=270) 
        cbar.ax.get_yaxis().labelpad = 15
        return None
    
    def plot(self, smooth=False, smooth_window=51, show_error=True):
        if hasattr(self, 'fnorm'):
            fig, axs = plt.subplots(1,2, figsize=(10,4))
            ## Plot outliers in gray
            #for out in self.outliers:
            #    ax.semilogx(self.concs[out], self.fnorm[out], 'o', color='gray')
            ax = axs[1]
            if hasattr(self, 'fit_opt'):
                # Upper and lower limits for model (based on KD error)
                kd_label = kd_unit(self.fit_opt[0])
                if show_error:
                    kd_label = '$K_d=$%s$\pm$%.0f%%' % (kd_label, self.fit_err[0]/self.fit_opt[0]*100)
                else:
                    kd_label = '$K_d=$%s' % kd_label
                hp_fit, = ax.semilogx(self.concs_dense, self.fit, label=kd_label)
                if show_error:
                    ax.fill_between(self.concs_dense, self.fit_upper, self.fit_lower, facecolor=hp_fit.get_color(), alpha=.5, zorder=-20)
                ax.legend()
            ax.set_xlabel('Ligand concentration / M')
            ax.set_ylabel('F$_\mathrm{norm}$ / ' + u'\u2030')
            ax = axs[0]
        else:
            fig, axs = plt.subplots(1)
            ax = axs
        # Plot
        lh = []   # Line handles
        alpha=1
        alpha_out = .2
        lw =1

        # Set up color map
        # uconcs = np.unique(self.concs)
        #cmap = plt.set_cmap(plt.jet())
        # Get smallest conc that is not zero (zero cannot be shown in log scale)
        minconc = np.min(self.concs[self.concs>0])
        # Define color map
        # cmap = iter(plt.cm.jet(np.linspace(0,1, len(np.unique(self.concs))+0)))
        cmap = plt.cm.jet(np.linspace(0,1, len(self.concs)))
        # setup the colorbar
        normalize = mcolors.LogNorm(vmin=np.min(self.concs), vmax=np.max(self.concs)) # Or Normalize 
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=plt.cm.jet) 
        scalarmappaple.set_array(self.concs)
        cbar = plt.colorbar(scalarmappaple, ax=ax)
        cbar.set_label('Lig. conc. / M', rotation=270) 
        cbar.ax.get_yaxis().labelpad = 15

        # Smooth data if chosen
        if smooth:
            data_temp = ssi.savgol_filter(self.decays, smooth_window, 2, axis=0)
        else:
            data_temp = self.decays
        
        # Full plot
        # Make sure that each conc. only has one color
        prev_conc = -1 
        for i in range(len(self.concs)):
            # if i in self.outliers:
            #     print(i)
            #     ax.plot(self.times, self.decays[:, out], alpha=1, lw=lw, color='gray')
            #     cmap.__next__()
            #     continue
            # Exception for 0 concentration (not defined in log scale colormap)
            if self.concs[i]==0:
                if i in self.outliers:
                    ax.plot(self.times, data_temp[:, i], label="%.1f uM" % (self.concs[i]*1E6), alpha=alpha_out, lw=lw, color='k')
                    if hasattr(self, 'fnorm'):
                        axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha_out, lw=lw, color='k')
                else:
                    ax.plot(self.times, data_temp[:, i], label="%.1f uM" % (self.concs[i]*1E6), alpha=alpha, lw=lw, color='k')
                    if hasattr(self, 'fnorm'):
                        axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha, lw=lw, color='k')
                continue
            if i in self.outliers:
                temp, = ax.plot(self.times, data_temp[:, i], label="%.1f uM" % (self.concs[i]*1E6), alpha=alpha_out, lw=lw, color=cmap[i])
                if hasattr(self, 'fnorm'):
                    axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha_out, lw=lw, color=cmap[i], markeredgecolor='k')
            else:
                ax.plot(self.times, data_temp[:, i], label="%.1f uM" % (self.concs[i]*1E6), alpha=alpha, lw=lw, color=cmap[i])
                if hasattr(self, 'fnorm'):
                    axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha, lw=lw, color=cmap[i], markeredgecolor='k')
                         
            # if prev_conc != self.concs[i]:
            #     if i in self.outliers:
            #         temp, = ax.plot(self.times, data_temp[:, i], label="%.1f uM" % (self.concs[i]*1E6), alpha=alpha_out, lw=lw, color=cmap.__next__())
            #         if hasattr(self, 'fnorm'):
            #             axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha_out, lw=lw, color=temp.get_color())
            #     else:
            #         temp, = ax.plot(self.times, data_temp[:, i], label="%.1f uM" % (self.concs[i]*1E6), alpha=alpha, lw=lw, color=cmap.__next__())
            #         if hasattr(self, 'fnorm'):
            #             axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha, lw=lw, color=temp.get_color())
            #     lh.append(temp)
            # else:
            #     if i in self.outliers:
            #         temp, = ax.plot(self.times, data_temp[:, i], alpha=alpha_out, lw=lw, color=temp.get_color())
            #         if hasattr(self, 'fnorm'):
            #             axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha_out, lw=lw, color=temp.get_color())
            #     else:
            #         temp, = ax.plot(self.times, data_temp[:, i], alpha=alpha, lw=lw, color=temp.get_color())
            #         if hasattr(self, 'fnorm'):
            #             axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha, lw=lw, color=temp.get_color())
            #     print("Conc. double")
            
        # Add legend
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=lh, title="Ligand conc.")
        # ax.set_xlim([dat.iloc[4,1], dat.iloc[-1,1]])
        ax.set_xlabel('Time / s')
        if np.max(self.decays) < 1.01:
            ax.set_ylabel('Norm. fluorescence')
        else:
            ax.set_ylabel('Fluorescence / Counts')
        ax.set_xlim((np.nanmin(self.times), np.nanmax(self.times)))

        # Add hot/cold areas
        if hasattr(self, 'fnorm'):
            ax.axvspan(self.cold-1, self.cold, facecolor='blue', alpha=.5) #, zorder=-20)
            ax.axvspan(self.hot-1, self.hot, facecolor='red', alpha=.5) #, zorder=-20)
        fig.tight_layout()
        fig.show()

        # Save figure
        if smooth:
            fig.savefig(self.fn.replace('.xlsx', '_pconc_%.1EM_smooth_%i.pdf' % (self.pconc, smooth_window)))
            fig.savefig(self.fn.replace('.xlsx', '_pconc_%.1EM_smooth_%i.png' % (self.pconc, smooth_window)), dpi=600)
        else:
            fig.savefig(self.fn.replace('.xlsx', '_pconc_%.1EM.pdf' % self.pconc))
            fig.savefig(self.fn.replace('.xlsx', '_pconc_%.1EM.png' % self.pconc), dpi=600)
        return fig, ax 

