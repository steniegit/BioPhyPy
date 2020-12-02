# -*- coding: utf-8 -*-
"""
This module contains helpful functions used by the libspec module
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.insert(0,'/home/snieblin/work/libspec')
sys.path.insert(0,'/home/snieblin/work/')
# import libspec
import scipy.signal as ssi
import opusFC as ofc
from scipy.optimize import curve_fit
import ipdb
import re
import glob, os
from IPython.core.debugger import set_trace
import pickle
import pandas as pd
import itertools, copy
from scipy import sparse
from scipy.sparse.linalg import spsolve
# For color bars
import matplotlib.colors as mcolors
from matplotlib import cm

# Move this to helpers later on
def kd_unit(kd):
    '''
    Converts Kd to label with suitable unit
    '''
    # Adjust labels to mM, muM or nM
    if 1E-9 < kd < 1E-6:
        kd_label = "%.3g$\,\mathrm{nM}$" % (kd*1E9)
    elif 1E-6 < kd < 1E-3:
        kd_label = "%.3g$\,\mu$M" % (kd*1E6)
    elif 1E-3 < kd < 1:
        kd_label = "%.3g$\,\mathrm{mM}$" % (kd*1E3)
    else:
        kd_label = "%.1E$\,\mathrm{M}$" % kd
    return kd_label

def baseline_als(y, lam=1E15, p=0.001, niter=10):
    '''
    Baseline Correction with
    Asymmetric Least Squares Smoothing
    Developed by Eilers and Boelens
    https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf
    Adjusted for python 3.6 according to 
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    Input
    y: spectrum
    lam: lambda value for smoothness (general recommendation 1E2 < lam < 1E9)
    p: parameter for asymmetry (general recommendation: 0.001 < p < 0.1)
    niter: number of iterations
    Outputs baseline
    '''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def single_site(conc_lig, kd, conc_prot, t_bottom, t_top):
    '''
    Single-site model to fit apparent Kd from melting temperatures
    Formula from: 
    Vivoli, M., Novak, H.R., Littlechild, J.A., Harmer, N.J. 
    Determination of Protein-ligand Interactions Using Differential Scanning Fluorimetry.
    J. Vis. Exp. (91), e51809, doi:10.3791/51809 (2014).
    '''
    # Derived from dissociation constant: Kd = [P][L]/[PL]
    # With [L] = [L]_0 - [PL] and [P] = [P]_0 - [PL]
    # we get this expression for the complex concentration [PL]
    complex_conc = (conc_lig + conc_prot + kd)/2 - np.sqrt((conc_lig + conc_prot + kd)**2 /4 - conc_lig*conc_prot)
    # Complexation degree (0 to 1)
    compl_degree = complex_conc/conc_prot
    # Old equation, identical to the one now used
    #expr_sqrt = (conc_prot-kd-conc_lig+np.sqrt(((conc_prot+conc_lig+kd)**2)-(4*conc_prot*conc_lig))) / (2*conc_prot)
    # return t_bottom + (t_top - t_bottom)*(1-expr_sqrt)
    return t_bottom + (t_top - t_bottom)*compl_degree

def single_site_kd(conc_prot):
    '''
    This is an auxiliary function used for fitting
    It returns a function for the single site model
    for a fixed protein concentration
    '''
    def return_func(conc_lig, kd, t_bottom, t_top):
        return single_site(conc_lig, kd, conc_prot, t_bottom, t_top)
    return return_func

def single_site_kd_only(conc_prot, t_bottom, t_top):
    def return_func(conc_lig, kd):
        return single_site(conc_lig, kd, conc_prot, t_bottom, t_top)
    return return_func

def coop_model_jove(conc_lig, kd, n, t_bottom, t_top):
    '''
    Cooperative model to fit apparent Kd from melting temperatures
    Formula from: 
    Vivoli, M., Novak, H.R., Littlechild, J.A., Harmer, N.J. 
    Determination of Protein-ligand Interactions Using Differential Scanning Fluorimetry.
    J. Vis. Exp. (91), e51809, doi:10.3791/51809 (2014).
    '''
    return t_bottom + (t_top-t_bottom)* ((conc_lig/kd)**n)/ (1+((conc_lig/kd)**n)) 

def hill_model(conc_lig, kd, n, t_bottom, t_top):
    '''
    Cooperative model to fit apparent Kd from melting temperatures
    Gives the same results as coop_model_jove
    '''
    return t_bottom + (t_top-t_bottom)* ((conc_lig)**n)/ (kd+conc_lig**n)

class MS_data():
    ''' 
    To do
    Write function to export table
    When plotting, adjust ylim to spectrum in xlim
    Convert distance in kDa when mfact=1000
    bl correction, based on peak_find with inverted data and spline interpolation
    '''
    def __init__(self, fn='', mfact=1000):
        '''
        Initialize ms data
        fn: File name
        mfact: 1000 for kDa, 1 for Da
        '''
        self.fn = fn
        #self.path = path
        self.mfact = mfact
        self.peaks = []
        # Load data and do outlier rejection
        self.load()
        # Safety copy of raw data (not affected by data processing)
        self.raw = copy.deepcopy(self.spec)
        # Normalization flag
        self.norm = False
        return None

    def load(self, verbose=False):
        '''
        This function loads the file and
        interpolates the points (for later smoothing)
        So far there is no error check done
        '''
        spec = np.genfromtxt(self.fn)
        spec[:,0] /= self.mfact
        # Interpolate for equal spacing
        step = np.min(np.diff(spec[:,0]))
        xi = np.arange(np.min(spec[:,0]), np.max(spec[:,0])+step, step) 
        yi = np.interp(xi, spec[:,0], spec[:,1])
        self.spec = np.vstack([xi, yi]).T
        return None

    def smooth(self, sg_window=101, sg_pol=2):
        '''
        This function uses a Savitzky-Golay filter for smoothening the data
        sg_window: Window size for smoothing
        sg_pol:    Polynomial order for smoothing
        '''
        self.spec[:,1] = ssi.savgol_filter(self.spec[:,1], sg_window, sg_pol)
        return None

    def bl_correct(self, lam=1E15, p=0.0001, niter=10):
        '''
        Baseline Correction with
        Asymmetric Least Squares Smoothing
        Developed by Eilers and Boelens
        https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf
        Adjusted for python 3.6 according to 
        https://stackoverflow.com/questions/29156532/python-baseline-correction-library
        Input
        y: spectrum
        lam: lambda value for smoothness (general recommendation 1E2 < lam < 1E9)
        p: parameter for asymmetry (general recommendation: 0.001 < p < 0.1
        niter: number of iterations
        Outputs baseline
        For MS data higher values for lambda and lower for p might be necessary
        '''
        # Backup spectrum
        self.spec_before_bl = copy.deepcopy(self.spec)
        # Subtract baseline
        self.bl = baseline_als(self.spec[:,1], lam, p, niter)
        self.spec[:,1] -= self.bl
        return None
    

    def normalize(self, region=[]):
        '''
        This option normalizes the spectra
        '''
        if len(region)==2:
            x1 = np.argmin(np.abs(self.spec[:,0] - region[0]))
            x2 = np.argmin(np.abs(self.spec[:,0] - region[1]))
            maxval = np.max(self.spec[x1:x2,1])
            self.spec[:,1] /= maxval
        else:
            self.spec[:,1] /= np.max(self.spec[:,1])
        self.norm = True
        return None
            

    def peak_pick(self, params={}):
        '''
        To do: convert distance threshold in points to Da

        This function picks peaks with scipy.signal.find_peaks
        params: dictionary with optional parameters
          distance: distance threshold
          height: Required height of peaks. Either a number, None, 
                  an array matching x or a 2-element sequence of the former. 
                  The first element is always interpreted as the minimal and 
                  the second, if supplied, as the maximal required height.
          threshold : Required threshold of peaks, the vertical distance 
                      to its neighbouring samples
          distance: Required minimal horizontal distance (>= 1) in samples 
                    between neighbouring peaks. Smaller peaks are removed 
                    first until the condition is fulfilled for all remaining 
                    peaks.
          prominence: Required prominence of peaks. Either a number, None, 
                      an array matching x or a 2-element sequence of the 
                      former. The first element is always interpreted as the 
                      minimal and the second, if supplied, as the maximal 
                      required prominence. The prominence of a peak measures 
                      how much a peak stands out from the surrounding baseline 
                      of the signal and is defined as the vertical distance 
                      between the peak and its lowest contour line.
        '''
        # Convert distance to points
        #distance = distance // np.min(np.diff(self.spec[:,0]))
        peaks, info = ssi.find_peaks(self.spec[:,1], **params)
        self.peaks = peaks
        self.peak_info = info
        print("Found %i peaks" % len(self.peaks))
        print(self.peaks)
        return None

    # def bl_correction(self, params={}):
    #     '''
    #     Do baseline correction based on peak_finding with inverted data
    #     The same parameters as for peak_pick are used
    #     A spline curve will be calculated based on the minima and subtracted
    #     The baseline will be saved as self.bl
    #     '''
    #     peaks, info = ssi.find_peaks(self.spec[:,1]*(-1), **params)
    #     print("Found %i peaks for baseline" % len(peaks))
    #     print(peaks)
    #     return None

        
    def plot(self, fn='', xlim=[], ax=None):
        if ax==None:
            fig, ax = plt.subplots(1)
        else:
            fig = ax.figure
        if len(xlim) == 0:
            xlim = [np.min(self.spec[:,0]), np.max(self.spec[:,0])]
        ylim = [np.min(self.spec[:,1]), np.max(self.spec[:,1])]
        ax.set_title(ax.get_title() + '\n' + self.fn)
        ax.plot(self.spec[:,0], self.spec[:,1], label=self.fn.split('/')[-1].replace('.txt',''))
        if self.mfact == 1000:
            acc = "%.2f"
        else: acc = "%.0f"
        # Add peak labels
        if len(self.peaks) > 0:
            for peak in self.peaks:
                ax.plot([self.spec[peak,0], self.spec[peak,0]], [self.spec[peak,1]+0.01*ylim[1], self.spec[peak,1]+0.05*ylim[1]], color='k')
                ax.text(self.spec[peak,0], self.spec[peak,1]+0.06*ylim[1], acc % (self.spec[peak,0]), rotation='vertical', ha='center', va='bottom')
            ax.set_ylim([0, ylim[1]*1.27])
        # Set xlabel for lowest panel
        if self.mfact == 1000:
            ax.set_xlabel('m/z / kDa/e')
        elif self.mfact == 1:
            ax.set_xlabel('m/z / Da/e')
        if self.norm:
            ax.set_ylabel('Norm. counts')
            ax.set_yticks([])
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Counts')
        ax.set_xlim(xlim)
        fig.tight_layout()
        if len(fn) < 0:
            fig.savefig(fn)
        ax.legend()
        return fig, ax

class DLS_Data():
    
    def __init__(self, folder='', verbose=False, labels={}, pfolder=''):
        '''
        Initialize dls data
        folder: Folder with dat files
        labels: dictionary with labels (optional for plotting)
        pfolder: folder with pictures (optional)
        '''
        self.folder = folder
        self.labels = labels
        self.pfolder = pfolder
        # Load data and do outlier rejection
        self.load(verbose=verbose)
        self.reject_incomplete()
        return None
    
    def load(self, verbose=False):
        # Find files in folder
        fns = glob.glob(self.folder + '/*.dat')
        print("Found %i files in folder %s" % (len(fns), self.folder))
        fns.sort()
        # Create list for dictionary
        run, acf, acf_x, fit, fit_x, dis, dis_x, pos, spos, row, col, pics = [], [], [], [], [], [], [], [], [], [], [], []
        for fn in fns:
            # Extract position
            pos_ = fn.split('.')[0].split('-')[-1]
            pfns_ = self.pfolder + '/' + pos_ + '-0.png'
            if os.path.isfile(pfns_):
                #print("Picture %s found!" % pfns_)
                pics.append(pfns_)
            else:
                pics.append('')
            # Create second position with zero filling (easier for sorting later)
            spos_ = pos_[0] + pos_[1:].zfill(2)
            # Row, columns and run numbers
            row_ = pos_[0]
            col_ = int(pos_[1:])
            run_ = int(fn.split('.')[1].split('-')[1])
            with open(fn, 'r') as f:
                # Get acf
                acf_ = list(itertools.takewhile(lambda x: '&' not in x, 
                                                itertools.dropwhile(lambda x: '#acf0' not in x, f)))
                # Get fit 
                fit_ = list(itertools.takewhile(lambda x: '&' not in x, 
                                                itertools.dropwhile(lambda x: '#fit0' not in x, f)))
                # Get dis0 
                dis_ = list(itertools.takewhile(lambda x: '&' not in x, 
                                                itertools.dropwhile(lambda x: '#dis0' not in x, f)))
            acf_ = np.array([list(map(float, b.split())) for b in acf_[1:]])
            fit_ = np.array([list(map(float, b.split())) for b in fit_[1:]])
            dis_ = np.array([list(map(float, b.split())) for b in dis_[1:]])
            # Fill lists
            run.append(run_)
            acf.append(acf_)
            #acf_x = acf_[:,0]
            fit.append(fit_)
            #fit_x = fit_[:,0]
            dis.append(dis_)
            #dis_x = dis_[:,0]
            pos.append(pos_)
            row.append(row_)
            col.append(col_)
            spos.append(spos_)
        # Sort indices based on column, row and runnr
        inds = np.lexsort((run, spos)) #np.argsort(spos_)
        # Fill class 
        self.fns = [fns[i] for i in inds]
        self.run = [run[i] for i in inds]
        self.acf = [acf[i] for i in inds]
        self.fit = [fit[i] for i in inds]
        self.dis = [dis[i] for i in inds]
        self.pos = [pos[i] for i in inds]
        self.spos = [spos[i] for i in inds]
        self.row = [row[i] for i in inds]
        self.col = [col[i] for i in inds]
        self.keep = np.ones(len(self.pos))
        self.out = np.zeros(len(self.pos))
        self.pics = pics
        return None
    
    def reject_incomplete(self, verbose=False):
        '''
        This is to remove incomplete acf scans and generates
        arrays for acfs, fits and diss.
        The x-values are now stored in acf_x, fit_x and dis_x
        This makes averaging, plotting etc. easier
        '''
        # Get list with lengths
        acf_size = []
        for i in self.acf:
            acf_size.append(len(i))
        acf_size = np.array(acf_size)
        self.acf_size = acf_size
        # Determine median and throw out all non-median values
        size_median = int(np.median(self.acf_size))
        indices = np.argwhere(self.acf_size == size_median).squeeze()
        out_ind = np.argwhere(self.acf_size != size_median).squeeze()
        if verbose:
            print("Remove %i out of %i entries due to inconsistent acf length" % (len(self.run) - len(indices), len(self.run)))
            for index in out_ind:
                print(self.fns[index])
        # Modify entries
        self.run   = np.array([self.run[index] for index in indices])
        self.acf_x = np.array(self.acf[0][:,0])
        self.acf   = np.array([self.acf[index][:,1] for index in indices]).T
        self.fit   = [self.fit[index] for index in indices]
        #self.fit_x = np.array(self.fit[0][:,0])
        #self.fit   = np.array([self.fit[index][:,1] for index in indices])
        self.dis_x = np.array(self.dis[0][:,0])
        self.dis   = [self.dis[index] for index in indices]
        self.pos   = np.array([self.pos[index] for index in indices])
        self.spos  = np.array([self.spos[index] for index in indices])
        self.row   = np.array([self.row[index] for index in indices])
        self.col   = np.array([self.col[index] for index in indices])
        self.fns   = np.array([self.fns[index] for index in indices])
        self.keep  = self.keep[indices]
        self.out   = self.out[indices]
        self.pics  = np.array([self.pics[index] for index in indices])
        return None

    def reject_outliers_acf(self, tol='3percent', verbose=False):
        '''
        Outlier rejection based on acf 
        @param tol: The tolerance cutoff. Can be 'XXpercent' or 'XXunits', where XX is a number in any 
        format. With 'percent', the cutoff is defined as a percentage of the median, with 'units' it is
        absolute. All exposures for which the q-averaged scattering falls further than one cutoff 
        distance from the median are rejected.
        '''
        # Tolerance factor
        if 'percent' in tol:
            tol_fact = float(tol.split('percent')[0]) * .01
        else:
            raise Exception('Bad input')
        
        # Loop through positions
        for spos in np.unique(self.spos):
            # Use pos internally
            pos = self.pos[self.spos == spos][0]
            # calculate median and tolerances, here we take mean
            mean = np.median(self.acf[:, self.pos==pos], axis=1).reshape((-1,1)) 
            # work out which repeats to keep:
            # keep = np.arange(len(statistics))[np.abs(statistics-mean) < tolerance]
            keep = np.sum(np.abs(self.acf[:,self.pos==pos]-mean), axis=0) < tol_fact * np.sum(np.abs(mean))
            self.keep[self.pos==pos] = keep
            self.out[self.pos==pos] = np.invert(keep)
            if verbose:
                print('%s: RejectS() rejected %u out of %u repeats (%.0f%%)'%(pos, np.sum(self.pos==pos)-np.sum(keep), np.sum(self.pos==pos), float(np.sum(self.pos==pos)-np.sum(keep))/np.sum(self.pos==pos)*100))
        self.keep = self.keep.astype('int')
        self.out = self.out.astype('int')
        return None

    def average_pos(self, plot=False, plotall=False, savefig=False):
        '''
        Averages acf for each position
        plot: Plot results
        plotall: Even plot positions averaging cannot be done (e.g. only one spectrum or only outliers)
        savefig: saves each position as pos.pdf
        '''
        acf_average, pos_average = [], []
        for spos in np.unique(self.spos):
            pos = self.pos[self.spos==spos][0]
            if np.sum(self.pos==pos) < 2:
                print('%s: Less than 2 spectra available (before outlier rejection). Cannot do averaging' % pos)
                if plotall:
                    # Plot
                    fig, ax = plt.subplots(1)
                    if np.sum(self.keep[self.pos==pos]) > 0:
                        h1 = ax.semilogx(self.acf_x, sub_in, label='Keep (%i)' % np.sum(self.keep[self.pos==pos]), color='green', lw=.5, alpha=.5)
                    if np.sum(self.out[self.pos==pos]) > 0:
                        h3 = ax.semilogx(self.acf_x, sub_out, label='Out (%i)' % np.sum(self.out[self.pos==pos]), color='red', lw=.5, alpha=.5)
                    ax.legend()
                    fig.savefig(pos + '_avg.pdf')
                continue
            else:
                if np.sum(self.keep[self.pos==pos])>1:
                    sub = self.acf[:,self.pos==pos]
                    pic = self.pics[self.pos==pos][0]
                    sub_in = sub[:, np.argwhere(self.keep[self.pos==pos]).squeeze()]
                    sub_out = sub[:, np.argwhere(self.out[self.pos==pos]).squeeze()]
                    sub_av = np.average(sub_in, axis=1)
                    acf_average.append(sub_av)
                    pos_average.append(pos)
                    if plot:
                        # Plot
                        fig, ax = plt.subplots(1)
                        h1 = ax.semilogx(self.acf_x, sub_in, label='Keep (%i)' % np.sum(self.keep[self.pos==pos]), color='green', lw=.5, alpha=.5)
                        h2 = ax.semilogx(self.acf_x, acf_average[-1], label='Average', color='blue', lw=2, alpha=.5)
                        if np.sum(self.out[self.pos==pos]) > 0:
                            h3 = ax.semilogx(self.acf_x, sub_out, label='Out (%i)' % np.sum(self.out[self.pos==pos]), color='red', lw=.5, alpha=.5)
                            ax.legend(handles=[h1[0], h3[0], h2[0]])
                        else:
                            ax.legend(handles=[h1[0],  h2[0]])
                        ax.set_xlabel('Time / s')
                        ax.set_ylabel('ACF')
                        ax.set_xlim([np.min(self.acf_x), np.max(self.acf_x)])
                        try:
                            label = pos + ': ' +self.labels[pos]
                        except:
                            label = pos 
                        ax.set_title(label)
                        if len(pic) > 0:
                            pic = mpimg.imread(pic)
                            ax_pic = fig.add_axes([.3, .46, .4, .4])
                            ax_pic.imshow(pic)
                            ax_pic.axis('off')
                        fig.savefig(pos + '_avg.pdf')
                    #pdb.set_trace()
                else:
                    print("%s: Not more than one accepted spectrum available. Cannot do averaging" % pos)
        self.acf_average = np.array(acf_average)
        self.pos_average = np.array(pos_average)

def fit_octet(folder, sensor=0, seg_rise=3, seg_decay=4, func='biexp', plot=True, conc=1E-6, order='a', norm=True, ptitle='', leg='', save_all=False, loading=np.NaN, ref_fn='', flip=False):
    '''
    Wrapper script to determine KD from 
    Octet data
    folder: folder name
    sensor: which sensor to take
    seg_rise: segment for rise
    seg_decay: segment for decay
    function: 'exp' or 'biexp'
    conc: analyte concentration
    order: 'a' based on fit factor, or 'sort' based on value
    norm: Normalize values (0-1)
    save_all: if this is True, then also the rise/decay and the fitted rise/decay are saved ! This takes space!!!
    flip: Flip data, i.e. decaying signal when binding; happens for instance for lipid vesicle binding
    Returns fitvalues_rise, fitvalues_decay and fit
    If plot=True it returns fig, axs, fitvalues_rise, fitvalues_decay and fit
    '''
    # Load data
    try:
        rise, rise_time = extract_octetSeg(folder, seg=seg_rise, sensor=sensor, norm=norm)
        decay, decay_time = extract_octetSeg(folder, seg=seg_decay, sensor=sensor, norm=norm)
        if len(ref_fn) > 0:
            # Load reference
            ref_rise, ref_rise_time = extract_octetSeg(folder, seg=seg_rise, sensor=sensor, norm=norm)
            ref_rise, ref_rise_time = extract_octetSeg(folder, seg=seg_rise, sensor=sensor, norm=norm)
            rise = rise - ref_rise
            decay = decay - ref_decay
        if flip:
            rise = - rise +1
            decay = - decay +1
            print("Flipped data!")
    except:
        print("Could not load data! Exiting")
        return None, None

    # Prepare fitting function and starting parameters
    if func=='biexp':
        func_rise = biexp_rise
        func_decay = biexp_decay
        guess = (0.5, 0.5, .1, .1, np.max(rise))
    elif func=='exp':
        func_rise = exp_rise
        func_decay = exp_decay
        guess = (0.5, 1) #0, 1)
    else:
        print("Error: Unknown function!!")
        return None

    # Perform rise fitting
    try:
        fitvalues_rise, fit_pcov_rise = curve_fit(func_rise, rise_time, rise, p0=guess, bounds=(0, np.inf))
    except:
        print("Could not fit data. Exiting")
        return None, None
    fitted_rise = func_rise(rise_time, *fitvalues_rise)
    r2_rise = r_sq(rise, fitted_rise)

    # Perform exp decay fitting
    try:
        fitvalues_decay, fit_pcov_decay = curve_fit(func_decay, decay_time, decay, p0=guess, bounds=(0, np.inf))
    except:
        print("Could not fit data. Exiting")
        return None, None
    fitted_decay = func_decay(decay_time, *fitvalues_decay)
    r2_decay = r_sq(decay, fitted_decay)

    # Plot data
    if plot:
        fig, axs = plt.subplots(2)
        ax = axs[0]
        ax.plot(rise_time, rise, '.')
        ax.plot(rise_time, fitted_rise)
        ax = axs[1]
        ax.plot(decay_time, decay, '.')
        ax.plot(decay_time, fitted_decay)
        for ax in axs:
            ax.set_xlabel('Time / s')
            if norm:
                ax.set_ylabel('Norm. binding')
            else:
                ax.set_ylabel('Binding / nm')
    else:
        fig = None
        axs = None

    # Determine KD
    fit = output_fit(fitvalues_rise, fitvalues_decay, r2_rise=r2_rise, r2_decay=r2_decay, conc=conc, order=order)
    # Extend fit dictionary
    if save_all:
        fit['decay_time']   = decay_time
        fit['decay']        = decay
        fit['fitted_decay'] = fitted_decay
        fit['rise_time']    = rise_time
        fit['rise']         = rise
        fit['fitted_rise']  = fitted_rise
    fit['seg_rise']     = seg_rise
    fit['seg_decay']    = seg_decay
    fit['func']         = func
    fit['order']        = order
    fit['sensor']       = sensor
    fit['folder']       = folder
    fit['ptitle']       = ptitle
    fit['leg']          = leg
    fit['loading']      = loading
    # Save dictionary in fits
    fn_pickle = "" + folder.replace('./','').replace('/', '_') + "sensor%i_riseseg%i_decayseg%i_func%s.p" % (sensor, seg_rise, seg_decay, func)
    pickle.dump(fit, open(fn_pickle, 'wb'))
    print("Fit results saved in %s" % fn_pickle)
    if plot:
        return fig, axs, fitvalues_rise, fitvalues_decay, fit
    else:
        return fitvalues_rise, fitvalues_decay, fit


    
def extract_octetSeg(folder, seg=3, sensor=1, norm=True):
    '''
    This extracts a segment from 
    a Octet data set. This can for 
    instance be used to fit a 
    rise or decay later.
    folder: folder name
    seg: which segment to use
    sensor: wich sensor to use
    returns segment and segment_time
    '''
    try:
        seg = int(seg)
        sensor = int(sensor)
    except:
        print("Error: seg and/or sensore could not be converted to integer!")
        return None
    # Load data
    data, act_times = get_octet(data_folder = folder)
    # Determine indices
    indices = (act_times[seg-1] < data[:,0]) * (data[:,0] < act_times[seg])
    segment = data[indices,sensor*2+1]
    seg_time = data[indices,0] - act_times[seg-1]
    # Normalize
    if norm:
        segment -= np.min(segment)
        segment /= np.max(segment)
    return segment, seg_time
    

def output_fit(fitvalues_rise, fitvalues_decay, r2_rise=np.nan, r2_decay=np.nan, conc=1E-6, order='a'):
    '''
    This is a simple helper function
    to sort and output results from a 
    kinetic fit (Octet).
    fitvalues_rise: from fit
    fitvalues_decay: from fit
    conc: analyte concentration
    order: 'a' based on fit factor, or 'sort' based on value
    '''
    # Check if length of parameters are the same
    if len(fitvalues_rise) != len(fitvalues_decay):
        print("Lengths of fitting parameters for rise and decay are not identical! Exiting!")
        return None
    # Determine if exp or biexp fit parameters were used as input
    if len(fitvalues_rise) ==2:
        func = 'exp'
    elif len(fitvalues_rise) == 5:
        func = 'biexp'
    else:
        print("Parameter of unknown function given! Exiting")
        return None
    # Create dictionary with fit values
    fit = {}
    # If biexponential fit: Sort values
    if func == 'exp':
        kobs  = fitvalues_rise[0]
        kobsmax = fitvalues_rise[1]
        kdiss = fitvalues_decay[0]
        kdissmax = fitvalues_decay[1]
    elif func == 'biexp':
        if order=='a':
            print('Sorting based on pre-exponential factor')
            if fitvalues_rise[0] > fitvalues_rise[1]:
                kobsmax1 = fitvalues_rise[0] / np.sum(fitvalues_rise[:2])
                kobsmax2 = fitvalues_rise[1] / np.sum(fitvalues_rise[:2])
                kobs1 = fitvalues_rise[2]
                kobs2 = fitvalues_rise[3]
            else:
                kobsmax1 = fitvalues_rise[1] / np.sum(fitvalues_rise[:2])
                kobsmax2 = fitvalues_rise[0] / np.sum(fitvalues_rise[:2])
                kobs1 = fitvalues_rise[3]
                kobs2 = fitvalues_rise[2]
            # Sort values decay
            if fitvalues_decay[0] > fitvalues_decay[1]:
                kdissmax1 = fitvalues_decay[0] / np.sum(fitvalues_decay[:2])
                kdissmax2 = fitvalues_decay[1] / np.sum(fitvalues_decay[:2])
                kdiss1 = fitvalues_decay[2]
                kdiss2 = fitvalues_decay[3]
            else:
                kdissmax1 = fitvalues_decay[1] / np.sum(fitvalues_decay[:2])
                kdissmax2 = fitvalues_decay[0] / np.sum(fitvalues_decay[:2])
                kdiss1 = fitvalues_decay[3]
                kdiss2 = fitvalues_decay[2]
        elif order=='sort':
            print('Sorting based on k values')
            if fitvalues_rise[2] > fitvalues_rise[3]:
                kobsmax1 = fitvalues_rise[0] / np.sum(fitvalues_rise[:2])
                kobsmax2 = fitvalues_rise[1] / np.sum(fitvalues_rise[:2])
                kobs1 = fitvalues_rise[2]
                kobs2 = fitvalues_rise[3]
            else:
                kobsmax1 = fitvalues_rise[1] / np.sum(fitvalues_rise[:2])
                kobsmax2 = fitvalues_rise[0] / np.sum(fitvalues_rise[:2])
                kobs1 = fitvalues_rise[3]
                kobs2 = fitvalues_rise[2]
            # Sort values decay
            if fitvalues_decay[2] < fitvalues_decay[3]:
                kdissmax1 = fitvalues_decay[0] / np.sum(fitvalues_decay[:2])
                kdissmax2 = fitvalues_decay[1] / np.sum(fitvalues_decay[:2])
                kdiss1 = fitvalues_decay[2]
                kdiss2 = fitvalues_decay[3]
            else:
                kdissmax1 = fitvalues_decay[1] / np.sum(fitvalues_decay[:2])
                kdissmax2 = fitvalues_decay[0] / np.sum(fitvalues_decay[:2])
                kdiss1 = fitvalues_decay[3]
                kdiss2 = fitvalues_decay[2]
        else:
            print('Order string not recognized! Exiting!')
            return None

    if func == 'biexp':
        # Calculate kons from kobs, kdiss and conc
        kon1 = (kobs1 + kdiss1)/conc
        kon2 = (kobs2 + kdiss2)/conc
        # Calculate KDs
        KD1  = kdiss1 / kon1
        KD2  = kdiss2 / kon2
        # Print results
        print("k_obs1: %.2E 1/s (k_obs_max1: %.2f)" % (kobs1, kobsmax1))
        print("k_obs2: %.2E 1/s (k_obs_max2: %.2f)" % (kobs2, kobsmax2))
        if not np.isnan(r2_rise):
            print("R2 for rise: %.4f" % r2_rise)
        print("k_diss1: %.2E 1/s (k_diss_max1: %.2f)" % (kdiss1, kdissmax1))
        print("k_diss2: %.2E 1/s (K_diss_max2: %.2f)" % (kdiss2, kdissmax2))
        if not np.isnan(r2_decay):
            print("R2 for decay: %.4f" % r2_decay)
        print("c(analyte) = %.2E M" % conc)
        print("k_on1: %.2E 1/sM" % (kon1))
        print("k_on2: %.2E 1/sM" % (kon2))
        print("KD1: %.2E M" % KD1)
        print("KD2: %.2E M" % KD2)
        fit = {'k_obs1':  kobs1,
               'k_obs2':  kobs2,
               'k_obsmax1': kobsmax1,
               'k_obsmax2': kobsmax2,
               'k_diss1': kdiss1,
               'k_diss2': kdiss2,
               'k_dissmax1': kdissmax1,
               'k_dissmax2': kdissmax2,
               'r2_rise': r2_rise,
               'r2_decay': r2_decay,
               'conc': conc,
               'k_on1': kon1,
               'k_on2': kon2,
               'KD1': KD1,
               'KD2': KD2
        }
    elif func == 'exp':
        # Calculate kons from kobs, kdiss and conc
        kon = (kobs + kdiss)/conc
        # Calculate KDs
        KD  = kdiss / kon
        # Print results
        print("k_obs: %.2E 1/s (k_obs_max: %.2f)" % (kobs, kobsmax))
        if not np.isnan(r2_rise):
            print("R2 for rise: %.4f" % r2_rise)
        print("k_diss: %.2E 1/s (k_diss_max: %.2f)" % (kdiss, kdissmax))
        if not np.isnan(r2_decay):
            print("R2 for rise: %.4f" % r2_decay)        
        print("c(analyte) = %.2E M" % conc)
        print("k_on: %.2E 1/sM" % (kon))
        print("KD: %.2E M" % KD)
        fit = {'k_obs':  kobs,
               'k_obsmax': kobsmax,
               'k_diss': kdiss,
               'k_dissmax': kdissmax,
               'r2_rise': r2_rise,
               'r2_decay': r2_decay,
               'conc': conc,
               'k_on': kon,
               'KD': KD,
        }
    return fit

##### The following four kinetic functions are from Godfrey
def exp_rise(x, k, c):
    return c*(1 - np.exp(-k*x))

def exp_decay(x, k, c):
    return c*np.exp(-k*x)
    
def biexp_rise(x, a1, a2, k1, k2, c):
    return c*(1 - a1*np.exp(-k1*x) - a2*np.exp(-k2*x))

# might need one with a pre-exponential factor if decay starts with a value other than 1.0
def biexp_decay(x, a1, a2, k1, k2, c):
    return c*(a1*np.exp(-k1*x) + a2*np.exp(-k2*x))

def r_sq(data, fit):
    '''
    R squared
    '''
    mean_data = np.mean(data)
    ss_tot = np.sum(np.power(fit - mean_data, 2))
    ss_res = np.sum(np.power(data - fit, 2))
    return 1 - (ss_res/ss_tot)

##### End: kinetic functions from Godfrey

def get_octet(data_folder=''):
    ''' 
    This function is to read Octet data
    '''
    if len(data_folder) == 0:
        print("Please specify data folder!\n Exiting")
        return None
    # Check if folder exists
    if (not os.path.isdir(data_folder)):
        raise Exception("Folder does not exist!")
        return None
    # Check if file is there
    if not (os.path.isfile(data_folder + '/RawData0.xls')):
        raise Exception("File %s not found! Exiting" % fn)
        return None
    # Load data
    data = np.genfromtxt(data_folder + 'RawData0.xls', skip_header=2)

    # Get boundaries between steps
    fns = glob.glob(data_folder +'*.frd')
    # Sort it
    fns.sort()
    # Take first one (does not matter)
    fn = fns[0]
    # Open file and find 'actual times'
    act_times = []
    with open(fn,'r') as f:
        for line in f:
            if "ActualTime" in line:
                act_time = float(re.search('<ActualTime>(.+?)</ActualTime>', line).group(1))
                act_times.append(act_time)
    act_times = np.array(act_times)
    act_times = np.cumsum(act_times)
    return data, act_times
    

def plot_octet(data_folder='', seg_labels=['BL1', 'Load', 'BL2', 'Assoc.', 'Diss.'], ptitle='', legs='', l_labels=[], l_posis=[], b_labels=[], b_posis=[], a_labels=[], a_posis=[], d_labels=[], d_posis=[], sensors=[]):
    '''
    This function plots Octet data
      Input:
    data_folder: path with octet data (including raw data xls)
    seg_labels:  labels for segments
    ptitle:      plot title
    legs:        Place legend with these entries next to plot
                 If left empty, no legend will be shown
    l_labels, l_posis: Labels for loading
    b_labels, b_posis: Labels for BL2
    a_labels, a_posis: Labels for association
    d_labels, d_posis: Labels for dissociation
      Output:
    fig, ax: Figure and axis handle
    '''

    if len(data_folder) == 0:
        print("Please specify data folder!\n Exiting")
        return None
    # Load data
    data = np.genfromtxt(data_folder + 'RawData0.xls', skip_header=2)

    # Get boundaries between steps
    fns = glob.glob(data_folder +'*.frd')
    # Sort it
    fns.sort()
    # Take first one (does not matter)
    fn = fns[0]
    # Open file and find 'actual times'
    act_times = []
    with open(fn,'r') as f:
        for line in f:
            if "ActualTime" in line:
                act_time = float(re.search('<ActualTime>(.+?)</ActualTime>', line).group(1))
                act_times.append(act_time)
    act_times = np.array(act_times)
    act_times = np.cumsum(act_times)

    # Create figure
    #fig, axs = plt.subplots(1)
    #ax = axs
    fig = plt.figure(figsize=[8,5])
    ax = fig.add_axes([.1, .1, .7, .75])

    # Plot data
    hps = [] # Plot handles
    if len(sensors) == 0:
        sensors = np.arange(data.shape[1]//2-1)
        plot_it = True
    else:
        plot_it = False
        sensors = np.array(sensors)
    for i in sensors: # Last two columns contain temperature information
        if len(legs)>0:
            hp, = ax.plot(data[:,i*2],data[:,i*2+1], label=legs[i], lw=1)
        else:
            hp, = ax.plot(data[:,i*2], data[:,i*2+1], label=str(i), lw=1)
        hps.append(hp)
    if len(legs)>0:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Add legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Add borders
    for time in act_times:
        ax.axvline(time, color='grey', linestyle=':', lw=.5)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Binding / nm')
    ax.set_xlim([0, act_times[-1]])
    #ax.set_ylim([0, 4])

    # Determine middle for labels
    temp = [0] + list(act_times)
    temp = np.array(temp[:-1])
    middles = []
    #set_trace()
    for i in range(len(act_times)):
        middles.append(.5*(temp[i]+act_times[i]))
    # Add labels
    for i in range(len(act_times)):
        pos = (act_times[i] - np.min(act_times))
        if i < len(seg_labels):
            ax.text(middles[i], np.max(data[:,sensors*2+1])*1.07, seg_labels[i], ha='center')
    # Labels in loading
    for i in range(len(l_labels)): 
        ax.text(middles[1]*1.1, l_posis[i], l_labels[i], ha='center', color=hps[i].get_color()) # ,bbox=dict(facecolor='white', alpha=.5) $\,\mu$g/ml #bbox=dict(facecolor='white', alpha=.5)
    # Labels for BL2
    for i in range(len(b_labels)): 
        ax.text(middles[2], b_posis[i], b_labels[i], ha='center', color=hps[i].get_color())
    # Labels for association
    for i in range(len(a_labels)): 
        ax.text(middles[3], a_posis[i], a_labels[i], ha='center', color=hps[i].get_color())
    # Labels for dissociation
    for i in range(len(d_labels)): 
        ax.text(middles[4], d_posis[i], d_labels[i], ha='center', color=hps[i].get_color())
    # Set title
    ht = ax.set_title(ptitle)
    ht.set_position((.5, 1.1))
    # Add grid
    #ax.grid()

    # Save figure
    if plot_it:
        fig.savefig(data_folder + '/' + data_folder.split('/')[-2] + '.pdf')
        print("Figure saved as %s" % (data_folder + '/' + data_folder.split('/')[-2] + '.pdf'))
    else:
        print("Figure not saved!")

    return fig, ax
        

def mult_gauss(x, *params):
    '''
    Multiple gaussian function
    Inputs x values and gaussian parameters
    Outputs y values
    '''
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

def mult_lorentz(x, *params):
    '''
    Multiple lorentzian function
    Inputs x values and gaussian parameters
    Outputs y values
    '''
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        tau = params[i+1]
        wid = params[i+2]
        y = y + tau * wid / (2*np.pi) / ((x-ctr)**2+(wid/2)**2)
    return y

def gauss(x,center,height,fwhm=10):
    """ Single gaussian function                                     
    """
    c=fwhm/(2*np.sqrt(2*np.log(2)))
    return height*np.exp(-(x-center)**2/(2*(c**2))
)

def extract_int(fn, scalf=1):
    '''
    Extracts intensities from Turbomole
    intense.out file
    fn: file name
    scalf: frequency scaling factor
    '''
    # Open file
    with open(fn) as f:
        s = f.readlines()
    # extract lines with frequencies and intensities
    freq=[]
    ir_int=[]
    raman_parpar, raman_ortort, raman_ortunpol=[], [], []
    # Extract data
    for i in range(0,len(s)):
        if '     frequency' in s[i]:
            tmpline=s[i].split()
            del tmpline[0]
            freq.extend(tmpline)
        if 'intensity (km/mol)  ' in s[i]:
            tmpline=s[i].split()
            del tmpline[0]
            del tmpline[0]
            ir_int.extend(tmpline)
        if  '    (par,par)' in s[i]:
            tmpline=s[i].split()
            del tmpline[0]
            raman_parpar.extend(tmpline)
        if  '    (ort,ort)' in s[i]:
            tmpline=s[i].split()
            del tmpline[0]
            raman_ortort.extend(tmpline)
        if  '    (ort,unpol)' in s[i]:
            tmpline=s[i].split()
            del tmpline[0]
            raman_ortunpol.extend(tmpline)
        
    # Delete imaginary frequencies
    counter=0
    while "i" in freq[counter]:
        print("Imaginary frequency detected! "+ freq[counter])
        freq[counter]=freq[counter].replace('i','-')
        counter+=1
    # Delete rotations and translations
    freq[0:6]=[]
    ir_int[0:6]=[]
    raman_parpar[0:6]=[]
    raman_ortort[0:6]=[]
    raman_ortunpol[0:6]=[]
    # Create arrays
    print("Scaling factor for last column: " + str(scalf))
    freq = np.array(list(map(float,freq)))
    freq_scalf = freq * scalf
    raman_parpar = np.array(list(map(float,raman_parpar)))
    raman_ortort = np.array(list(map(float,raman_ortort)))
    raman_ortunpol = np.array(list(map(float,raman_ortunpol)))
    ir_int = np.array(list(map(float,ir_int)))
    #freq_scalf = scalf * freq
    # Create structured array
    out_array = np.zeros(len(freq),
                         dtype=[('freq', 'float32'),
                                ('freq_scalf', 'float32'),
                                ('ir_int', 'float32'),
                                ('raman_parpar', 'float32'),
                                ('raman_ortort', 'float32'),
                                ('raman_ortunpol', 'float32')])
    out_array['freq'] = freq
    out_array['freq_scalf'] = freq_scalf
    out_array['ir_int'] = ir_int
    out_array['raman_parpar'] = raman_parpar
    out_array['raman_ortort'] = raman_ortort
    out_array['raman_ortunpol'] = raman_ortunpol
    return out_array

def create_spec(ints, x=np.linspace(1000,1900,1000), which_int='raman_parpar'):
    '''
    This inputs a list of frequencies and intensities and
    outputs an array with the individual gaussian curves
    '''
    # Initialize array
    arr = np.zeros((len(x), len(ints)))
    # Decide whether to take Raman or IR
    # Plot spectra
    for i in range(0,len(ints['freq_scalf'])):
        arr[:,i]= gauss(x,ints['freq_scalf'][i],ints[which_int][i])
    return arr

def plot_spec(ax, x, spec, which_int='raman_parpar', color='blue'):
    '''
    This plots a spectrum
    with annotated frequencies
    ax: axis handle
    x_spec:  x-values
    spec: spectrum (y-values)
    which_int: 'ir_int', 'raman_ortort', 'raman_parpar' or 'raman_ortunpol'
    '''
    # Plot model
    for i in range(0,spec.shape[1]):
        ax.plot(x, spec[:,i], '--', color='grey')
        # Determine maximum for each curve
        if np.max(spec[:,i]) > 0:
            y_max = np.max(spec[:,i])
            x_max = x[np.argmax(spec[:,i])]
            # Add labels
            ax.annotate("{:.0f}".format(x_max),xy=(x_max,y_max),ha='center')
    ax.plot(x, np.sum(spec, axis=1), color='blue')
    return None

def process_bruker(fn, spec_lim=[2100, 2200], peak_lim=[2145, 2175], p_order=6, sg_window=13, sg_poly=2, guess=[2155, 0.2, 5, 2165, 1, 5], func_type='gauss', gauss_pos = 'deriv', fit_tol=0.001, norm=True, gauss_lim = [2155, 2172]):
    '''
    This is a function that reads in a opus file 
    and returns an array with these columns
    x-values y-raw y-smooth y-bl y-bl-corrected
    spec_lim:  Spectral boundaries
    peak_lim:  Boundaries for peaks, in this region the weighing factor for the bl polynomial is set to zero!
    p_order:   Polynomial order for baseline
    sg_order:  Polynomial order for savitzgy golay smoothing
    sg_window: Window size for savitzgy golay smoothing
    guess:     Guess for fits
    func:      Function type for fits ('gauss' or 'lorentz')
    gauss_pos: Positions for gauss fit.
               Can be 'guess' or 'deriv'
    gauss_lim: Region in which to search for 2nd deriv zeros for
               gaussian fits
    fit_tol:   Tolerance for gaussian fit when using 'deriv'
    norm:      Normalize baseline corrected spectrum
    '''

    # Savitzky-Golay smoothing parameters
    # Function for fit
    func = mult_gauss
    if func_type=='gauss':
        func = mult_gauss
    elif func_type=='lorentz':
        func = mult_lorentz
    else:
        print("No proper function type given: Please select either 'gauss' or 'lorentz'")
        print("Exiting")
        return None
    
    # Load data
    dbs = ofc.listContents(fn)
    data = {}
    for db in dbs:
        data[str(db[0])] = ofc.getOpusData(fn, db)
    #print(data.keys())

    # Chose AB as spectrum
    spec_full = np.vstack((data['AB'].x, data['AB'].y)).T
    # Convert to mOD
    spec_full[:,1] = spec_full[:,1]*1000

    # If x values are decreasing, flip matrix
    if spec_full[0,0] > spec_full[-1,0]:
        spec_full = np.flipud(spec_full)
    
    # Extract data between these boundaries
    limits = spec_lim
    pos = [np.argmin(np.abs(spec_full[:,0] - limits[0])), np.argmin(np.abs(spec_full[:,0] - limits[1]))]
    # Sort it
    #pos = np.sort(pos)
    # x values
    x_val = np.array(spec_full[pos[0]:pos[1], 0]).transpose()
    # Only take these
    spec_raw = np.array(spec_full[pos[0]:pos[1], 1]).transpose()
    # Smooth it
    spec_smooth = np.array(ssi.savgol_filter(spec_full[pos[0]:pos[1], 1], sg_window, sg_poly)).transpose()
    # Get second derivative
    deriv = np.gradient(np.gradient(spec_smooth))

    # Calculate weighting matrix in range
    peak_range = peak_lim
    peak_pos = [np.argmin(np.abs(x_val - peak_range[0])), np.argmin(np.abs(x_val - peak_range[1]))]
    peak_pos = np.sort(peak_pos)
    min_pos = np.argmin(np.abs(spec_smooth - np.min(spec_smooth)))
    w_vector = np.ones(len(spec_smooth))
    w_vector[peak_pos[0]: peak_pos[1]] = 0

    # Polynomial fit
    p = np.polyfit(x_val, spec_smooth, p_order, w = w_vector)
    bl = np.polyval(p, x_val).transpose()
    bl_corrected = np.array(spec_smooth - bl).transpose()
    if norm==True:
        bl_corrected = bl_corrected / np.max(bl_corrected)
    # Determine minimum of the difference in peak_limits
    yshift = np.min(bl_corrected)
    # Shift the processed spectrum by yshift
    #bl_corrected = bl_corrected# - yshift

    # Gaussian fit
    # Cut out part for gaussian fit
    limits = spec_lim
    pos = [np.argmin(np.abs(x_val - limits[0])), np.argmin(np.abs(x_val - limits[1]))]
    # Only take these
    spec_part = np.array([x_val[pos[0]:pos[1]], bl_corrected[pos[0]:pos[1]]]).transpose()
    # Gaussian fit
    #if gauss_pos == 'guess':
    #    popt, pcov = curve_fit(func, spec_part[:,0], spec_part[:,1] + yshift, p0=guess, maxfev=10000, bounds=([2155, 0, 0, 2165, 0, 0], [2155.1, np.inf, np.inf, 2165.1, np.inf, np.inf]))
    if gauss_pos == 'deriv':
        # Get zero transitions in 2nd derivative
        gauss_pos = [np.argmin(np.abs(spec_part[:,0]-gauss_lim[0])), np.argmin(np.abs(spec_part[:,0]-gauss_lim[1]))]
        # Find minimum
        cent = np.argmin(deriv[gauss_pos[0]:gauss_pos[1]]) + gauss_pos[0] 
        pos1 = np.argmin(np.abs(deriv[gauss_pos[0]:cent])) + gauss_pos[0] + pos[0]
        pos1 = x_val[pos1]
        pos2 = np.argmin(np.abs(deriv[cent:gauss_pos[1]])) + cent + pos[0]
        pos2 = x_val[pos2]
        guess[0], guess[3] = pos1, pos2
    popt, pcov = curve_fit(func, spec_part[:,0], spec_part[:,1] + yshift, p0=guess, maxfev=10000, bounds=([guess[0]-fit_tol, 0, 0, guess[3]-fit_tol, 0, 0], [guess[0]+fit_tol, np.inf, np.inf, guess[3]+fit_tol, np.inf, np.inf]))
    fit = func(x_val, *popt)   
    # Get single gaussians
    sing_fit = []
    for i in range(len(popt)//3):
        sing_fit.append(func(x_val, *popt[i*3:i*3+3]))
    sing_fit=np.array(sing_fit).transpose()
    
    # Create empty array
    output = np.recarray((len(x_val)), 
                         dtype=[('x', '>f4'), ('raw', '>f4'), ('smooth', '>f4'), ('deriv', '>f4'), ('bl', '>f4'), ('blcorr', '>f4'), ('fit', '>f4'), ('fit1', '>f4'), ('fit2', '>f4')])
    output['x'] = x_val
    output['raw']     = spec_raw
    output['smooth']  = spec_smooth
    output['deriv']   = deriv
    output['bl']      = bl
    output['blcorr']  = bl_corrected
    output['fit']     = fit
    output['fit1']    = sing_fit[:,0]
    output['fit2']    = sing_fit[:,1]
    deriv_0 = [guess[0], guess[3]]
    
    return output, popt, deriv_0

def quick_test(fn, spec_lim=[2100, 2200], peak_lim=[2145, 2175], p_order=6, sg_window=13, sg_poly=2, guess=[2155, 0.2, 5, 2165, 1, 5], func_type='gauss', gauss_pos = 'deriv', fit_tol=10, norm=False, gauss_lim = [2155, 2172], plot=True, title='', plot_fits=True):
    '''
    Just a quick way to plot data and fits
    directly from the 
    lr_position: Move band position labels to left and right
    '''
    # Get colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # Load data
    #spec, spec_opt, deriv_0 = process_bruker(fn, peak_lim=peak_lim, gauss_pos = 'deriv', guess=[2161, 0.2, 5, 2168, 0.2, 5], fit_tol=fit_tol)
    spec, spec_opt, deriv_0 = process_bruker(fn, spec_lim=spec_lim, peak_lim=peak_lim, p_order=p_order, sg_window=sg_window, sg_poly=sg_poly, guess=guess, func_type=func_type, gauss_pos=gauss_pos, fit_tol=fit_tol, norm=norm, gauss_lim=gauss_lim)
    if not plot:
        return None, None, spec_opt, spec
    # Create figure
    fig, axs = plt.subplots(3,1, sharex=True)
    fig.canvas.set_window_title(fn)
    # Plot raw, smoothed and BL
    ax = axs[0]
    ax.plot(spec.x, spec.raw, label='Raw spectrum', color=colors[1])
    hs, = ax.plot(spec.x, spec.smooth, '--', label='Smoothed spectrum', color=colors[0])
    ax.plot(spec.x, spec.bl, label='Baseline', color=colors[2])
    ax.legend(loc='lower right')
    ax.set_ylabel('Absorb. / mOD')
    # Plot deriv
    ax = axs[1]
    ax.plot(spec.x, spec.deriv, label='2nd derivative')
    ax.axhline(color='grey', linestyle='--', lw=1, zorder=-20)
    for pos in deriv_0:
        ax.axvline(pos, color='grey', linestyle='--', lw=1, zorder=-20)
    # Plot labels
    for i, pos in enumerate(deriv_0):
        if i==0:
            ax.text(pos, ax.get_ylim()[1]*1.1, '%.0f' % pos, ha='right')
        else:
            ax.text(pos, ax.get_ylim()[1]*1.1, '%.0f' % pos, ha='left')
    #pos = deriv_0[0]; ax.text(pos, ax.get_ylim()[1]*1.05, '%.0f' % pos, ha='right')
    #pos = deriv_0[1]; ax.text(pos, ax.get_ylim()[1]*1.05, '%.0f' % pos, ha='left')
    ax.legend()
    ax.set_ylabel('$\Delta\Delta$(Abs)') # / mOD/cm$^-2$')
    # Plot fits
    ax = axs[2]
    ax.plot(spec.x, spec.blcorr, label='Experimental', color=hs.get_color())
    if plot_fits:
        hf, = ax.plot(spec.x, spec.fit, '--', label='Fit', color=colors[3])
        ax.plot(spec.x, spec.fit1, ':', color=hf.get_color())
        ax.plot(spec.x, spec.fit2, ':', color=hf.get_color())
        #ax.plot(spec.x, spec.fit1 + spec.fit2, label='Fit')
        # Plot labels
        for i in range(len(spec_opt) // 3):
            if i==0:
                #ax.text(spec_opt[i*3], spec_opt[i*3+1], '%.0f(%.0f)' % (spec_opt[i*3], spec_opt[i*3+2]), ha='right', va='center')
                if spec_opt[0] < spec_opt[3]:
                    ax.text(spec_opt[i*3], ax.get_ylim()[1]*1.05, '%.0f(%.0f)' % (spec_opt[i*3], spec_opt[i*3+2]), ha='right', va='center')
                else:
                    ax.text(spec_opt[i*3], ax.get_ylim()[1]*1.05, '%.0f(%.0f)' % (spec_opt[i*3], spec_opt[i*3+2]), ha='left', va='center')
            else:
                if spec_opt[0] < spec_opt[3]: 
                    ax.text(spec_opt[i*3], ax.get_ylim()[1]*1.05, '%.0f(%.0f)' % (spec_opt[i*3], spec_opt[i*3+2]), ha='left', va='center')
                else:
                    ax.text(spec_opt[i*3], ax.get_ylim()[1]*1.05, '%.0f(%.0f)' % (spec_opt[i*3], spec_opt[i*3+2]), ha='right', va='center')
            # Add ratio
            if spec_opt[0] < spec_opt[3]:
                ratio = spec_opt[1] / spec_opt[4]
            else:
                ratio = spec_opt[4] / spec_opt[1]
            posx = 2200 #ax.get_xlim()[1] #np.mean([spec_opt[0], spec_opt[3]])
            posy = 0.5*np.mean([spec_opt[1], spec_opt[4]])
            ax.text(posx, posy, "Amp. ratio: %.1f " % ratio, ha='right')
            ax.axvline(spec_opt[i*3], color='grey', linestyle='--', lw=1, zorder=-20)
    ax.set_xlim([np.min(spec.x), np.max(spec.x)])
    ax.legend()
    # Label axes
    ax.set_xlabel('Wavenumber / cm$^{-1}$')
    if norm:
        ax.set_ylabel('Norm. abs.')
    else:
        ax.set_ylabel('Absorb. / mOD')
    # Put filename in figure and windows title
    if len(title) ==0:
        fig.suptitle(fn)
    else:
        fig.suptitle(title)
    fig.canvas.set_window_title(fn)
    if norm:
        fig.savefig(fn.replace(".","_") + "_norm.pdf")
    else:
        fig.savefig(fn.replace(".","_") + "_nonorm.pdf")
    return fig, axs, spec_opt, spec

def time_plot(specs, times, pos, plot=True):
    '''
    This script inputs several spectra and plots
    the signal intensities at a certain spectral
    position vs time
    spec: List time-dependent of spectra
    times: List of respective times
    pos: List of spectral positions

    It outputs a recarray with the intensities
    ints: list with intensities vs time
    '''
    # Check if length of spec is equal to length of times
    if len(specs) != len(times):
        print("Length of spec and times lists is not the same!\n Exciting")
        return None, None
    # Go through list of spectra and determine intensity
    ints = []
    for spec in specs:
        # Determine index
        ind = np.argmin(np.abs(spec[:,0] - pos))
        ints.append(spec[ind,1])
    return ints

def bl_correction(spec, xpos, tol=5, normlim=[628, 730], norm=True):
    '''
    !!! This version is deprecated !!!
    !!! Use bl_correction_new !!!
    This script does a baseline correction and extends
    two columns to the spectrum: the baseline and the
    bl subtracted spectrum
    spec: input spectrum
    xpos: position of baseline points
    tol: +- range in which to search for minimum
    '''
    print('!!! This version of bl_correction is deprecated !!!')
    print('!!! Please use bl_correction_new in the future !!!\n')
    import numpy as np
    from scipy import interpolate as interp
    # Extract x and y values
    x = spec[:,0]
    if x[0] > x[-1]:
        print("x-values are descending! Please flip before using bl_correction!\n Exiting function")
        return None
    y = spec[:,1]
    # Determine minimum positions around xpos
    minpos = []
    miny = []
    for pos in xpos:
        pos_start = np.argmin(np.abs(x - pos + tol))
        pos_end   = np.argmin(np.abs(x - pos - tol))
        if pos_start == pos_end:
            tempmin = pos_start
        else:
            tempmin = np.argmin(y[pos_start:pos_end])
        minpos.append(x[tempmin+pos_start])
        miny.append(y[tempmin+pos_start])
    # Cubic spline interpolation
    bl_func = interp.CubicSpline(minpos, miny)
    bl_y = bl_func(x)
    # Subtract bl
    y_sub = y - bl_y
    # Do normalisation only if norm=True
    if norm:
        # Find indices
        xnorm_lower = np.argmin(np.abs(x-normlim[0]))
        xnorm_upper = np.argmin(np.abs(x-normlim[1]))
        norm_curve = y_sub/np.max(y_sub[xnorm_lower:xnorm_upper])
    else:
        norm_curve = np.zeros(len(x)) * np.nan
    # Create output
    pos = np.vstack((minpos, miny)).T
    bl = np.hstack((spec, np.vstack((bl_y, y_sub, norm_curve)).T))
    return bl

def bl_correction_new(spec, xpos, tol=5, normlim=[628, 730], norm=True, smooth=False, sg_window=21, sg_pol=2):
    '''
    This script reads in raw data (2 columns), 
    does a baseline correction and smoothes data (optional)
    spec: input spectrum
    xpos: position of baseline points
    tol: +- range in which to search for minimum
    normlim: limits for normalization
    norm: normalize (True or False)
    smooth: Do Savitzky-Golay smoothing (True or False)
    sg_window: Savitzky Golay window (has to be odd number)
    sg_pol: Polynomial for SG filter
    The output is a recarray with the following fields
    out.x
    out.y
    out.y_smooth (if requested)
    out.y_blsub
    out.y_norm  (if requested)
    '''
    # Extract x and y values
    x = spec[:,0]
    if x[0] > x[-1]:
        print("x-values are descending! Please flip before using bl_correction!\n Exiting function")
        return None
    if smooth:
        y = spec[:,1]
        y_smooth = ssi.savgol_filter(y,21,2)
    else:
        y = spec[:,1]
        y_smooth = np.zeros(len(x)) * np.nan
    # Determine minimum positions around xpos
    minpos = []
    miny = []
    for pos in xpos:
        pos_start = np.argmin(np.abs(x - pos + tol))
        pos_end   = np.argmin(np.abs(x - pos - tol))
        if pos_start == pos_end:
            tempmin = pos_start
        else:
            tempmin = np.argmin(y[pos_start:pos_end])
        minpos.append(x[tempmin+pos_start])
        miny.append(y[tempmin+pos_start])
    # Cubic spline interpolation
    bl_func = interp.CubicSpline(minpos, miny)
    bl = bl_func(x)
    # Subtract bl
    y_sub = y - bl
    # Do normalisation only if norm=True
    if norm:
        # Find indices
        xnorm_lower = np.argmin(np.abs(x-normlim[0]))
        xnorm_upper = np.argmin(np.abs(x-normlim[1]))
        norm_curve = y_sub/np.max(y_sub[xnorm_lower:xnorm_upper])
    else:
        norm_curve = np.zeros(len(x)) * np.nan
    # Create output
    pos = np.vstack((minpos, miny)).T
    # Create recarray with fields
    dt = spec.dtype.name
    dtype_list = [('x',dt), ('y',dt), ('y_smooth',dt), ('bl',dt), ('y_blsub',dt), ('y_norm',dt)]
    out = np.recarray(len(spec), dtype=dtype_list)
    # Fill array
    out.x = spec[:,0]
    out.y = spec[:,1]
    out.bl = bl
    out.y_smooth = y_smooth
    out.y_blsub = y_sub
    out.y_norm = norm_curve
    return out

def subtract_sol(spec, sol, check=False, which='y_blsub', tol=5, label1='before', label2='irradiated'):
    '''
    Subtracts pure solvent spectrum from mixture
    spec: Spectrum with solvent peaks
    sol: pure solvent spectrum (but can be any spectrum as well)
    check: plot spectra
    which: which column to take from spectra, e.g. 'y', 'y_blsub'...
    tol: tolerance for maximum search in spectrum
    label1: if you don't want to have the default labels you can change it here
    label2: dito
    The output is a recarray with the following fields
    out.x:     x-values
    out.spec:  scaled and interpolated spectrum
    out.sol:   scaled and interpolated solvent spectrum
    out.subtr: subtracted spectrum (spec - sol)
    '''
    # Determine maximum position in solvent spectrum
    sol_ind = np.argmax(sol[which])
    sol_int = np.max(sol[which])
    sol_wn  = sol.x[sol_ind]
    # find position for other spectrum
    spec_ind = np.argmin(np.abs(spec.x-sol_wn))
    spec_int = np.max(spec[which][spec_ind-10:spec_ind+10])
    spec_ind = np.argmax(spec[which][spec_ind-10:spec_ind+10])
    # Determine lower and upper limits for interpolation
    interp_lower = np.max([np.min(sol.x), np.min(spec.x)])
    interp_upper = np.min([np.max(sol.x), np.max(spec.x)])
    # Create x values for interpolation
    x_interp = np.linspace(interp_lower, interp_upper, 1000)
    # Interpolate
    spec_interp = np.interp(x_interp, spec.x, spec[which]).squeeze() / spec_int
    sol_interp  = np.interp(x_interp, sol.x, sol[which]).squeeze() / sol_int
    # Create array
    dt = spec_interp.dtype.name
    out = np.recarray(len(x_interp), dtype=[('x',dt), ('spec',dt), ('sol',dt), ('subtr',dt)])
    out.x = x_interp
    out.spec = spec_interp
    out.sol = sol_interp
    out.subtr = spec_interp - sol_interp
    # Output plot if asked 
    if check: 
        fig, axs = plt.subplots(3, sharex=True)
        ax = axs[0]
        ax.set_title('BL corrected spectra, normalized to maximum, no smoothing')
        ax.plot(spec.x, (spec.y-spec.bl)/np.max(spec.y_blsub), label=label1)
        ax.plot(sol.x, (sol.y-sol.bl)/np.max(sol.y_blsub), label=label2)
        ax.legend()
        ax = axs[1]
        ax.plot(x_interp, spec_interp, label=label1)
        ax.set_title('Smoothed + BL corrected + interpolated)')
        ax.axhline(0, color='black', zorder=-10)
        ax.plot(x_interp, sol_interp, label=label2)
        ax.axhline(0, color='black', zorder=-10)
        ax.legend()
        ax.plot(sol_wn, 1, 'o')
        ax = axs[2]
        ax.plot(x_interp, spec_interp - sol_interp)
        ax.set_title('Irradiated-before')
        ax.axhline(0, color='black', zorder=-10)
        ax.set_xlim([np.min(x_interp), np.max(x_interp)])
        fig.show()
        return out, fig
    else:
        return out, None


def read_aoforce(fn):
    '''
    This function plots a spectrum 
    from a Turbomole aoforce file
    fn: File name
    fwhm: width of gaussian
    xlim: plot limits
    '''
    # Check if file is there
    if not os.path.isfile(fn):
        print("File %s not found! Exiting" % fn)
        return None
    # Open file
    f=open(fn)
    s=f.readlines()
    f.close()

    # extract lines with frequencies and intensities
    freq=[]
    intensities=[]
    for i in range(0,len(s)):
        if '     frequency' in s[i]:
           tmpline=s[i].split()
           del tmpline[0]
           freq.extend(tmpline)
        if 'intensity (km/mol)  ' in s[i]:
           tmpline=s[i].split()
           del tmpline[0]
           del tmpline[0]
           intensities.extend(tmpline)

    # Delete imaginary frequencies
    counter=0
    while "i" in freq[counter]:
        print("Imaginary frequency detected! "+ freq[counter])
        freq[counter]=freq[counter].replace('i','-')
        counter+=1
    # Delete rotations and translations
    freq[0:6]=[]
    intensities[0:6]=[]
    
    # Return values
    return (np.array(list(map(float,freq))), np.array(list(map(float, intensities,))))

def plot_aoforce(freq, intensities,ax=None, fwhm=20, xlim=(0,4000), scalf=0.9711):
    '''
    Plots calculated IR spectrum 
    generated from Turbomole aoforce
    ax: 
    '''
    print("Using a scaling factor of %.5f." % scalf)
    # Generate new figure if no axis is given
    if ax == None:
        fig, ax = plt.subplots(1)
        
    # Initialize spectrum
    x=np.array(range(*xlim))
    #specsum1=np.zeros(len(x))
    specsum=np.zeros(len(x))

    # Plot spectra
    for i in range(len(freq)):
        tempg=gauss(x,freq[i]*scalf,intensities[i], fwhm=fwhm)
        ax.plot(x,tempg,'k-')
        specsum += tempg
        ax.annotate("{:.0f}".format(freq[i]*scalf),xy=(freq[i]*scalf,intensities[i]),ha='center')
    ax.plot(x,specsum)

    # Adjust plot
    ax.set_ylabel('Absorbance')
    ax.set_yticks([])
    ax.set_xlabel('Wavenumber / cm$^\mathregular{-1}$')

    return (x, specsum,)

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
        dat_pos = np.argwhere(dat.iloc[:,0] == 'Time [s]')[0,0] +1
        # Get concentrations and locations
        lig_pos_ver = np.argwhere(dat.iloc[:,0] == 'Ligand Concentration:')[0,0] 
        lig_pos_hor = np.argwhere(dat.iloc[lig_pos_ver,:] == 'Ligand Concentration:').squeeze() +1
        self.concs = np.array(dat.iloc[lig_pos_ver, lig_pos_hor]).astype(np.float32) * 1E-6
        # Get ligand name
        lig_pos_ver = np.argwhere(dat.iloc[:,0] == 'Ligand:')[0,0] 
        lig_pos_hor = np.argwhere(dat.iloc[lig_pos_ver,:] == 'Ligand:').squeeze() +1
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
        # self.sort()
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
            label = "$K_D=$%s\pm%.0f$" % (kd_label, fit_fnorm_bl_err[0] / fit_fnorm_bl_opt[0]  * 100) +'%' 
            ax.semilogx(self.concs, fit_fnorm_bl, '--', zorder=-20, label=label)
            ax.legend()
            ax.set_ylabel('F$_\mathrm{norm,bl}$ / Counts' )
        else:
            ax.set_title('MST analysis')
            self.plot_colored(ax, self.concs, fnorm, self.concs)
            if (len(fit_fnorm) > 0 ):
                if fit_fnorm_err[0] / fit_fnorm_opt[0] < 1:
                    print("Fit error for fnorm larger than 100%. Will not plot it")
                    label = "$K_D$=%.0EM$\pm%.0f$" % (fit_fnorm_opt[0], fit_fnorm_err[0] / fit_fnorm_opt[0]  * 100) +'%' 
                    ax.semilogx(self.concs, fit_fnorm, '--', zorder=-20, label=label)
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
                ax.semilogx(self.concs, fit_f_init, '--', zorder=-20, label=label)
                ax.legend()
        ax.set_xlabel('Ligand concentration / M')
        ax.set_ylabel('F$_\mathrm{init}$ / Counts')
        # Plot tolerance area +- 20%
        F_mean = np.mean(f_init)
        ax.axhline(F_mean, linestyle='--', color='grey', zorder=-10)
        ax.axhspan(.8*F_mean, 1.2*F_mean, facecolor='grey', alpha=.5, zorder=-20)
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
        return None

    def fit_kd(self,concs, y, fix_pconc=False):
       # Chose fitting function
        if fix_pconc:
            func = single_site_kd(self.pconc)
            print("Will fit with fixed protein concentration of %.1e." % self.pconc)
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
                    ax.semilogx(xs[i], ys[i], 'o', alpha=alpha_out, lw=lw, color=cmap[i])
                    # if hasattr(self, 'fnorm'):
                #     axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha_out, lw=lw, color=cmap[i])
            else:
                # Check if is number or array
                if isinstance(ys[i], np.ndarray):
                    ax.plot(xs, ys[i],  alpha=alpha, lw=lw, color=cmap[i], linestyle=linestyle)
                else:
                    ax.semilogx(xs[i], ys[i], 'o', alpha=alpha, lw=lw, color=cmap[i])

                # if hasattr(self, 'fnorm'):
                #     axs[1].semilogx(self.concs[i], self.fnorm[i], 'o', alpha=alpha, lw=lw, color=cmap[i])
        return None


    def add_colorbar(self, ax, concs):
        # setup the colorbar
        normalize = mcolors.LogNorm(vmin=np.min(concs), vmax=np.max(concs)) # Or Normalize 
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=plt.cm.jet) 
        scalarmappaple.set_array(concs)
        cbar = plt.colorbar(scalarmappaple, ax=ax)
        cbar.set_label('Lig. conc. / M', rotation=270) 
        cbar.ax.get_yaxis().labelpad = 15
        return None
    
    def plot(self, smooth=False, smooth_window=51):
        if hasattr(self, 'fnorm'):
            fig, axs = plt.subplots(1,2, figsize=(10,5))
            ## Plot outliers in gray
            #for out in self.outliers:
            #    ax.semilogx(self.concs[out], self.fnorm[out], 'o', color='gray')
            ax = axs[1]
            if hasattr(self, 'fit_opt'):
                # Upper and lower limits for model (based on KD error)
                kd_label = kd_unit(self.fit_opt[0])
                hp_fit, = ax.semilogx(self.concs_dense, self.fit, label='$K_D=$%s$\pm$%.0f%%' % (kd_label, self.fit_err[0]/self.fit_opt[0]*100))
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
        return None

class CD_data():
    '''
    This is a class to read in Chirascan CD data 
    that has been exported as csv
    '''
    def __init__(self, fn=''):
        '''
        Initialize ms data
        fn: File name
        '''
        self.fn = fn
        #self.path = path
        # Load data
        self.read_CD()
        self.average()
        return None

    def read_CD(self):
        '''
        Reads comma separated csv from Chirascan export
        
        Required: 
        fn: File name
        
        Returns: spectrum as ndarray
        '''
        # List with entries
        spectrum = []
        with open(self.fn, 'r') as f:
            for line in f:
                if line[0].isdigit():
                    spectrum.append(line.strip().split(','))
                    # Convert to array
        cd = np.array(spectrum, dtype=float)
        self.x = cd[:,0]
        self.y = cd[:,1:]
        return cd

    def average(self):
        '''
        Averages y values
        '''
        self.av_y = np.average(self.y, axis=1)
        return None

    def savetxt(self):
        # Generate first part of header
        header = '%18s' % 'Wavelength/nm'
        for i in range(self.y.shape[1]):
            header += '%21s' % ('Repeat %i' % (i+1))
        if hasattr(self, 'av_y'):
            dat = np.hstack((self.x.reshape(-1,1), self.y, self.av_y.reshape(-1,1)))
            header += '%21s' % 'Average'
        else:
            dat = np.hstack((self.x.reshape(-1,1), self.y))
        np.savetxt(self.fn.replace('.csv', '') + '.txt', dat, fmt='%20.10f', header=header)
        print("Text file saved as %s" % (self.fn.replace('.csv', '') + '.txt'))
    
