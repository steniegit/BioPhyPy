'''
Mass spec data
'''

from .helpers import *
import pandas as pd
import numpy as np

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
        self.peaks_kda = self.spec[peaks,0]
        self.peak_info = info
        print("Found %i peaks" % len(self.peaks_kda))
        print(self.peaks_kda)
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
