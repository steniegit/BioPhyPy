# -*- coding: utf-8 -*-
"""
This module contains helpful functions used by the libspec module
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/snieblin/work/libspec')
sys.path.insert(0,'/home/snieblin/work/')
import libspec
import scipy.signal as ssi
import opusFC as ofc
from scipy.optimize import curve_fit
import ipdb

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
    return height*np.exp(-np.subtract(x,center)**2/(2*(c**2))\
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

def quick_test(fn, peak_lim=[2150, 2195], plot=True):
    '''
    Just a quick way to plot data and fits
    directly from the 
    '''
    # Load data
    spec, spec_opt, deriv_0 = process_bruker(fn, peak_lim=peak_lim, gauss_pos = 'deriv', guess=[2161, 0.2, 5, 2168, 0.2, 5], fit_tol=10)
    if not plot:
        return None, None, spec_opt
    # Create figure
    fig, axs = plt.subplots(3,1, sharex=True)
    fig.canvas.set_window_title(fn)
    # Plot raw, smoothed and BL
    ax = axs[0]
    ax.plot(spec.x, spec.raw)
    ax.plot(spec.x, spec.smooth)
    ax.plot(spec.x, spec.bl)
    # Plot deriv
    ax = axs[1]
    ax.plot(spec.x, spec.deriv)
    ax.axhline(color='grey', linestyle='--')
    for pos in deriv_0:
        ax.axvline(pos, color='grey', linestyle='--')
    # Plot labels
    for pos in deriv_0:
        ax.text(pos, ax.get_ylim()[1], '%.0f' % pos, ha='center')
    # Plot fits
    ax = axs[2]
    h54, = ax.plot(spec.x, spec.blcorr, label='spec')
    ax.plot(spec.x, spec.fit1, ':', color=h54.get_color())
    ax.plot(spec.x, spec.fit2, ':', color=h54.get_color())
    ax.plot(spec.x, spec.fit)
    ax.plot(spec.x, spec.fit1 + spec.fit2)
    # Plot labels
    for i in range(len(spec_opt) // 3):
        ax.text(spec_opt[i*3], spec_opt[i*3+1], '%.0f(%.0f)' % (spec_opt[i*3], spec_opt[i*3+2]), ha='center', va='center')
    return fig, axs, spec_opt
