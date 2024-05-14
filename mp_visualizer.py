import streamlit as st
import numpy as np
from typing import Any
import h5py
# import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
import time
import sys, glob, os
sys.path.append("../")
from libspec import MP_data
from tempfile import NamedTemporaryFile
import ipdb

# Initial parameters
movie = False
events = False

# Label
st.sidebar.header("1. Upload file(s)")
# st.sidebar.subheader("Movie file")
# Loader
fn_movie = st.sidebar.file_uploader("Movie file", type=['mp','mpr'])
# st.sidebar.subheader("Events file (optional)")
fn_events = st.sidebar.file_uploader("Events file (optional)", type=['h5','csv'])
st.sidebar.header("Processing parameters")

st.sidebar.header("Play movie")
# Interactive Streamlit elements, like these sliders, return their value.
# This gives you an extremely simple interaction model.
slider = st.sidebar.empty() #slider("Level of detail", 2, 20, 10, 1)
most_counts = st.sidebar.button("Jump to frame with most counts")
if most_counts:
    print("Test")

# Non-interactive elements return a placeholder to their location
# in the app. Here we're storing progress_bar to update it later.
progress_bar = st.sidebar.progress(0)

# Create columns
col1, = st.columns(1)

# These two elements will be filled in later, so we create a placeholder
# for them using st.empty()
frame_text = st.sidebar.empty()
image = col1.empty()
image_hist = col1.empty()
table = col1.empty()
# Histogram
hist = col1.empty()

# Parameters
threshold_big=1000
ratiometric_size=10
frame_range=2

# Variable to check if data was loaded
data_loaded = False

# # Slider for animation
# initial_value = 25

# slider_ph = st.empty()
# info_ph = st.empty()

# value = slider_ph.slider("slider", 0, 50, initial_value, 1, key="initial")
# info_ph.info(value)

# if st.button('animate'):
#     for x in range(50 - value):
#         time.sleep(.5)

#         value = slider_ph.slider("slider", 0, 50, value + 1, 1, key="animated")
#         info_ph.info(value)

@st.cache_data
def read_data(fn_movie, fn_events):
    if fn_movie and fn_events:
        # Load file
        video = h5py.File(fn_movie)
        video = np.array(video['movie']['frame']).astype('int16')
        # Change sign
        video = video[:]*-1
        # Load events
        # Load fitted events to obtain more parameters
        data = h5py.File(fn_events, 'r')
        if 'masses_kDa' in data.keys():
            masses_kDa = np.array(data['masses_kDa']).squeeze()
        elif 'calibrated_values' in data.keys():
            masses_kDa = np.array(data['calibrated_values']).squeeze()
        else:
            print("Could neither find calibrated_values nor masses_kDa! Will only load contrasts.")
            masses_kDa = None
        # Create dataframe
        events = pd.DataFrame({'frame_ind': data['frame_indices'],
                               'contrasts': data['contrasts'],
                               'kDa': masses_kDa, # used to be  test['masses_kDa']
                               'x_coords': data['x_coords'],
                               'y_coords': data['y_coords']})
        data_loaded = True
    else:
        data_loaded = False
        events = None
        dra = None
    return video, events, data_loaded

@st.cache_data
def create_dra(frame_num, video, events, frame_range=0):
    # Obtain ratiometric contrast
    dra = np.mean(video[frame_num+1:frame_num+1+ratiometric_size//2], axis=0) / np.mean(video[frame_num-ratiometric_size//2:frame_num], axis=0) - 1
    # Center
    dra -= np.median(dra)
    # Obtain events
    events_frame = events[events['frame_ind'].between(frame_num-frame_range, frame_num+frame_range)]    
    return dra, events_frame

@st.cache_data
def create_histogram(events, bin_width=4):
    '''
    Creates histogram of masses
    '''
    # Get min and maximum value
    window = [np.floor(np.min(events['kDa'])), np.ceil(np.max(events['kDa']))]
    # Determine number of bins based on bin_width
    nbins = int((window[1] - window[0]) // bin_width)
    # Create histogram
    hist_counts, hist_bins = np.histogram(events['kDa'], range=window, bins=nbins)
    return hist_counts, hist_bins, bin_width

@st.cache_data
def create_image(dra, events_frame, frame_num):
    # Determine limits for plot
    im_min = np.min(dra)
    im_max = np.max(dra)
    # Take arithmetic middle
    thresh = np.sqrt(np.abs(im_min*im_max))
    # Plot one frame in image
    #fig = px.imshow(dra)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(dra, norm=colors.SymLogNorm(linthresh=0.02, base=10, linscale=1)) #, vmin=-thresh, vmax=thresh))
    # Draw circles
    for event in events_frame.iterrows():
        event = event[1]
        # Plotly
        #fig.add_shape(type="circle", x0=int(event['x_coords']), y0=int(event['y_coords']))
        # Matplotlib
        if event['frame_ind'] == frame_num:
            alpha = 1
        elif np.abs(event['frame_ind'] - frame_num) == 1:
            alpha = 0.5
        elif np.abs(event['frame_ind'] - frame_num) == 2:
            alpha = 0.25
        else:
            alpha= 0.1
        circ = Circle((int(event['x_coords']), int(event['y_coords'])), 5, fc='None', ec='red', lw=2, alpha=alpha)
        ax.add_patch(circ)
        ax.text(int(event['x_coords']), int(event['y_coords'])+5, int(event['kDa']), ha='center', va='top', fontsize=6)
    return fig

@st.cache_data
def plot_histogram(hist_counts, hist_bins, bin_width, events_frame, frame_num):
    # Plot one frame in image
    #fig = px.imshow(dra)
    # Calculate centers
    hist_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    # Initialize figure
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(hist_centers, hist_counts, width=bin_width)
    return fig

# @st.cache_data
# def load_data(fn_events, fn_movie='', frame_range=2):
#     # Load data
#     with NamedTemporaryFile(dir='.', suffix='.mpr') as f:
#         f.write(fn_events.getbuffer())
#         with NamedTemporaryFile(dir='.', suffix='.h5') as f2:
#             f2.write(fn_movie.getbuffer())
#             dataset = load_data(fn_events=f.name, fn_movie=f2.name)
#     return dataset

# print("Hallo")
# ipdb.set_trace()


# dataset = load_data(fn_events, fn_movie)
# fig, ax = plt.subplots(1)
# dataset.plot_histo(ax=ax)
# image.pyplot(fig)

# Read data
video, events, data_loaded = read_data(fn_movie, fn_events)
print(video)
hist_counts, hist_bins, bin_width = create_histogram(events)

# Extract one frame
if data_loaded:
    frame_num = slider.slider("Frame number", 1, video.shape[0]+1, 1, 1)
    # Obtain dra and frame_events
    dra, events_frame = create_dra(frame_num, video, events, frame_range=frame_range)
    # Delete previous image
    image.empty()
    # Create image
    fig = create_image(dra, events_frame, frame_num)
    # Create histogram
    hist_counts, hist_bins, bin_width = create_histogram(events)
    # Create histo plot
    fig_hist = plot_histogram(hist_counts, hist_bins, bin_width, events_frame, frame_num)
    # Plot it 
    # image.plotly_chart(fig)
    image.pyplot(fig)
    image_hist.pyplot(fig_hist)
    # And show events
    table.dataframe(events_frame)

# m, n, s = 960, 640, 400
# x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
# y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))

# for frame_num, a in enumerate(np.linspace(0.0, 4 * np.pi, 100)):
#     # Here were setting value for these two elements.
#     progress_bar.progress(frame_num)
#     frame_text.text("Frame %i/100" % (frame_num + 1))

#     # Performing some fractal wizardry.
#     c = separation * np.exp(1j * a)
#     Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
#     C = np.full((n, m), c)
#     M: Any = np.full((n, m), True, dtype=bool)
#     N = np.zeros((n, m))

#     for i in range(iterations):
#         Z[M] = Z[M] * Z[M] + C[M]
#         M[np.abs(Z) > 2] = False
#         N[M] = i

#     # Update the image placeholder by calling the image() function on it.
#     image.image(1.0 - (N / N.max()), use_column_width=True)

# We clear elements by calling empty on them.
progress_bar.empty()
frame_text.empty()
