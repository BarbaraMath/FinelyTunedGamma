import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import scipy
import mne
import sys

from mne.time_frequency import tfr_morlet
from mne.baseline import rescale
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, hann, butter, filtfilt
from scipy import signal, interpolate
from scipy.interpolate import make_interp_spline, BSpline
from datetime import datetime

os.getcwd() #for finding current working directory
sys.path.append('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\RampUpThres')

sys.modules[__name__].__dict__.clear()

raw = mne.io.read_raw_fieldtrip('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\RampUpThres\\sub-021_ses-2022082612233578_run-BrainSense20220826125900.mat', info = None)
#for macbook at home
#raw = mne.io.read_raw_fieldtrip('Macintosh HD\\Users\\barbaramathiopoulou\\OneDrive - Charité - Universitätsmedizin Berlin\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub025\\\
#    sub-20210630PSTN_ses-2022062806215184_run-BrainSense20220628070600.mat', info = None)

#sys.path.insert(0,'Users/barbaramathiopoulou/OneDrive - Charité - Universitätsmedizin Berlin/OneDrive - Charité - Universitätsmedizin Berlin/FTG_PROJECT/Sub025')

#raw = mne.io.read_raw_fieldtrip('sub-20210630PSTN_ses-2022062806215184_run-BrainSense20220628070600.mat', info = None)

print(raw) #here you can see the n samples and the time
print(raw.info) #ch_names, sfreq, nchan.
n_time_samps = raw.n_times #nsamples
time_secs = raw.times #timepoints
raw.info.keys()
ch_names = raw.ch_names
n_chan = len(ch_names) 

channel_names = ['LFP_L_13_STN', 'LFP_R_13_STN']
two_meg_chans = raw[channel_names, 1:n_time_samps]
y_offset = np.array([200, 0])  # just enough to separate the channel traces
x = two_meg_chans[1]
y = two_meg_chans[0].T + y_offset
lines = plt.plot(x, y)
plt.legend(lines, channel_names)
plt.show()



raw_data = raw.get_data()
print(raw_data.shape)
RSTN_dat = raw.get_data(picks='LFP_Stn_R_02')
LSTN_dat = raw.get_data(picks = 'LFP_L_13_STN')
#fig, ax = plt.subplots(figsize=[15, 5])
#ax.plot(mydat)
#plt.show()

######################## WORKING WITH EVENTS ########################
d, t = raw[raw.ch_names.index('STIM_R_125Hz_60us'), :]
plt.plot(d[0,:])
plt.show(block = False)

#plt.plot(raw.times, raw.get_data(picks = 5)[0])
#plt.show(block = False)


my_annot = mne.Annotations(onset=[1, 60, 90, 110, 140, 161, 182],  # in seconds
                           duration=[5,5,5,5,5,5,5],  # in seconds, too
                           description=['NoStim','STIM_1','STIM_2','STIM_3','STIM_4','STIM_5','ThresStim'])
raw.set_annotations(my_annot)

raw.annotations.save('saved-annotations.csv',overwrite=True)

events_from_annot, event_dict = mne.events_from_annotations(raw)

fig = mne.viz.plot_events(events_from_annot, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp, event_id=event_dict)

epochs = mne.Epochs(raw, events_from_annot, event_id=event_dict,
                    preload=True)


epochs['ThresStim'].plot_psd(picks = 'LFP_R_13_STN', average=False)

epo_spectrum = epochs.compute_psd(method = 'multitaper', picks = 'LFP_R_13_STN')

raw.compute_psd(picks = 'LFP_R_13_STN', method = 'welch').plot(picks = 'LFP_R_13_STN')

psds, freqs = epo_spectrum.get_data(return_freqs=True, picks = 'LFP_R_13_STN')

plt.plot(freqs[40:70], psds[0,0,40:70], label = 'NoStim')
plt.plot(freqs[40:70], psds[6,0,40:70], label = 'ThresStim')
plt.legend(['NoStim','ThresStim'])
plt.show()

for jk in np.arange(0,7):
    #.plot(freqs[40:70], psds[jk,0,40:70])
    epochs[jk].compute_psd(picks = 'LFP_R_13_STN').plot(picks = 'LFP_R_13_STN')

plt.show()


psd_fig = raw.plot_psd(picks='LFP_R_13_STN', fmin=2, fmax=40, n_fft=int(3 * raw.info['sfreq']),
                           reject_by_annotation=True)

epochs['STIM_1'].compute_psd(picks = 'LFP_R_13_STN', method = 'welch').plot(picks = 'LFP_R_13_STN')

fft_res = fft(RSTN_dat)

epochs['NoStim'].plot_psd(picks = 'LFP_R_13_STN', average = True)