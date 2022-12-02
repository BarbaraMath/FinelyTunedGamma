import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import mne

from mne.time_frequency import tfr_morlet
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from scipy import signal

os.getcwd() #for finding current working directory
os.chdir('')

raw = mne.io.read_raw_fieldtrip('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub025\\sub-20210630PSTN\
_ses-2022062806215184_run-BrainSense20220628070600.mat', info = None)

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
RSTN_dat = raw.get_data(picks='LFP_R_13_STN')
LSTN_dat = raw.get_data(picks = 'LFP_L_13_STN')
#fig, ax = plt.subplots(figsize=[15, 5])
#ax.plot(mydat)
#plt.show()

#Make some nice plots of the data

chNamesList = raw.info['ch_names']
chNamesArr = np.array(chNamesList)

#channels to plot:
chs_to_plot = [ 
    'LFP_R_13_PEAK76Hz_THR20-30_AVG1000ms',
    'LFP_R_13_STN',
    'STIM_R_125Hz_60us'
]

xticks = np.linspace(0, time_secs[-1], 5) #make 5 x-axis ticks, dividing the seconds by 5

fig, axes = plt.subplots(
    1, len(chs_to_plot), figsize=(18, 6)
) #define n of subplots and size

# axes = axes.flatten()
ax_c = 0

for i, name in enumerate(chNamesList):
    
    if name in chs_to_plot:

        if name[-3:] == 'STN':
            axes[ax_c].psd(raw_data[i, :])
            axes[ax_c].set_title(
                f'PSD {name}',
                fontsize=16, color='r'
            )

        else:
            
            axes[ax_c].plot(time_secs, raw_data[i, :])
            axes[ax_c].set_title(name, fontsize=16, color='r')
            axes[ax_c].set_xticks(xticks)
            axes[ax_c].set_xticklabels(np.around(xticks / 60, 1))
                
        ax_c += 1

        
# chNamesArr == ch_to_plot

######################## WORKING WITH EVENTS ########################

fig = raw.plot(start=1, duration=120)
fig.fake_keypress('a')
NoStim = mne.Annotations(onset=1, duration=11, description='NoStim')
ThresStim = mne.Annotations(onset=161, duration=11, description='ThresStim')

my_annot = mne.Annotations(onset=[1, 161],  # in seconds
                           duration=[11, 11],  # in seconds, too
                           description=['NoStim', 'ThresStim'])
raw.set_annotations(my_annot)

raw.annotations.save('saved-annotations.csv',overwrite=True)

events_from_annot, event_dict = mne.events_from_annotations(raw)

fig = mne.viz.plot_events(events_from_annot, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp, event_id=event_dict)
fig.subplots_adjust(right=0.7)  # make room for legend

raw.plot(events=events_from_annot, start=0, duration=180, color='gray',
         event_color={1: 'r', 2: 'g'})