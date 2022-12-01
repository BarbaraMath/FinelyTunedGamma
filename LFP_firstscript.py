import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import scipy
import mne

from mne.time_frequency import tfr_morlet
from scipy.fft import fft, fftfreq

#os.getcwd() for finding current working directory
os.chdir('')

print(raw) #here you can see the n samples and the time
print(raw.info) #ch_names, sfreq, nchan.
n_time_samps = raw.n_times #nsamples
time_secs = raw.times #timepoints
raw.info.keys()
ch_names = raw.ch_names
n_chan = len(ch_names) 

raw_data = raw.get_data()
print(raw_data.shape)
mydat = raw.get_data(picks='LFP_R_13_STN')
print(mydat)
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

plot_times = raw.times #time
xticks = np.linspace(0, plot_times[-1], 5) #make 5 x-axis ticks, dividing the seconds by 5

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
            
            axes[ax_c].plot(plot_times, raw_data[i, :])
            axes[ax_c].set_title(name, fontsize=16, color='r')
            axes[ax_c].set_xticks(xticks)
            axes[ax_c].set_xticklabels(np.around(xticks / 60, 1))
                
        ax_c += 1
plt.show()
        
# chNamesArr == ch_to_plot


#Fast Fourier Transformation

N = 250 * (n_time_samps/250)

yf = fft(mydat)
xf = fftfreq(N, 1 / 250)

plt.plot(xf, np.abs(yf))