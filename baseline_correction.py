####### BASELINE CORRECTION #######
import mne
import numpy as np
import pandas as pd
from mne.baseline import rescale
from matplotlib import pyplot as plt

def baseline_corr(data, t, baseline, raw = 0, stim_ch = 0):
    """
    baseline_corr performs a zlogratio baseline correction of the input recording

    input:
    - data: np.array of shape 1 x frequencies x time
    - t: last dimension of the data (time)
    - baseline: tuple with two numbers i.e. the baseline e.g. (1,100)
    - raw: optional argument, when we want to ploit stimulation on top of recording
    - stim_ch: optional argument, to specify stimulation channel e.g. 4(LSTN) or 5(RSTN)

    """

    bs_data = mne.baseline.rescale(data = data, times = t, baseline = baseline, mode = 'zscore')

    #Plot Spectrograms of both STNs
    fig, axes = plt.subplots(1,2, figsize = (18,6))
    fig.suptitle('Z-Scored Spectrogram')


    ax_c = 0
    stim = 4

    for kj in np.array([0,1]):
        ax2 = axes[kj].twinx() #make right axis linked to the left one
        ''''
        if kj == 1:
            stim_data = (raw.get_data(picks = stim)[0,:]) #define stim channel
        elif kj == 0:
            stim_data = (raw.get_data(picks = stim)[0,:])
'''
        #Plot LFP data
        axes[ax_c].pcolormesh(bs_data[ax_c,:], cmap = 'viridis', vmin = -1, vmax = 3)
        axes[ax_c].set_ylim(bottom = 5,top = 100)
        axes[ax_c].set_xlim(0,raw.n_times/250)

        axes[ax_c].set_ylim(5, 100)
        axes[ax_c].set_ylabel('Frequency [Hz]')
        axes[ax_c].set_xlabel('Time [sec]')
        axes[ax_c].set_title(raw.ch_names[kj])

        ax_c += 1
        stim += 1

    if raw != 0:
        stim_data = (raw.get_data(picks = stim_ch)[0,:])
        ax2.plot(raw.times, stim_data,'w',linewidth = 1.5)
        ax2.set_yticks(np.arange(0,4.5,0.5))
        ax2.set_ylabel('Stimulation Amplitude [mA]')
    
    
        

        #cbar = fig.colorbar(cf, ax = ax, location = 'bottom', pad = 0.1, shrink = 0.5)
        #cbar.set_label('zlogratio')
    
    plt.show(block = False)

    #np.save(new_bcfname + '.npy', bs_data)
    #plt.savefig(new_bcfname + '.pdf')
    return bs_data