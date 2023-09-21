import mne
import matplotlib.pyplot as plt
import numpy as np
import dat_preproc
from scipy.signal import hilbert 
import pandas as pd
import os

def anal_signal_transform(raw, path, subID, SIDE, peakMed, peakStim):
    x = raw.get_data() 

    x1 = x[SIDE,:]

    if SIDE == 0:
        x2 = x[4, :] 
    elif SIDE == 1:
        x2 = x[5, :]

    dat_ngam = dat_preproc.low_highpass_filter(x1, peakMed-2, peakMed+2) 
    dat_subh = dat_preproc.low_highpass_filter(x1, peakStim-2, peakStim+2) 
    dat_inb = dat_preproc.low_highpass_filter(x1, peakStim+3, peakMed-3) 

    datall = [dat_ngam, dat_subh, dat_inb] 
    labels = ['Peak'+str(peakMed)+'Hz','Peak'+str(peakStim)+'Hz', str(peakStim+3) + '-' + str(peakMed-3)+'Hz']

    tstamps_sec = (1 / raw.info['sfreq']) * np.arange(raw.n_times)

    ### HILBERT TRANSFORMATION ###
    all_signal_np = np.empty(shape = (len(datall), x1.shape[0]))
    all_signal_np[:] = np.nan

    #wintosmooth = 500
    for idx, dat in enumerate(datall):
        hiltr = hilbert(dat)
        amplitude_envelope = np.abs(hiltr)
        #zscore_sign = stats.zscore(np.squeeze(amplitude_envelope))
        #sm_signal = window_rms(zscore_sign, wintosmooth)
        
        all_signal_np[idx,:] = amplitude_envelope

    ### PLOT IT ###
    fig, ax1 = plt.subplots(figsize = (18,6))
    ax2 = ax1.twinx()
    for idx, dat in enumerate(all_signal_np):
        ax1.plot(tstamps_sec, all_signal_np[idx,:], label = labels[idx], lw = 2)
    
    ax2.plot(tstamps_sec, x2[:], label = 'Stimulation', color = 'grey', ls='--', lw=3, alpha = 0.4)

    ax1.set_ylabel('Analytic Signal')
    ax2.set_ylabel('Stimulation Amplitude [mA]')

    plt.title(str(subID))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    fig.legend(lines, labels, loc='upper right')

    plt.show()

    #### MAKE THEM ONE ARRAY ####
    all_signal_np = np.transpose(np.squeeze(np.array([[all_signal_np[0]], [all_signal_np[1]],[all_signal_np[2]],[x2]])))
    #sm_analSignal = np.transpose(np.squeeze(np.array([[sm_signal_np[0]],[sm_stim1]])))
    all_signal_np_df = pd.DataFrame(all_signal_np, 
        columns = ['Spontan', 'StimOn', 'InBetween', 'StimVec'],
        )

    #### TRANSFORM THE SIGNAL TO MNE OBJECT ####
    ch_names = list(all_signal_np_df.columns)
    sfreq = 250
    info = mne.create_info(ch_names, sfreq)
    raw_anal = mne.io.RawArray(all_signal_np_df.T, info)

    #Add time channel
    new_ch_name = 'TimeinSec'
    new_data_2d = tstamps_sec.reshape(1,-1)
    new_info = mne.create_info(['Time_Sec'], raw_anal.info['sfreq'], ch_types=['misc'])

    # Create a RawArray object for the new channel
    new_raw_array = mne.io.RawArray(new_data_2d, new_info)

    # Add the new channel to the original Raw object
    raw_anal.add_channels([new_raw_array])

    plt.savefig(os.path.join(path, str(subID)+'anal_Signal'),dpi = 300)
    fif_name = str(subID)+'anal_SignalOnly.fif'
    
    raw_anal.save(os.path.join(path, fif_name))
    return raw_anal





    




def anal_transitions_2tp(anal_epochs, anal_fif, subID, dur):


    filtered_df = anal_epochs[anal_epochs['Percept_ID'] == subID]
    subh_value = filtered_df['Subh_On'].values[0]
    presubh_value = filtered_df['preSub_On'].values[0]

    # Extract the data starting from 'preSub_On' for 10 seconds
    preSub_On_start = presubh_value
    preSub_On_end = preSub_On_start + dur  # Duration of 10 seconds
    preSub_On_data = anal_fif.copy().crop(tmin=preSub_On_start, tmax=preSub_On_end)

    # Extract the data starting from 'Subh_On' for 10 seconds
    subh_On_start = subh_value
    subh_On_end = subh_On_start + dur  # Duration of 10 seconds
    subh_On_data = anal_fif.copy().crop(tmin=subh_On_start, tmax=subh_On_end)

    # Concatenate the two data segments
    combined_data = mne.concatenate_raws([preSub_On_data, subh_On_data])

    # Extract the channel names and data arrays
    channels = combined_data.ch_names
    cropped_anal_2tp = combined_data.get_data()

    # Create a time vector
    sfreq = combined_data.info['sfreq']
    n_samples = combined_data.n_times
    times = np.arange(n_samples) / sfreq

    # Plot the data using matplotlib
    plt.figure(figsize=(12, 6))
    for i, channel in enumerate(channels):
        plt.plot(times, cropped_anal_2tp[i], label=channel)

    plt.axvline(x=dur, color='grey', linestyle='--', lw = 3, alpha = 0.4, label = 'Subharmonic On')
    plt.xticks(np.arange(0,45,5), labels=np.arange(-20,25,5))
    plt.legend(loc='upper right')

    plt.xlabel('Time [sec]')
    plt.ylabel('Z-Scored Smoothed Analytic Signal')
    plt.title(str(subID) + '-Two Time Points')

    return cropped_anal_2tp


def anal_transitions_1tp(anal_epochs, anal_fif, subID, dur):


    filtered_df = anal_epochs[anal_epochs['Percept_ID'] == subID]
    subh_value = filtered_df['Subh_On'].values[0]

    # Extract the data starting from 'preSub_On' for 10 seconds
    preSub_On_start = subh_value - dur
    preSub_On_end = subh_value + dur  # Duration of 10 seconds
    preSub_On_data = anal_fif.copy().crop(tmin=preSub_On_start, tmax=preSub_On_end)

    # Extract the channel names and data arrays
    channels = preSub_On_data.ch_names
    cropped_anal_1tp = preSub_On_data.get_data()

    # Create a time vector
    sfreq = preSub_On_data.info['sfreq']
    n_samples = preSub_On_data.n_times
    times = np.arange(n_samples) / sfreq

    # Plot the data using matplotlib
    plt.figure(figsize=(12, 6))
    for i, channel in enumerate(channels):
        plt.plot(times, cropped_anal_1tp[i], label=channel)

    plt.axvline(x=dur, color='grey', linestyle='--', lw = 3, alpha = 0.4, label = 'Subharmonic On')
    plt.xticks(np.arange(0,45,5), labels=np.arange(-20,25,5))
    plt.legend(loc='upper right')

    plt.xlabel('Time [sec]')
    plt.ylabel('Z-Scored Smoothed Analytic Signal')
    plt.title(str(subID) + '-One Time Point')
    plt.ylim(0,3)

    return cropped_anal_1tp