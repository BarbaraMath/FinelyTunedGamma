import mne
import matplotlib.pyplot as plt
import numpy as np
import dat_preproc
from scipy.signal import hilbert 
import pandas as pd
import os
from importlib import reload
reload(dat_preproc)

def anal_signal_transform(raw, path, subID, SIDE, peakBeta):
    
    #def anal_signal_transform(raw, path, subID, SIDE, peakMed, peakStim):
    x = raw.get_data() 

    x1 = x[SIDE,:]

    if SIDE == 0:
        x2 = x[4, :] 
    elif SIDE == 1:
        x2 = x[5, :]

    #dat_ngam = dat_preproc.low_highpass_filter(x1, peakMed-2, peakMed+2) 
    #dat_subh = dat_preproc.low_highpass_filter(x1, peakStim-2, peakStim+2) 
    #dat_inb = dat_preproc.low_highpass_filter(x1, peakStim+3, peakMed-3) 
    
    dat_LowBeta = dat_preproc.low_highpass_filter(x1, 13, 20) 
    dat_Highbeta = dat_preproc.low_highpass_filter(x1, 20, 35) 
    dat_Peakbeta = dat_preproc.low_highpass_filter(x1, peakBeta-2, peakBeta+2) 
    dat_Beta = dat_preproc.low_highpass_filter(x1, 13, 35) 

    #datall = [dat_ngam, dat_subh, dat_inb] 
    #labels = ['Peak'+str(peakMed)+'Hz','Peak'+str(peakStim)+'Hz', str(peakStim+3) + '-' + str(peakMed-3)+'Hz']
    
    datall = [dat_LowBeta, dat_Highbeta, dat_Peakbeta, dat_Beta]
    labels = ['LowBeta','HighBeta','BetaPeak','Beta']
    
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
    '''for idx, dat in enumerate(all_signal_np):
        ax1.plot(tstamps_sec, all_signal_np[idx,:], label = labels[idx], lw = 2)'''
    
    ax1.plot(tstamps_sec, all_signal_np[2,:], label = labels[2], lw = 2)
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

    min_val = np.nanmin(x2)
    max_val = np.nanmax(x2)
    stim_norm = np.where(np.isnan(x2), np.nan, ((x2 - min_val) / (max_val - min_val)) * 100)

    
    #### MAKE THEM ONE ARRAY ####
    all_signal_np = np.transpose(np.squeeze(np.array([[all_signal_np[0]], [all_signal_np[1]],[all_signal_np[2]],[all_signal_np[3]], [x2], [stim_norm]])))
    #sm_analSignal = np.transpose(np.squeeze(np.array([[sm_signal_np[0]],[sm_stim1]])))
    all_signal_np_df = pd.DataFrame(all_signal_np, 
        #columns = ['Spontan', 'StimOn', 'InBetween', 'StimVec'],
        columns = ['LowBeta', 'HighBeta', 'PeakBeta', 'Beta', 'StimVec', 'StimNormalized']
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
    fif_name = str(subID)+'BetaAnal_SignalOnly.fif'
    
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

    return 

def entrainment_latency(path, anal_unsm, filename, timeon, timeoff, amp, save):
    subID = filename[0:6]

    #Get the channels:
    tstamps_sec = anal_unsm.get_data(picks ='Time_Sec' )[0]
    subharm = anal_unsm.get_data(picks = 'StimOn')[0]
    stim = anal_unsm.get_data(picks = 'StimVec')[0]

    #tstamps_sec = tstamps_sec[220 * 250:]
    #subharm = subharm[220 * 250:]
    #stim = stim[220 * 250:]

    #Zscore to Rest & save it
    first_samples = subharm[timeon:timeoff]
    mean = np.mean(first_samples)
    std_dev = np.std(first_samples)
    z_scores = (subharm - mean) / std_dev
    threshold = np.max(z_scores)*(1.8/3)

    if save == 1:
        np.save(os.path.join(path, f'{subID}_ZscoredRest_Entrainment.npy'), z_scores)

    #CheckPoint
    plt.plot(tstamps_sec, z_scores)
    plt.plot(tstamps_sec, stim)
    plt.xlabel('Time [sec]')
    plt.ylabel('Zscored Analytic Signal')
    plt.show()

    exceeds_threshold_index = np.where((z_scores > threshold) & (stim > amp))[0]
    stim_SubhOn = stim[exceeds_threshold_index[0]]
    print(f'Threshold being used is {np.round(threshold, decimals = 2)}')
    print(f'Amplitude where Entrainment occurs is: {stim_SubhOn}mA')

    #Plot Filtered Data
    mask = stim >= stim_SubhOn
    filtered_data = np.where(mask, subharm, np.nan)
    stim_filtered = np.where(mask, stim, np.nan)

    filtered_subh = filtered_data[~np.isnan(filtered_data)]
    filtered_stim = stim_filtered[~np.isnan(stim_filtered)]

    idx = np.where(~np.isnan(filtered_data))[0][0]


    adjust_thres = exceeds_threshold_index[0] - idx
    adjust_thres1 = (adjust_thres/250)

    print(f'Entrainment occurs after {adjust_thres1} Sec at {stim_SubhOn}mA')

    ####### PLOT 1 ###############################################################
    xax = np.arange(len(filtered_subh))
    time = (xax / 250)

    fig, ax1 = plt.subplots(figsize = (10,6))
    ax1.plot(time,filtered_subh, color='blue', alpha = 0.5, label = 'Entrainment Envelope')
    ax2 = ax1.twinx()
    ax2.plot(time,filtered_stim, color='black', label = 'Stimulation')
    legend_added = False
    # Plot vertical lines at regular intervals every 2 samples for the first 5 seconds
    for i in range(0, 100, 10):
        # Add the label 'xxx' only once
        if not legend_added:
            plt.axvline(x=i, color='gray', alpha=0.2, label='1250 Pulses')
            legend_added = True
        else:
            plt.axvline(x=i, color='gray', alpha=0.2)

    ax1.axvline(x=adjust_thres1, color='r', linestyle='--', label= f'{np.round(threshold, decimals = 2)} STD')
    ax2.set_ylabel('Stimulation Amplitude [mA]')
    ax1.set_ylabel('Envelope Z-Scored to Rest M1S0')
    plt.xlim(0,15)
    ax1.set_xlabel('Time [sec]')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    fig.legend(lines, labels)  
    plt.title(f'{subID}Latency_Long') 
    plt.show()

    if save == 1:
        plt.savefig(os.path.join(path,
                                f'{subID}Latency_Long'
                                ), dpi = 200)

    ####### PLOT 2 ###############################################################
    fig, ax1 = plt.subplots(figsize = (10,6))
    ax1.plot(time,filtered_subh, color='blue', alpha = 0.5, label = 'Entrainment Envelope')
    ax2 = ax1.twinx()
    ax2.plot(time,filtered_stim, color='black', label = 'Stimulation')
    legend_added = False
    ax1.axvline(x=adjust_thres1, color='r', linestyle='--', label= f'Threshcold Entrainment >{np.round(threshold, decimals = 2)} STD')
    ax2.set_ylabel('Stimulation Amplitude [mA]')
    ax1.set_ylabel('Envelope Z-Scored to Rest M1S0')
    plt.xlim(0,adjust_thres1+3)
    ax1.set_xlabel('Time [sec]')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    fig.legend(lines, labels)   
    plt.title(f'{subID}Latency_Short') 
    plt.show()

    if save == 1:
        plt.savefig(os.path.join(path,
                                f'{subID}Latency_Short'
                                ), dpi = 200)


    filtered_array2save = np.vstack((filtered_subh, filtered_stim))

    if save == 1:
        np.save(os.path.join(path,
                f'{subID}_FilteredArray.npy'), filtered_array2save)
    return z_scores, adjust_thres1, filtered_array2save