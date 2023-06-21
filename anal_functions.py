import mne
import matplotlib.pyplot as plt
import numpy as np

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