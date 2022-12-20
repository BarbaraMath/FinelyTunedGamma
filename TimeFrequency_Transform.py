
trial_onsets = np.array([
       [ 1 ,      0 ,    1],
       ]) #we need the samples
event_dict = {'One_epoch':1}

fig = mne.viz.plot_events(trial_onsets, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)

epochs = mne.Epochs(raw,events=trial_onsets,event_id = event_dict, tmin=0, tmax=200, baseline = None, preload = True)

freqs = np.arange(1, 101)
D = tfr_morlet(epochs, freqs=freqs, n_cycles=6, return_itc=False, average=True, picks = 1)

D.plot(mode='zlogratio', picks = 0, baseline=None,
        vmin = -20, vmax = 150,
       cmap='inferno',
)


wav = mne.time_frequency.tfr_morlet(inst = epochs, freqs = np.arange(3,100), n_cycles=7, zero_mean=True, use_fft=False,
       return_itc=False, decim=int(raw.info['sfreq'] / 20), n_jobs=1, picks=1, average=False, output='power', verbose='INFO')


#FILTERING FUNCTION
lowcut = 100
highcut = 3
nyq = raw.info["sfreq"] / 2
data = raw.get_data(picks=[0,1])
filt_dat = data_filtering(lowcut, highcut, nyq, data)

#FFT TRANSFORMATION & PLOTTING
x = filt_dat
win_samp = 250
noverlap = 0.25
f, t, Sxx = fft_transform(x, win_samp, noverlap)

