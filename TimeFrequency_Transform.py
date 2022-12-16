
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



raw_filt = raw.filter(l_freq = 100, h_freq=2, picks=[0,1], method = 'fir')

f, t, Sxx = signal.spectrogram(x = raw.get_data(picks='LFP_R_13_STN'), fs = raw.info["sfreq"], window = hann(250, sym=False), noverlap = 0.25)

plt.pcolormesh(t, f, Sxx[0,:,:])
plt.ylim(5, 120)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("Time frequency plot "+str(raw.ch_names[1]))
plt.show()


plt.plot(np.arange(1,127), np.mean(Sxx[0,:,:],1))
plt.xlim(10,100)
plt.ylim(0,1)
plt.show()