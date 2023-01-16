
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


#FILTERING
filter_order = 5 
frequency_cutoff_low = 5 
frequency_cutoff_high = 100 
fs = raw.info['sfreq'] # sample frequency: 250 Hz
# create the filter
b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
data = raw.get_data(picks=[0,1])
filt_dat = scipy.signal.filtfilt(b, a, data) # .get_data()

#FFT TRANSFORMATION & PLOTTING
x = filt_dat
win_samp = 250
noverlap = 0.5
f, t, Sxx = fft_transform(x, win_samp, noverlap)

#EPOCH AND PLOT
time_onsets = {'No_Stim': 1,
              'Clinical': 102,
              'Threshold': 820}
window = 250
noverlap = 0.5*250
side = 1
ps = epoch_PS(filt_dat, time_onsets, window, noverlap, side)

plt.plot(np.arange(1, 127), np.mean(Sxx[1,:,1:10], axis = 1), label = 'NoStim')
plt.plot(np.arange(1, 127), np.mean(Sxx[1,:,102:112], axis = 1), label = 'Clinical')
plt.plot(np.arange(1, 127), np.mean(Sxx[1,:,820:830], axis = 1), label = 'Threshold')
plt.legend()
plt.xlim(5, 40)
plt.show()

#BASELINE CORRECTION WITHIN SAME RECORDING
data = Sxx[1]
t = t
baseline = (1,100)
stim_ch = 5
bs_data = baseline_corr(data, t, baseline, stim_ch)


#BASELINE CORRECTION WITH M0S0 (different recording)
m0s0_data = scipy.io.loadmat('\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\Sub021_M0S0_BSTD_2022-05-20T081151_ZERO_TWO_RL.mat')

m0s0_data = m0s0_data['data']

m0s0_data_filt = scipy.signal.filtfilt(b, a, m0s0_data) # .get_data()
window = hann(win_samp, sym=False)
f_m0s0, t_m0s0, Sxx_m0s0 = signal.spectrogram(x = m0s0_data_filt, fs = fs, window = window, noverlap = noverlap)
plt.specgram(x = m0s0_data_filt[1,:], Fs = fs, noverlap = noverlap, cmap = 'viridis',
                        vmin = -25, vmax = 10)
plt.ylim(5,100)
plt.show(block = False)


#NORMALIZE BOTH Sxx & Sxx_m0s0

#Sxx normalization
Sxx_norm = normalization(Sxx)
Sxx_m0s0_norm = normalization(Sxx_m0s0)

plt.pcolormesh(Sxx_norm[1], cmap = 'viridis')
plt.ylim(5,100)
plt.show(block = False)


baseline_data = Sxx_m0s0
m1s1_data = Sxx
chan = 0

avg_m0s0 = np.mean(baseline_data[chan,:,:], axis = 1)
log_m0s0 = np.log10(baseline_data[chan,:,:])
std_bs = np.std(log_m0s0)

bs_corrected = np.array([[np.nan] * Sxx.shape[2]] * Sxx.shape[1])
print(f'shape of nan array: {bs_corrected.shape}')

for k in range(m1s1_data[chan,:,:].shape[1]): #for each column (i.e. time in seconds)
       #corr_line1 = ((Sxx_norm[1,:,k] - avg_m0s0)/avg_m0s0)*100
       corr_line0 = np.log10(m1s1_data[chan,:,k]/avg_m0s0)
       corr_line1 = corr_line0/std_bs
       bs_corrected[:,k] = corr_line1

fig, ax = plt.subplots(1,1,figsize = (7,5))
cf = plt.pcolormesh(bs_corrected, cmap = 'viridis', vmin = -0.2, vmax = 0.3)
plt.ylim(40,100)
cbar = fig.colorbar(cf, ax = ax)
plt.show(block = False)

bs_corrected.min()

