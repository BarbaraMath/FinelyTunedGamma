
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
data = new_raw.get_data(picks=[0,1])
pass_filtered_dat = low_highpass_filtering(data)

#FFT TRANSFORMATION & PLOTTING
x = filt_dat
win_samp = 250
noverlap = 0.5
new_fname = 'FFTClean_sub-021_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.npy'
f, t, Sxx = fft_rawviz(new_raw, x, win_samp, noverlap, new_fname)

#EPOCH AND PLOT
time_onsets = {'1-200 sec': 1,
              '200-400 sec': 200,
              '400-600 sec': 400,
              '600-800 sec': 600}
window = 250
noverlap = 0.5*250
side = 1
ps = epoch_PS(filt_dat, time_onsets, window, noverlap, side)

plt.plot(np.arange(1, 127), np.mean(Sxx[side,:,1:200], axis = 1), label = 'NoStim')
plt.plot(np.arange(1, 127), np.mean(Sxx[side,:,200:400], axis = 1), label = 'Clinical')
plt.plot(np.arange(1, 127), np.mean(Sxx[side,:,400:600], axis = 1), label = 'Threshold')
plt.plot(np.arange(1, 127), np.mean(Sxx[side,:,600:800], axis = 1), label = 'Over')
plt.legend()
plt.xlim(50, 100)
plt.ylim(0, 0.06)
plt.show()

### try sum of squared differences
a = np.mean(Sxx[side,:,1:200], axis = 1)
b = np.mean(Sxx[side,:,200:400], axis = 1)
c = np.mean(Sxx[side,:,400:600], axis = 1)
d = np.mean(Sxx[side,:,600:800], axis = 1)

bdiff = (b-a)
cdiff = (c-a)
ddiff = (d-a)

plt.plot(bdiff)
plt.plot(cdiff)
plt.plot(ddiff)
plt.xlim(40, 100)
plt.ylim(-0.02, 0.03)
plt.show()


###

#BASELINE CORRECTION WITHIN SAME RECORDING
Sxx = np.load('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\FFT_sub-021_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.npy')
chan = 0
data = Sxx[chan]
t = np.arange(1,Sxx.shape[2]+1)
baseline = (1,100)
stim_ch = 4
new_bcfname = 'BC-Stim_sub-021_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01'
bs_data = baseline_corr(new_raw, data, t, baseline, stim_ch, new_bcfname)

plt.savefig('BCLSTN-Stim_sub-021_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.pdf')

#BASELINE CORRECTION WITH M0S0 (different recording)
m0s0_data = scipy.io.loadmat('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\Sub021_M0S0_BSTD_2022-05-20T081151_ZERO_TWO_RL.mat')

m0s0_data = m0s0_data['data']


m0s0_data_filt = scipy.signal.filtfilt(b, a, m0s0_data) # .get_data()
window = hann(win_samp, sym=False)
f_m0s0, t_m0s0, Sxx_m0s0 = signal.spectrogram(x = m0s0_data_filt, fs = fs, window = window, noverlap = noverlap)
plt.specgram(x = m0s0_data_filt[0,:], Fs = fs, noverlap = noverlap, cmap = 'viridis',
                        vmin = -25, vmax = 10)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('M0S0 LSTN Rest')
plt.ylim(5,100)
plt.show(block = False)


#NORMALIZE BOTH Sxx & Sxx_m0s0

#Sxx normalization
Sxx_norm = normalization(Sxx)
Sxx_m0s0_norm = normalization(Sxx_m0s0)

plt.pcolormesh(Sxx_norm[0], cmap = 'viridis')
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
plt.ylim(5,100)
cbar = fig.colorbar(cf, ax = ax)
cbar.set_label('zlogratio')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('LSTN: M1S0 Corrected to M0S0')
plt.show(block = False)

plt.savefig('Sub021_M1S0_LSTN_bcm0s0.pdf')

bs_corrected.min()

