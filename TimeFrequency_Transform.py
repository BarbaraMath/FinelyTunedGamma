
raw = mne.io.read_raw_fif('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub033\\Rej_sub-20210902PSTN_ses-2022041211000080_run-BrainSense20220902120700.fif')

#FILTERING
data = raw.get_data(picks=[0,1])
pass_filtered_dat = low_highpass_filter(data)

#FFT TRANSFORMATION & PLOTTING
x = pass_filtered_dat
win_samp = 250
noverlap = 0.5
f, t, Sxx = fft_rawviz(raw, x, win_samp, noverlap)

np.save('FFT_sub-20210902PSTN_ses-2022041211000080_run-BrainSense20220902122700.npy',Sxx)

#PLOT POWER SPECTRA IN 0.0 - CLINICAL - THRESHOLD mA
Sxx = np.load('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub033\\FFT_sub-033_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.npy')
power_spectrum(dat = Sxx[1,:,1:10], xlim = (5,40), label = 'NoStim')
power_spectrum(dat = Sxx[1,:,525:535], xlim = (5,40), label = 'Clinical')
power_spectrum(dat = Sxx[1,:,875:885], xlim = (5,40), label = 'Threshold')

plt.legend()

plt.savefig('Sub028_PS_Beta.jpg')
###

#BASELINE CORRECTION WITHIN SAME RECORDING
Sxx = np.load('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Subharmonics\\FFT_sub-021_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.npy')
chan = 1
data = Sxx[chan]
t = np.arange(1,Sxx.shape[2]+1)
baseline = (1,60)
stim_ch = 5
raw = raw
#new_bcfname = 'BC-Stim_sub-021_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01'
bs_data = baseline_corr(data, t, baseline, raw, stim_ch)

np.save('BCRSTN-Stim_sub-033_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.npy', bs_data)

plt.savefig('BCRSTN-Stim_sub-033_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.pdf')

#BASELINE CORRECTION WITH M0S0 (different recording)
m0s0_data = scipy.io.loadmat('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\Sub021_M0S0_BSTD_2022-05-20T081151_ZERO_TWO_RL.mat')

m0s0_data = m0s0_data['data']


m0s0_data_filt = low_highpass_filter(m0s0_data) # .get_data()
window = hann(win_samp, sym=False)
fs = 250
f_m0s0, t_m0s0, Sxx_m0s0 = signal.spectrogram(x = m0s0_data_filt, fs = fs, window = window, noverlap = noverlap)
plt.specgram(x = m0s0_data_filt[1,:], Fs = fs, noverlap = noverlap, cmap = 'viridis',
                        vmin = -25, vmax = 10)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('M0S0 RSTN Rest')
plt.ylim(5,100)
plt.show(block = False)


#NORMALIZE BOTH Sxx & Sxx_m0s0

#Sxx normalization
Sxx_norm = sum_normalization(Sxx)
Sxx_m0s0_norm = sum_normalization(Sxx_m0s0)

plt.pcolormesh(Sxx_norm[0], cmap = 'viridis')
plt.ylim(5,100)
plt.show(block = False)


baseline_data = Sxx_m0s0_norm
m1s1_data = Sxx_norm
chan = 1

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
plt.title('RSTN: M1S0 Corrected to M0S0')
plt.show(block = False)

plt.savefig('Sub021_M1S0_LSTN_bcm0s0.pdf')

bs_corrected.min()

## Plot Power Spectra 

Sxx = np.load('C:/Users/mathiopv\OneDrive - Charité - Universitätsmedizin Berlin/FTG_PROJECT/Sub021/FFT_sub-021_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.npy')
dat = Sxx
power_spectrum(dat=dat, xlim=(5,40))

acc_dat = open('C:/Users/mathiopv/OneDrive - Charité - Universitätsmedizin Berlin/FTG_PROJECT/Sub021/Sub-021_12mfu_dysk_segEnt_ramping_2-20220826T145332.DATA.Poly5','rb')
acc_dat1 = acc_dat.read()


#trial