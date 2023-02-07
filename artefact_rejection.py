#### CLEAR RECORDING FROM ARTEFACTS AND NORMALIZE ####




new_raw.save('Rej_sub-20210902PSTN_ses-2022041211000080_run-BrainSense20220902120700.fif')

raw = mne.io.read_raw_fif('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\Rej_sub-20210511PStn_ses-2022082612233578_run-BrainSense20220826012900.fif')


# filter bandstop 
lowcut = 70
highcut = 74
data = new_raw.get_data(picks = (0,1))

#FFT TRANSFORMATION & PLOTTING
x = filt_dat
win_samp = 250
noverlap = 0.5
window = hann(win_samp, sym=False)
f, t, Sxx = signal.spectrogram(x = x, fs = fs, window = window, noverlap = noverlap)
plt.specgram(x = x[1,:], Fs = fs, noverlap = noverlap, cmap = 'viridis',
                        vmin = -25, vmax = 10)
plt.ylim(5, 100)
plt.show(block = False)

f, t, Sxx = fft_transform(new_raw, filt_dat, win_samp, noverlap)

np.save('FFT_sub-033_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.npy', Sxx)
plt.savefig('FFT_sub-021_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.pdf')