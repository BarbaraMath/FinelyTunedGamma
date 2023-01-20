#### CLEAR RECORDING FROM ARTEFACTS AND NORMALIZE ####

raw = mne.io.read_raw_fieldtrip('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub033\\sub-20210902PSTN_ses-2022041211000080_run-BrainSense20220902120700.mat', info = None)

fig = raw.plot(n_channels = 2, highpass = 5, lowpass = 100, filtorder = 5, duration = 20)
fig.fake_keypress('a')

#First time making them
interactive_annot = raw.annotations
raw.annotations.save('Sub033_RampUpThres_artefactsAnnotations.csv', overwrite = True)

#Importing them later
annot_from_file = pd.read_csv('Sub021_Freq110Hz_artefactsAnnotations.csv')
new_onsets = fix_annot_onsets(annot_from_file)
print(new_onsets)

my_annot = mne.Annotations(onset=new_onsets,  # in seconds
                           duration=annot_from_file.duration,  # in seconds, too
                           description=annot_from_file.description,
                           orig_time=raw.info['meas_date'])

raw2 = raw.copy().set_annotations(interactive_annot)
Draw = raw2.get_data(reject_by_annotation = 'omit')
info = raw.info
new_raw = mne.io.RawArray(Draw, info)
new_raw.plot(n_channels = 2, highpass = 5, lowpass = 100, filtorder = 5, duration = 20)

new_raw.save('Rej_sub-20210902PSTN_ses-2022041211000080_run-BrainSense20220902120700.fif')

new_raw = mne.io.read_raw_fif('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\Rej_sub-20210511PStn_ses-2022082612233578_run-BrainSense20220826125900.fif')

#FILTERING
filter_order = 5 
frequency_cutoff_low = 5 
frequency_cutoff_high = 100 
fs = new_raw.info['sfreq'] # sample frequency: 250 Hz
# create the filter
b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
data = new_raw.get_data(picks = (0,1))
filt_dat = scipy.signal.filtfilt(b, a, data) # .get_data()

# filter bandstop 
lowcut = 70
highcut = 74
order = 4
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = signal.butter(order, [low, high], btype='bandstop')
data = new_raw.get_data(picks = (0,1))
filt_dat = signal.filtfilt(b,a,data)

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