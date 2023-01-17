#### CLEAR RECORDING FROM ARTEFACTS AND NORMALIZE ####

raw = mne.io.read_raw_fieldtrip('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\sub-20210511PStn_ses-2022082612233578_run-BrainSense20220826125900.mat', info = None)

fig = raw.plot(n_channels = 2, highpass = 5, lowpass = 100, filtorder = 5, duration = 20)
fig.fake_keypress('a')

#First time making them
interactive_annot = raw.annotations
raw.annotations.save('Sub021_RampUpThres_artefactsAnnotations.csv', overwrite = True)

#Second time importing them
annot_from_file = mne.read_annotations('Sub021_RampUpThres_artefactsAnnotations.csv')

my_annot = mne.Annotations(onset=annot_from_file.onset,  # in seconds
                           duration=annot_from_file.duration,  # in seconds, too
                           description=annot_from_file.description)

raw2 = raw.copy().set_annotations(my_annot)

trial_cropped = raw.crop_by_annotations(interactive_annot)

Draw = raw.get_data(reject_by_annotation = 'omit')

#FILTERING
filter_order = 5 
frequency_cutoff_low = 5 
frequency_cutoff_high = 100 
fs = raw.info['sfreq'] # sample frequency: 250 Hz
# create the filter
b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
data = Draw[0:2,:]
filt_dat = scipy.signal.filtfilt(b, a, data) # .get_data()

#FFT TRANSFORMATION & PLOTTING
x = filt_dat
win_samp = 250
noverlap = 0.5
window = hann(win_samp, sym=False)
f, t, Sxx = signal.spectrogram(x = x, fs = fs, window = window, noverlap = noverlap)
plt.specgram(x = x[1,:], Fs = fs, noverlap = noverlap, cmap = 'viridis',
                        vmin = -25, vmax = 10)
plt.show(block = False)