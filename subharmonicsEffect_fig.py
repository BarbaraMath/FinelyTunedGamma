### FIGURE WITH STIMULATION SUBHARMONICS ###

raw110 = mne.io.read_raw_fieldtrip('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\sub-20210511PStn_ses-2022082612233578_run-BrainSense20220826011800.mat', info = None)
raw125 = mne.io.read_raw_fieldtrip('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\sub-20210511PStn_ses-2022082612233578_run-BrainSense20220826125900.mat', info = None)
raw145 = mne.io.read_raw_fieldtrip('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Sub021\\sub-20210511PStn_ses-2022082612233578_run-BrainSense20220826012900.mat', info = None)

#FILTERING
filter_order = 5 
frequency_cutoff_low = 5 
frequency_cutoff_high = 100 
fs = 250 # sample frequency: 250 Hz
# create the filter
b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), btype='bandpass', output='ba', fs=fs)
data = raw145.get_data(picks=[0,1])
filt_dat145 = scipy.signal.filtfilt(b, a, data) # .get_data()

#FFT TRANSFORMATION & PLOTTING
x = filt_dat110
raw = raw110
win_samp = 250
noverlap = 0.5
new_fname = 'FFT_sub-021_ses-DbsFu12mMedOn01_task-Freq110Hz_acq-Streaming_run-01.npy' #task-Freq110Hz
f, t, Sxx = fft_transform(raw, x, win_samp, noverlap, new_fname)

##LOAD NEW FILES
fft110 = np.load('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Subharmonics\\FFT_sub-033_ses-DbsFu12mMedOn01_task-Freq110Hz_acq-Streaming_run-01.npy')
fft125 = np.load('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Subharmonics\\FFT_sub-033_ses-DbsFu12mMedOn01_task-RampUpThres_acq-Streaming_run-01.npy')
fft145 = np.load('C:\\Users\\mathiopv\\OneDrive - Charité - Universitätsmedizin Berlin\\FTG_PROJECT\\Subharmonics\\FFT_sub-028_ses-DbsFu12mMedOn01_task-Freq145Hz_acq-Streaming_run-01.npy')

fig = plt.figure()
ax = [plt.subplot(1,3,i+1) for i in range(3)]
fig.suptitle('Sub028 - Subharmonics')
for a in ax[1:3]:
    a.set_yticklabels([])
for a in ax:  a.set_xlabel('Time [sec]')
#plt.subplots_adjust(wspace = 1, hspace = 0)

ax[0].pcolormesh(fft110[1,:,155:170], cmap = 'viridis', vmin = 0, vmax = 0.5)
ax[0].set_ylabel('Frequency [Hz]')
ax[0].set_title('3.7mA - 110Hz')
ax[1].pcolormesh(fft125[1,:,875:890], cmap = 'viridis', vmin = 0, vmax = 0.5)
ax[1].set_title('2.8mA - 125Hz')
ax[2].pcolormesh(fft145[1,:,875:890], cmap = 'viridis', vmin = 0, vmax = 0.5)
ax[2].set_title('2.8mA - 145Hz')


for a in ax:
    a.set_ylim(5,100)
plt.show(block = False)

plt.savefig('Sub033_Subharmonics.pdf')
plt.savefig('Sub033_Subharmonics.png')