#Data Filtering
def data_filtering(lowcut, highcut, nyq, data):
    
    #data e.g. raw.get_data(picks=[0,1])

    order = 5 #standard

    b, a = signal.butter(
            order,
            [highcut / nyq],
            btype='highpass'
    )

    b, a = signal.butter(
            order,
            [lowcut / nyq],
            btype='lowpass'
    )

    filt_dat = filtfilt(b,a, data)
    
    return filt_dat #returns 2 x n_samples array


def fft_transform(x, win_samp, noverlap):
        #x = filt_dat
        #win_samp = window for fft in samples, e.g. 250 for 1 sec
        #noverlap e.g. 0.25 (for 25%)
        f, t, Sxx = signal.spectrogram(x = x, fs = raw.info["sfreq"], window = hann(win_samp, sym=False), noverlap = noverlap)
        return f, t, Sxx 
        #Sxx.shape: n_chan x n_freqs x seconds (n_samples/fs)
