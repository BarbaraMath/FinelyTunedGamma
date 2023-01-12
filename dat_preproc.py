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


#FFT Transformation and Spectrogram Plotting
def fft_transform(x, win_samp, noverlap):
        
        # Input:
        #x = filt_dat
        #win_samp = window for fft in samples, e.g. 250 for 1 sec
        #noverlap e.g. 0.25 (for 25%)

        fs = raw.info["sfreq"]
        window = hann(win_samp, sym=False)
        f, t, Sxx = signal.spectrogram(x = x, fs = fs, window = window, noverlap = noverlap)
         
        #Plot Spectrograms of both STNs
        fig, axes = plt.subplots(1,2, figsize = (18,6))
        fig.suptitle('FFT Transformations')

        ax_c = 0
        stim = 4
        for kj in np.array([0,1]):
                
                ax2 = axes[kj].twinx() #make right axis linked to the left one

                stim_data = (raw.get_data(picks = stim)[0,:]/3) #define stim channel
                axes[ax_c].specgram(x = x[kj,:], Fs = fs, noverlap = noverlap, cmap = 'viridis',
                        vmin = -25, vmax = 10)
                axes[ax_c].set_ylim(bottom = 5,top = 100)
                
                #Plot stim channel on top
                ax2.plot(raw.times, stim_data, 'w', linewidth = 1.5)
                ax2.set_yticks(np.arange(0,4.5,0.5))

                #Right y axis label only for second plot to avoid crowd
                if kj == 1:
                        ax2.set_ylabel('Stimulation Amplitude [mA]')
                
                axes[ax_c].set_ylabel('Frequency [Hz]')
                axes[ax_c].set_xlabel('Time [sec]')
                axes[ax_c].set_title(raw.ch_names[kj])

                ax_c += 1
                stim += 1

        
        plt.show(block = False)

        return f, t, Sxx
        return fig
        

#PS Plotting of epochs in no stim/clinical/threshold
def epoch_PS(filt_dat, time_onsets, window, noverlap, side):

        ps = np.empty([len(time_onsets),126])

        for key, value in time_onsets.items():
                this_onset = value*250
                this_offset = this_onset + (10*250)
                
                #window = hann(250, sym=False)

                ff, Pxx = scipy.signal.welch(filt_dat[side,this_onset:this_offset], fs = 250, 
                        nperseg = window, noverlap = noverlap)
                
                #xnew = np.linspace(ff.min(), ff.max(), 50)
                #spl = make_interp_spline(ff, Pxx, k = 3)
                #y_smooth = spl(xnew)
                
                
                #plt.plot(xnew, y_smooth, label = key)
                plt.plot(ff, Pxx, label = key)
                #ps1 = ps.append(ps, Pxx)
        plt.xlim([5,40])

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V^2/Hz]')
        plt.legend()
        plt.show(block = False)

        return ps



