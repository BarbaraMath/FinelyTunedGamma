#DATA FILTERING

#A. LOW/HIGH PASS FILTER
def low_highpass_filter(data):
        """
        low_highpass_filter applies low (100Hz) and high pass (5Hz) filter, with a 
        5th order butterworth filter.

        input <-- data:
        np.array of shape channels x freqs x time e.g. data = raw.get_data(picks=[0,1])

        output <-- pass_filtered_dat:
        np.array with the same shape as input array

        requires: scipy.signal
        """
        filter_order = 5 
        frequency_cutoff_low = 5 
        frequency_cutoff_high = 100 
        fs = 250 
            
        # create the filter
        b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), 
                btype='bandpass', output='ba', fs=fs)              
        pass_filtered_dat = scipy.signal.filtfilt(b, a, data) 
    
        return pass_filtered_dat

#B. BANDSTOP FILTER
def bandstop_filter(lowcut, highcut, data):
        """
        bandstop_filter applies a bandstop filter around the low/highcut given

        input <-- data:
        np.array of shape channels x freqs x time e.g. data = raw.get_data(picks=[0,1])

        output <-- stop_filtered_dat:
        np.array with the same shape as input array

        requires: scipy.signal
        """      
        order = 4
        nyq = 0.5 * 250 #sampling rate
        low = lowcut / nyq
        high = highcut / nyq
        
        #create the filter
        b, a = signal.butter(order, [low, high], btype='bandstop')
        
        stop_filtered_dat = signal.filtfilt(b,a,data)

        return stop_filtered_dat

#FFT Transformation and Spectrogram Plotting
def fft_rawviz(raw, x, win_samp, noverlap):
        """
        fft_rawviz performs a Fast Fourier Transformation to data and creates TF plots 
        with stimulation amplitude on top

        # Input:
        #x = filt_dat
        #win_samp = window for fft in samples, e.g. 250 for 1 sec
        #noverlap e.g. 0.25 (for 25%)
        """
        fs = 250
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
                
                #Plot LFP data
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

        #np.save(new_fname, Sxx)
        return f, t, Sxx


def fft_transform(x, win_samp, noverlap):
        """
        fft_transform performs a Fast Fourier Transformation to data without plotting them

        input:
        - x = filt_dat as np.array of shape channel x freq x time
        - win_samp = window for fft in samples, e.g. 250 for 1 sec
        - noverlap e.g. 0.25 (for 25%)

        output:
        - f (frequencies), t (time), Sxx: transformed data of shape same as input data
        """
        fs = 250
        window = hann(win_samp, sym=False)
        
        f, t, Sxx = signal.spectrogram(x = x, fs = fs, window = window, noverlap = noverlap)

        plt.specgram(x = x, Fs = fs, noverlap = noverlap, cmap = 'viridis',
                        vmin = -25, vmax = 10)
        
        plt.ylim(5,100)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        return f, t, Sxx

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
        plt.xlim([60, 100])

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V^2/Hz]')
        plt.legend()
        plt.show(block = False)

        return ps



#make a change