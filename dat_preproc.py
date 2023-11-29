from matplotlib import pyplot as plt
import mne
import scipy
import numpy as np
#from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, hann, butter, filtfilt
import os

#DATA FILTERING

#A. LOW/HIGH PASS FILTER
def low_highpass_filter(data, frequency_cutoff_low, frequency_cutoff_high):
        """
        low_highpass_filter applies low (100Hz) and high pass (5Hz) filter, with a 
        5th order butterworth filter.

        input <-- data:
        np.array of data = raw.get_data(picks=[0,1])

        output <-- pass_filtered_dat:
        np.array with the same shape as input array

        requires: scipy.signal
        """
        filter_order = 5 
        #frequency_cutoff_low = 5 
        #frequency_cutoff_high = 100 
        fs = 250 
            
        # create the filter
        b, a = scipy.signal.butter(filter_order, (frequency_cutoff_low, frequency_cutoff_high), 
                btype='bandpass', output='ba', fs=fs)              
        
        # Identify NaN values in the data
        nan_mask = np.isnan(data)
        # Replace NaN values with zeros before filtering
        data_no_nans = np.nan_to_num(data)

        # Filter the data without NaNs
        pass_filtered_dat_no_nans = scipy.signal.filtfilt(b, a, data_no_nans)

        # Apply NaN mask to restore NaNs in the filtered data
        pass_filtered_dat = np.where(nan_mask, np.nan, pass_filtered_dat_no_nans)
 
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
        nyq = 0.5 * 4096 #sampling rate
        low = lowcut / nyq
        high = highcut / nyq
        
        #create the filter
        b, a = scipy.signal.butter(order, [low, high], btype='bandstop')
        
        stop_filtered_dat = scipy.signal.filtfilt(b,a,data)

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

        fs = raw.info['sfreq']
        window = hann(win_samp, sym=False)
        f, t, Sxx = scipy.signal.spectrogram(x = x, fs = fs, window = window, noverlap = noverlap)
         
        #Plot Spectrograms of both STNs
        fig, axes = plt.subplots(1,2, figsize = (18,6))
        fig.suptitle('FFT Transformations')

        ax_c = 0
        stim = 4
        for kj in np.array([0,1]):
                
                ax2 = axes[kj].twinx() #make right axis linked to the left one
                if kj == 1:
                        stim_data = (raw.get_data(picks = stim)[0,:]) #define stim channel
                elif kj == 0:
                        stim_data = (raw.get_data(picks = stim)[0,:])
                
                #Plot LFP data
                axes[ax_c].specgram(x = x[kj,:], Fs = fs, noverlap = noverlap, cmap = 'viridis',
                        vmin = -25, vmax = 10) #-25,10
                axes[ax_c].set_ylim(bottom = 0,top = 100)
                axes[ax_c].set_xlim(0,raw.n_times/250)
                
                #Plot stim channel on top
                ax2.plot(raw.times, stim_data, 'white', linewidth = 2.5, linestyle = 'dotted')
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
        
        f, t, Sxx = scipy.signal.spectrogram(x = x, fs = fs, window = window, noverlap = noverlap)

        plt.specgram(x = x, Fs = fs, noverlap = noverlap, cmap = 'viridis',
                        vmin = -25, vmax = 10)
        
        plt.ylim(5,100)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        return f, t, Sxx

#PS Plotting of epochs in no stim/clinical/threshold
def epoch_ps(filt_dat, epoch_df, window, noverlap, side, ylim2, title):
        """A dummy docstring."""
        ps = np.empty([epoch_df.shape[0],126])

        for index, row in epoch_df.iterrows():
                this_onset = epoch_df.onset[index]*250
                this_offset = this_onset + (epoch_df.duration[index]*250)
                
                #window = hann(250, sym=False)

                ff, Pxx = scipy.signal.welch(filt_dat[side,this_onset:this_offset], fs = 250, 
                        nperseg = window, noverlap = noverlap)
                
                #xnew = np.linspace(ff.min(), ff.max(), 50)
                #spl = make_interp_spline(ff, Pxx, k = 3)
                #y_smooth = spl(xnew)
                
                
                #plt.plot(xnew, y_smooth, label = key)
                plt.plot(ff, Pxx, label = epoch_df.description[index])
                ps[index] = Pxx
        
        plt.xlim([50, 100])
        plt.ylim([0, ylim2])

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [$V^{2}$/Hz]')
        plt.legend()
        plt.title(title)
        plt.show(block = False)

        return ps

from scipy.interpolate import make_interp_spline, BSpline
from scipy import stats

def mypower(ps):
        
        x = np.arange(1,127)
        xvals = np.linspace(1,127,1250)

        if ps.shape[1] > 1:
                y = np.mean(ps,1)
                spl = make_interp_spline(x,y, k=3)  # type: BSpline
                power_smooth = spl(xvals)
                sem = stats.sem(ps,1)
                spl_sem = make_interp_spline(x,sem,k=3)  # type: BSpline
                sem_smooth = spl_sem(xvals) 

                plt.plot(xvals, power_smooth)
                plt.fill_between(xvals, power_smooth - sem_smooth, power_smooth + sem_smooth, alpha = 0.3)
        else:
                y = ps
                spl = make_interp_spline(x,y, k=3)  # type: BSpline
                power_smooth = spl(xvals)
                plt.plot(xvals, power_smooth)

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Spectral Power')

def peak_function(raw, subID, SIDE, peakStim, resutls_path, figs_path, SAVE):

        x = raw.get_data(reject_by_annotation = 'omit')
        

        if SIDE == 0:
                STIM = 4
        elif SIDE == 1:
                STIM = 5

        x1 = x[SIDE,:]
        stim_vec = x[STIM,:]

        dat_subh = low_highpass_filter(x1, peakStim-2, peakStim+2) 
        #Peaks
        i_peaks, props = scipy.signal.find_peaks(dat_subh, distance=2)
        # Adjust peak indices to account for NaN values
        #plt.plot(dat_subh, alpha=.3,)
        #plt.scatter(i_peaks, dat_subh[i_peaks], s=30,
        #    color='orange', alpha=.5)


        #Interpeaks 
        ipi_total = np.diff(i_peaks)

        #Intrapeaks Interval Variation
        window_samp = 60
        ipi_var = [np.std(ipi_total[i:i + window_samp])
                for i in np.arange(0, len(ipi_total) - window_samp)]

        stim_vec_for_ipi = stim_vec[i_peaks[:-1] + 1]

        ##### PLOT IT #####
        fig, ax = plt.subplots(figsize = (18,6))
        ax.plot(ipi_var)

        ax1 = ax.twinx()
        ax1.plot(stim_vec_for_ipi/10, color = 'orange')

        yticks = ax1.get_yticks()
        yticklabels = [np.round(tick * 10, decimals = 2) for tick in yticks]
        ax1.set_yticklabels(yticklabels)

        ax.set_xlabel('Time [samples]')
        ax.set_ylabel('Intrapeak Interval Variation')
        ax1.set_ylabel('Amplitude [mA]')

        ax.set_title(str(subID))

        ipiVar_npy = np.array([ipi_var, stim_vec_for_ipi])

        if SAVE == 1:
                plt.savefig(os.path.join(
                        resutls_path,
                        f'{subID}_IntraTapVar'
                        ), dpi = 200)


                np.save(os.path.join(resutls_path, f'{subID}_Peaks.npy'), i_peaks)
                np.save(os.path.join(resutls_path, f'{subID}_IPI.npy'), ipi_total)
                np.save(os.path.join(resutls_path, f'{subID}_IPI_Var.npy'), ipiVar_npy)

        return x, i_peaks, ipi_total, ipiVar_npy
