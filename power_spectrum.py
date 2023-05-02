####### PLOT POWER SPECTRA #######
from scipy.signal import chirp, find_peaks, peak_widths
from matplotlib import pyplot as plt
import numpy as np


#A. AVERAGED PS
def power_spectrum(dat, xlim = (5,100), label = ''):
    """
    power_spectrum plots a (mean) power spectrum with the sem around the line as a shaded area

    inputs:
    - dat, np.array of shape n of power spectra to be averaged x freqs x time 
    - xlim, tuple that defines the limits of the x axis (optional input)
    """

    if dat.ndim == 2:
        p_vec = np.mean(dat,1)
        sem = 0
    elif dat.ndim > 2:
        p_meanall = np.mean(dat,2)
        p_vec = np.mean(p_meanall,0)
        sem = scipy.stats.sem(p_meanall,0)

    y = p_vec

    plt.plot(np.arange(1,127),y, label = label)

    if dat.ndim > 2:
        plt.fill_between(np.arange(1,127),y-sem, y+sem, alpha = 0.5)
    

    plt.xlim(xlim)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.show(block = False)

#B. WELCH PS

def pwelch(dat):
    #Function not functioning
    window = 250
    noverlap = 0.5*250
    Pwelch_f, Pwelch_x = scipy.signal.welch(dat, fs = 250, 
                        nperseg = window, noverlap = noverlap)

    plt.plot(Pwelch_f, Pwelch_x)

    plt.xlim(5,100)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V^2/Hz]')
    plt.legend()
    plt.show(block = False)

    return Pwelch_x


def powerSpectrum_width(x1, x2, height, subID):

    peaks1, _ = find_peaks(x1, height = height)
    print(peaks1+50)
    results_half1 = peak_widths(x1, peaks1, rel_height=0.5)

    peaks2, _ = find_peaks(x2, height = height)
    print(peaks2+50)
    results_half2 = peak_widths(x2, peaks2, rel_height=0.5)

    plt.plot(np.arange(0,51),x1, color = 'blue', label = 'MedOn-StimOff')
    plt.plot(peaks1,x1[peaks1+50],'x')

    plt.plot(np.arange(0,51),x2, color = 'red', label = 'MedOn-StimOn')
    plt.plot(peaks2,x2[peaks2+50],'x')

    plt.hlines(*results_half1[1:], color="blue")
    plt.hlines(*results_half2[1:], color="red")
    plt.xticks(np.arange(0,51,10), labels = np.arange(50,101,10))

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('LFP Power')
    plt.title(str(subID))

    plt.legend()

    return peaks1, results_half1[0], peaks2, results_half2[0]