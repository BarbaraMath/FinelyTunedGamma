####### PLOT POWER SPECTRA #######

#A. AVERAGED PS
def power_spectrum(dat, xlim = (5,100)):
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

    plt.plot(np.arange(1,127),y)

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


