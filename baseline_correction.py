####### BASELINE CORRECTION #######

def baseline_corr(data, t, baseline, raw = 0, stim_ch = 0):
    """
    baseline_corr performs a zlogratio baseline correction of the input recording

    input:
    - data: np.array of shape 1 x frequencies x time
    - t: last dimension of the data (time)
    - baseline: tuple with two numbers i.e. the baseline e.g. (1,100)
    - raw: optional argument, when we want to ploit stimulation on top of recording
    - stim_ch: optional argument, to specify stimulation channel e.g. 4(LSTN) or 5(RSTN)

    """
    bs_data = mne.baseline.rescale(data = data, times = t, baseline = baseline, mode = 'zlogratio')

    fig, ax = plt.subplots(1,1,figsize = (7,7))
    cf = ax.pcolormesh(bs_data, cmap = 'viridis', vmin = -1, vmax = 3)

    if raw != 0:
        ax2 = ax.twinx()
        stim_data = (raw.get_data(picks = stim_ch)[0,:]/3)
        ax2.plot(raw.times, stim_data,'w',linewidth = 1.5)
        ax2.set_yticks(np.arange(0,4.5,0.5))
        ax2.set_ylabel('Stimulation Amplitude [mA]')
    
    
    ax.set_ylim(5, 100)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title('Zlogratio Baseline Correction to 0-100 sec')

    cbar = fig.colorbar(cf, ax = ax, location = 'bottom', pad = 0.1, shrink = 0.5)
    cbar.set_label('zlogratio')

    plt.show(block = False)

    #np.save(new_bcfname + '.npy', bs_data)
    #plt.savefig(new_bcfname + '.pdf')
    return bs_data