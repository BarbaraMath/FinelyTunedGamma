from fooof import FOOOF
import os 
import numpy as np
import json
from matplotlib import pyplot as plt


def fit_fooof(SUBID, PATH, spectrum):

    # Set the frequency range to fit the model
    freqs = np.arange(1,127)
    freq_range = [1, 90]

    # Initialize a FOOOF object
    fm = FOOOF(
        max_n_peaks=6, 
        aperiodic_mode='knee')

    fm.fit(freqs, spectrum, freq_range)

    print(f'{SUBID}: Offset = {fm.aperiodic_params_[0]}, '
        f'Knee = {fm.aperiodic_params_[1]}, '
        f'Exponent = {fm.aperiodic_params_[2]}')

    # Plotting Periodic Components
    fig, ax = plt.subplots(1,1, figsize=[12,6])
    fm.plot(plot_peaks='dot-shade-outline', add_legend=True, ax=ax)
    plt.title(f'Error: {np.round(fm.error_, decimals = 4)}, R-squared: {np.round(fm.r_squared_, decimals = 4)}')
    
    #plt.savefig(os.path.join(
    #    PATH, f'{SUBID}_FOOOF_FIT.jpg'),
   #     dpi=200
    #)

    file_name = f'{SUBID}_fooof_fit.json'
    file_path = os.path.join(PATH,file_name)
    #fm.save(file_path, save_results=True, save_settings=True, save_data=True)

    return fm
