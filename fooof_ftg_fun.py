from fooof import FOOOF
import os 
import numpy as np
import json
from matplotlib import pyplot as plt


def fit_fooof(SUBID, PATH, spectrum):
    # Initialize a FOOOF object
    fm = FOOOF(min_peak_height=1,
        max_n_peaks=10,
        #peak_threshold=10,
        )

    # Set the frequency range to fit the model
    freqs = np.arange(1,127)
    freq_range = [1, 90]

    # Initialize a FOOOF object
    fm = FOOOF(
        aperiodic_mode="fixed", # fitting without knee component
        verbose=True,)

    fm.fit(freqs, spectrum, freq_range)
    
    # Plotting Periodic Components
    fig, ax = plt.subplots(1,1, figsize=[12,6])
    fm.plot(plot_peaks='dot-shade-outline', add_legend=True, ax=ax)

    plt.savefig(os.path.join(
        PATH, f'{SUBID}_FOOOF_FIT.jpg'),
        dpi=200
    )

    file_name = f'{SUBID}_fooof_fit.json'
    file_path = os.path.join(PATH,file_name)
    fm.save(file_path, save_results=True, save_settings=True, save_data=True)

    return fm
