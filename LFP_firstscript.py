import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import scipy
import mne

from mne.time_frequency import tfr_morlet

#os.getcwd() for finding current working directory
os.chdir('')

FileNames = pd.read_csv('OneDrive - Charité - Universitätsmedizin Berlin\FTG_PROJECT\ImportFiles.csv', sep=',', header=None)
print(FileNames.values)

raw = mne.io.read_raw_fieldtrip()

print(raw) #here you can see the n samples and the time
print(raw.info) #ch_names, sfreq, nchan.
n_time_samps = raw.n_times #nsamples
time_secs = raw.times #timepoints
raw.info.keys()
ch_names = raw.ch_names
n_chan = len(ch_names) 

print('we made many changes')
np.array([1,2,3])
