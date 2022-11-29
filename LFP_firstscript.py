import os
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from importlib import reload


from matplotlib import pyplot as plt
from scipy.signal import spectrogram, hanning
import numpy as np
import pandas as pd

import mne
from mne.time_frequency import tfr_morlet

#os.getcwd() for finding current working directory
os.chdir('')

raw = mne.io.read_raw_fieldtrip()

print(raw) #here you can see the n samples and the time
print(raw.info) #ch_names, sfreq, nchan.
n_time_samps = raw.n_times #nsamples
time_secs = raw.times #timepoints
raw.info.keys()
ch_names = raw.ch_names
n_chan = len(ch_names) 

print('Hi')