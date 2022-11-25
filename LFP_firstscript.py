%matplotlib inline
import os
import json
from importlib import reload

from matplotlib import pyplot as plt
from scipy.signal import spectrogram, hanning
import numpy as np
import pandas as pd

import mne
from mne.time_frequency import tfr_morlet