####### NORMALIZATION #######
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

def sum_normalization(Sxx):
   """
   sum_normalization performs a total sum normalization of the recording.
   Preferably apply this only to recordings cleared of stimulation or other artefacts

   input <-- Sxx:
   np.array of shape channels x freqs x time e.g. data = raw.get_data(picks=[0,1]) 
   data transformed to the time frequency domain

   output <-- Sxx_norm
   """
   fig, ax = plt.subplots(1,2, figsize = (18,6))
   fig.suptitle('Normalized Spectrogram')
   
   #pre-allocate empty array 
   Sxx_norm = np.empty(shape = (Sxx.shape[0], Sxx.shape[1], Sxx.shape[2]))
   Sxx_norm[:] = np.nan

   for kl in np.array([0,1]):
      #Sxx_norm = np.array([[np.nan] * Sxx.shape[2]] * Sxx.shape[1])

      Sxx_chan = np.array([[np.nan] * Sxx.shape[2]] * Sxx.shape[1])
      Sxx_totalsum = np.sum(Sxx[kl,:,:], axis = 1)
      
      for j in range(Sxx.shape[2]):
         #A. Total Sum Normalization
         Sxx_thisnorm = (Sxx[kl,:,j]/Sxx_totalsum)*100
         Sxx_chan[:,j] = Sxx_thisnorm

         #B. Julian's Method
         #Sxx_norm[kl,:,:] = (Sxx[kl,:,j]/np.max(Sxx[kl,:,:], axis = 1))
         #Sxx_norm[kl,:,:] = Sxx_norm[:,j]-np.mean(Sxx_norm[kl,:,j])

         #C. Mean Normalization
         #Sxx_thisnorm = (Sxx[chan,:,j]-np.mean(Sxx[chan,:,:], axis = 1))/np.std(Sxx[chan,:,:], axis = 1)
         #Sxx_norm[:,j] = Sxx_thisnorm
      #Sxx_chan = np.expand_dims(Sxx_chan, axis = 0)
      
      Sxx_norm[kl,:,:] = Sxx_chan
      ax[kl].pcolormesh(Sxx_chan, cmap = 'viridis')
      ax[kl].set_ylim(5,100)
   
   plt.show(block = False)
   
   return Sxx_norm

