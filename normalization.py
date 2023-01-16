def normalization(Sxx):
   
   Sxx_norm = np.empty(shape = (Sxx.shape[0], Sxx.shape[1], Sxx.shape[2]))
   Sxx_norm[:] = np.nan

   for kl in range(2):
      #Sxx_norm = np.array([[np.nan] * Sxx.shape[2]] * Sxx.shape[1])

      Sxx_chan = np.array([[np.nan] * Sxx.shape[2]] * Sxx.shape[1])
      Sxx_totalsum = np.sum(Sxx[kl,:,:], axis = 1)
      
      for j in range(Sxx.shape[2]):
         #A. Total Sum Normalization
         
         Sxx_thisnorm = (Sxx[kl,:,j]/Sxx_totalsum)*100
         Sxx_chan[:,j] = Sxx_thisnorm

         #B. Julian's Method
         #Sxx_norm[:,j] = (Sxx[1,:,j]/Sxx[1,:,j].sum())*100
         #Sxx_norm[:,j] = Sxx_norm[:,j]-np.mean(Sxx_norm[:,j])

         #C. Mean Normalization
         #Sxx_thisnorm = (Sxx[chan,:,j]-np.mean(Sxx[chan,:,:], axis = 1))/np.std(Sxx[chan,:,:], axis = 1)
         #Sxx_norm[:,j] = Sxx_thisnorm
      #Sxx_chan = np.expand_dims(Sxx_chan, axis = 0)
      Sxx_norm[kl,:,:] = Sxx_chan

   return Sxx_norm