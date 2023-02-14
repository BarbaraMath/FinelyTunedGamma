import mne

def mat2fif(raw, my_annot):
    raw2 = raw.copy().set_annotations(my_annot)
    
    Draw = raw2.get_data(reject_by_annotation = 'NaN')
    info = raw.info
    
    new_raw = mne.io.RawArray(Draw, info)
    
    new_raw.plot(n_channels = 2, highpass = 5, lowpass = 100, 
        filtorder = 5, duration = 20)
    
    return new_raw
