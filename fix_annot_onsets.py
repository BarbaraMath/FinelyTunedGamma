#### FIX ANNOTATIONS ONSETS ####

def fix_annot_onsets(annot_from_file):

    new_onsets = np.empty(shape = (len(annot_from_file.onset)))
    new_onsets[:] = np.nan

    for i in np.arange(len(annot_from_file.onset)):
        a = annot_from_file.onset[i]
        x = datetime.strptime(a,'%M:%S.%f')
        onset_corr = x.minute*60+x.second+x.microsecond/1000000
        new_onsets[i] = onset_corr

    return new_onsets




