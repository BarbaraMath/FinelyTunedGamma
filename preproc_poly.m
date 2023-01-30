function dn_accel = preproc_poly(accel_fname)
    addpath('T:\Dokumente\CORE_CODE\fieldtrip-20220912');
    addpath('T:\Dokumente\CORE_CODE\spm12\spm12');
    addpath('T:\Dokumente\CORE_CODE\wjn_toolbox-master\wjn_toolbox-master')

    spm('defaults','eeg')

    accel = spm_eeg_convert(accel_fname);

    % Downsampling accel traces
    S.D = accel; %create new structure
    S.fsample_new = 250; %new sampling rate
    dn_accel = spm_eeg_downsample(S); %downsample it
    
    figure; wjn_plot_raw_signals(dn_accel.time,dn_accel,dn_accel.chanlabels)
end