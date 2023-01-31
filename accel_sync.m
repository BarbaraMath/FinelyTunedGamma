function sync_accel = accel_sync(dn_accel, onset, data_dursec)
    % Cut traces
    
    samples=[1:wjn_sc(dn_accel.time, onset)]; %change to second of cutting
    cut_accel = wjn_remove_bad_samples(dn_accel.fname,samples); 

    end_samples = [wjn_sc(cut_accel.time,data_dursec)+1:cut_accel.nsamples]; %add one
    sync_accel = wjn_remove_bad_samples(cut_accel.fname,end_samples); 
end