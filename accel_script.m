%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('T:\Dokumente\CORE_CODE\fieldtrip-20220912');
addpath('T:\Dokumente\CORE_CODE\spm12\spm12');
addpath('T:\Dokumente\CORE_CODE\wjn_toolbox-master\wjn_toolbox-master');
addpath('T:\Dokumente\PROJECTS\DYSKINESIA_PROJECT\matlab_code');
spm('defaults','eeg')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

accel_fname = 'Sub-021_12mfu_dysk_segEnt_ramping_2-20220826T145332.DATA.Poly5';
dn_accel = preproc_poly(accel_fname);

data.time{1,1} = data.time{1,1} - data.time{1,1}(1);
sync_on = 11.26;
video_on = 25.28;
diff_trace = video_on - sync_on;
onset = diff_trace + 6.296;

data_dursec = data.time{1,1}(end);
sync_accel = accel_sync(dn_accel, onset, data_dursec)