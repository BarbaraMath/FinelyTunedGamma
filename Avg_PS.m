%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('T:\Dokumente\CORE_CODE\wjn_toolbox-master\wjn_toolbox-master\');
addpath('T:\Dokumente\CORE_CODE\fieldtrip-20220912');
addpath('T:\Dokumente\CORE_CODE\spm12\spm12');

spm('defaults','eeg')
%% figure with averaged power spectra

spontan = readtable('AllSpontan_PsZscored.csv')
subharm = readtable('AllSubharm_PsZscored.csv')


spontan = removevars(spontan, 'Sub029');

figure
mypower(1:126, nanmean(spontan{:,:},2),'blue','move')
hold on
mypower(1:126, nanmean(subharm{:,:},2),'magenta','move')

xlim([50 100])

%%








