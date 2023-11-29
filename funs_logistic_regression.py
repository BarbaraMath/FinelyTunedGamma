import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import find_folders
import mne
from scipy.stats import mannwhitneyu
import math
import dat_preproc
from scipy.signal import hilbert 

from scipy.signal import hilbert 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

def epoch_entrain_amp(SUBID, raw_entr, accel, peak_entr, amp_on, amp_off, saving, save_path):

    dat_ngam = dat_preproc.low_highpass_filter(raw_entr, peak_entr-2, peak_entr+2)
    svm_results = np.sqrt(accel[0,:]**2 + accel[1,:]**2 + accel[2,:]**2)
    
    #Calculate Envelope
    nan_mask = np.isnan(dat_ngam)
    data_no_nans = np.nan_to_num(dat_ngam)
    hilbert_transformed_data_no_nans = hilbert(data_no_nans)
    # Apply NaN mask to restore NaNs in the Hilbert-transformed data
    hiltr = np.where(nan_mask, np.nan, hilbert_transformed_data_no_nans)
    amplitude_envelope = np.abs(hiltr)

    #Epoch them
    dat_ngam_ep = amplitude_envelope[amp_on*250:amp_off*250]
    svm_ep = svm_results[amp_on*250:amp_off*250]

    original_dat = np.array([dat_ngam,svm_results])
    epoched_dat = np.array([dat_ngam_ep,svm_ep])
    
    if saving == 1:
        np.save(os.path.join(save_path, f'{SUBID}_Original_dat.npy'), original_dat)
        np.save(os.path.join(save_path, f'{SUBID}_Epoched_dat.npy'), epoched_dat)
    
    return original_dat, epoched_dat

def mean_smooth(SUBID, epoched_dat, window_size, saving, save_path):
    num_windows = epoched_dat.shape[1] // window_size

    mean_power_per_second = np.zeros(num_windows)
    svm_per_second = np.zeros(num_windows)

    # Calculate mean power and SVM every 1 second
    for i in range(num_windows):
        start_index = i * window_size
        end_index = (i + 1) * window_size
        
        # Extract the windowed data
        windowed_envelope = epoched_dat[0, start_index:end_index]
        windowed_svm = epoched_dat[1, start_index:end_index]

        # Calculate mean power for the window
        mean_power_per_second[i] = np.mean(windowed_envelope)

        # Calculate SVM for the window
        svm_per_second[i] = np.mean(windowed_svm)

        #mean_power_per_second = mean_power_per_second[~np.isnan(mean_power_per_second)]
        #svm_per_second = svm_per_second[~np.isnan(svm_per_second)]

        #new_xtps = np.arange(mean_power_per_second.shape[0])/250

    fig, axs = plt.subplots(2)
    axs[0].plot(svm_per_second)
    axs[0].set_ylabel('Mean SVM [1 sec]')
    axs[1].plot(mean_power_per_second)
    axs[1].set_ylabel('Mean Envelope [1 sec]')
    axs[1].set_xlabel('Time [sec]')
    axs[0].set_title(f'{SUBID}')

    if saving == 1:
        plt.savefig(os.path.join(save_path,
            f'{SUBID}_MeanDatSVM'), dpi = 150)
        
    plt.show()
        
    return mean_power_per_second, svm_per_second
        
def log_res_df(SUBID, epoch_df, row, mean_power_per_second, svm_per_second, save_path, saving):
    block1_on = int(epoch_df.loc[row,'Block1_on'])
    block1_off = int(epoch_df.loc[row,'Block1_off'])

    block2_on = int(epoch_df.loc[row,'Block2_on'])
    block2_off = int(epoch_df.loc[row,'Block2_off'])

    if not math.isnan(epoch_df.loc[row,'Block3_on']):
        block3_on = int(epoch_df.loc[row,'Block3_on'])
        block3_off = int(epoch_df.loc[row,'Block3_off'])
    else:
        block3_on = np.nan
        
    # Create log res df
    vec_label = np.zeros(len(svm_per_second))
    vec_label[block1_on:block1_off] = 1
    vec_label[block2_on:block2_off] = 1

    if not math.isnan(block3_on):
        vec_label[block3_on:block3_off] = 1
        
    regres_df = pd.DataFrame({
        'LFP_Power': mean_power_per_second,
        'SVM_Data': svm_per_second,
        'Label': vec_label
    })
    
    if saving == 1:
        regres_df.to_csv(os.path.join(
            save_path,
            f'{SUBID}_Regression_df.csv'
        ), index = False)
    
    return regres_df

def logistic_regression(SUBID, regres_df, saving, save_path):
    data_rest = regres_df['LFP_Power'][regres_df['Label'] == 0]
    data_tap = regres_df['LFP_Power'][regres_df['Label'] == 1]

    fig, ax = plt.subplots()

    ax.scatter(np.ones_like(data_rest) + np.random.normal(0, 0.1, size=len(data_rest)), data_rest,
            alpha=0.5, color='blue', s=40)  # s is the marker size
    ax.scatter(1.5 * np.ones_like(data_tap) + np.random.normal(0, 0.1, size=len(data_tap)), data_tap,
            alpha=0.5, color='green', s=40)

    # Set labels and title
    ax.set_xticks([1, 1.5])
    ax.set_xticklabels(['Rest', 'Tapping'])
    ax.set_ylabel('LFP Power')
    ax.set_title(f'{SUBID}')
    
    if saving == 1:
        plt.savefig(os.path.join(save_path,
            f'{SUBID}_TrueLabels'), dpi = 150)
    plt.show()
    
    ### LOGISTIC REGRESSION ###
    thresholds_reg = np.linspace(0, max(regres_df['LFP_Power']), num = 100)
    true_labels = regres_df['Label']

    all_true_pos_r = []
    all_false_pos_r = []

    for thres_reg in thresholds_reg:
        predicted_labels = (regres_df['LFP_Power'] > thres_reg).astype(int)

        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        true_negative, false_positive, false_negative, true_positive = conf_matrix.ravel()

        true_pos_r = true_positive / (true_positive + false_negative)
        false_pos_r = false_positive / (false_positive + true_negative)

        all_true_pos_r.append(true_pos_r)
        all_false_pos_r.append(false_pos_r)

    true_false = np.array([all_true_pos_r, all_false_pos_r])
    
    ### CREATE ROC ###
    roc_auc = auc(true_false[1], true_false[0])

    fig, ax = plt.subplots()
    plt.plot(all_false_pos_r, all_true_pos_r, 'o')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

    plt.text(0.7,0,f'AUC = {np.round(roc_auc, decimals = 3)}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {SUBID}')
    
    
    if saving == 1:
        np.save(os.path.join(save_path, f'{SUBID}_TrueFalse.npy'), true_false)
        
        plt.savefig(os.path.join(save_path,
            f'{SUBID}_ROC'), dpi = 150)
        
    plt.show()

    print(f'AUC = {roc_auc}')
    return true_false, roc_auc
