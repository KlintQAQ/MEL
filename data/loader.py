import math
import os
import re

import mne
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt

from data import preprocessing as preproc

def parse_labels_roi(labels_roi: str):
    if "|" in labels_roi:
        return [name.strip() for name in labels_roi.split("|") if name.strip()]
    return labels_roi


def download_vu_dataset(sub, scan, path_to_dataset, output_eegmneraw=True, out_eegchan='EEG'):
    patient = 'sub' + '{:02d}'.format(int(sub)) + '-scan' + '{:02d}'.format(int(scan))
    eeg_path_set_file = os.path.join(path_to_dataset, 'EEG', f'{patient}_eeg.set')
    fMRI_path_difumo = os.path.join(path_to_dataset, 'fMRI_difumo', f'{patient}_difumo64_roi.pkl')
    new_fps = 200

    raw = mne.io.read_raw_eeglab(eeg_path_set_file)
    original_fps = raw.info['sfreq']
    if original_fps != new_fps:
        raw.resample(new_fps)
    if len(raw.ch_names) > 32:
        vector_exclude = ['EOG1', 'EOG2', 'EMG1', 'EMG2', 'EMG3', 'ECG',
                          'CWL1', 'CWL2', 'CWL3', 'CWL4']
    else:
        vector_exclude = ['EOG1', 'EOG2', 'EMG1', 'EMG2', 'EMG3', 'ECG']

    if output_eegmneraw == True:
        if out_eegchan == 'EEG':
            raw.drop_channels(vector_exclude)
        elif out_eegchan == 'EMG':
            raw.pick_channels(['EMG1', 'EMG2', 'EMG3'])
        df_eeg = raw
    else:
        df_eeg = raw.to_data_frame()
        df_eeg = df_eeg.drop(vector_exclude, axis=1)

    df_fmri = pd.read_pickle(fMRI_path_difumo)
    ROI_labels = df_fmri.columns.to_list()

    return df_eeg, df_fmri, ROI_labels


def _load_intrascan_epoch_data(args):
    seed = getattr(args, "seed", 16)
    torch.manual_seed(seed)
    np.random.seed(seed)

    root = args.dataset_root
    tr = args.TR
    event_sync_name = args.mri_sync_event
    tmin = -16
    tmax = 0
    crop = 3200

    sub_ind, scan_ind = re.findall(r'\d+', args.dataname)
    sub_ind, scan_ind = int(sub_ind), int(scan_ind)

    eeg_raw, fmri, labels_roi_full = download_vu_dataset(sub_ind, scan_ind, root, output_eegmneraw=True)
    fmri_np = fmri.to_numpy().T

    df = pd.DataFrame(data=fmri_np.T, columns=labels_roi_full)
    df_filter = df[parse_labels_roi(args.labels_roi)]
    fmri_np = df_filter.to_numpy().T
    if len(fmri_np.shape) < 2:
        fmri_np = fmri_np[np.newaxis, ...]

    fs = 1 / tr
    nyquist = 0.5 * fs
    low = 0.15 / nyquist
    b, a = butter(N=5, Wn=low, btype='low', analog=False)
    fmri_np = filtfilt(b, a, fmri_np, axis=1)
    fmri_norm, _ = preproc.normalize_data(fmri_np)

    data_epoch, _ = preproc.epoching_seq2one(
        eeg_raw, fmri_norm, tmin, tmax, event_sync_name, ifnorm=0, crop=crop
    )
    total = len(data_epoch["eeg"])
    train_end = int(0.8 * total)
    val_end = int(0.1 * total) + train_end

    eeg_train = data_epoch["eeg"][:train_end]
    fmri_train = data_epoch["fmri"][:train_end]

    t_overlap = 20 if abs(tmin) <= 20 else abs(tmin)
    gap_tr = math.ceil(t_overlap / tr)
    val_start = train_end + gap_tr
    eeg_val = data_epoch["eeg"][val_start:val_end]
    fmri_val = data_epoch["fmri"][val_start:val_end]

    test_start = val_end + gap_tr
    eeg_test = data_epoch["eeg"][test_start:]
    fmri_test = data_epoch["fmri"][test_start:]

    split_info = {
        "scan_name": args.dataname,
        "total_usable_samples": total,
        "train_samples": len(eeg_train),
        "val_samples": len(eeg_val),
        "test_samples": len(eeg_test),
        "test_duration_sec": float(len(eeg_test) * tr),
        "gap_tr": int(gap_tr),
        "gap_sec": float(gap_tr * tr),
        "tr_sec": float(tr),
        "eeg_window_sec": 16.0,
        "eeg_samples_per_window": crop,
        "eeg_fs_hz": 200,
    }
    return (eeg_train, fmri_train, eeg_val, fmri_val, eeg_test, fmri_test), split_info
