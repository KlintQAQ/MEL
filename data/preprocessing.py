import mne
import numpy as np
import pandas as pd

def normalize_data(data, means_stds=None, mode='absmax'):

    if mode == 'absmax':
        means = np.mean(data, axis=-1, keepdims=True)
        data = data - means
        transform_data = data / (
                np.quantile(np.abs(data), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8)
        return transform_data, means_stds

    if means_stds is None:
        means = np.mean(data, axis=-1, keepdims=True)
        stds = np.std(data, axis=-1, keepdims=True)
        means_stds = (means, stds)

    transform_data = (data - means_stds[0]) / means_stds[1]

    return transform_data, means_stds


def epoching_seq2one(eegraw, fmridataset,
                     tmin, tmax, event_name, ifnorm=0, crop=0,
                     dropfmriframes=0):


    events, event_id = mne.events_from_annotations(eegraw)
    sync_eventname = event_name
    event_id = {sync_eventname: event_id[sync_eventname]}


    events = mne.pick_events(events, include=event_id[sync_eventname])
    if dropfmriframes != 0:
        events = events[dropfmriframes:]

    reject_criteria = None

    eeg_epoch = mne.Epochs(eegraw, events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True,
                           baseline=(None, None), reject=reject_criteria)

    eeg_epoch_data = eeg_epoch.get_data(units='uV')

    select_ind = eeg_epoch.selection
    fmri_epoch = fmridataset[:, select_ind]

    eeg_epoch_sample = [np.array(sample) for sample in eeg_epoch_data]
    fmri_sample = [np.array(sample) for sample in fmri_epoch.T]

    if crop > 0:
        eeg_epoch_sample = [sample[:, :crop] for sample in eeg_epoch_sample]
    data = {
        "eeg": eeg_epoch_sample,
        "fmri": fmri_sample
    }

    return data, eeg_epoch
