# src/Features.py
import numpy as np

def extract_firing_rate(T, label_type='slant', window=(0, 0.2)):
    X = []
    y = []

    # build map from the first trial
    trial0 = T[0]
    neuron_map = []
    for tt in range(1, 9):
        tt_field = f'UnitT_TT{tt}'
        if hasattr(trial0, tt_field):
            tt_units = getattr(trial0, tt_field)
            if not isinstance(tt_units, (list, tuple, np.ndarray)):
                tt_units = [tt_units]
            for unit_idx in range(len(tt_units)):
                neuron_map.append((tt, unit_idx))

    for i, trial in enumerate(T):
        stim_onset = None
        if hasattr(trial, 'EID') and hasattr(trial, 'EventT'):
            eid_list = np.atleast_1d(trial.EID)
            stim_indices = np.where(eid_list == 118)[0]
            if len(stim_indices) > 0:
                stim_onset = trial.EventT[stim_indices[0]]

        if stim_onset is None:
            continue  # skip no stimuli trial

        features = []
        for tt, unit_idx in neuron_map:
            tt_field = f'UnitT_TT{tt}'
            if not hasattr(trial, tt_field):
                features.append(0)
                continue

            units = getattr(trial, tt_field)
            if not isinstance(units, (list, tuple, np.ndarray)):
                units = [units]

            if unit_idx >= len(units):
                features.append(0)
                continue

            unit_spikes = units[unit_idx]
            if unit_spikes is None:
                features.append(0)
                continue

            unit_spikes = np.atleast_1d(unit_spikes)
            aligned_spikes = unit_spikes - stim_onset
            in_window = (aligned_spikes >= window[0]) & (aligned_spikes <= window[1])
            firing_rate = np.sum(in_window) / (window[1] - window[0])
            features.append(firing_rate)

        # make a warning and skip inconsistent trial
        if len(X) > 0 and len(features) != len(X[0]):
            print(f"[⚠️ Warning] Trial {i} has feature length {len(features)}, expected {len(X[0])}. Trial skipped.")
            continue

        label = getattr(trial, label_type) if hasattr(trial, label_type) else None
        if label is None:
            continue

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)