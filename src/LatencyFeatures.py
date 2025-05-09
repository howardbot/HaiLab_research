# src/LatencyFeatures.py
import numpy as np

def extract_latency_matrix(T, window=(0, 0.4), max_latency=np.inf):
    """
    Returns latency matrix: trials × neurons
    Each entry is the first spike time (after stimulus onset) for that neuron in that trial
    """
    trial0 = T[0]
    neuron_map = []
    for tt in range(1, 9):
        field = f'UnitT_TT{tt}'
        if hasattr(trial0, field):
            units = getattr(trial0, field)
            if not isinstance(units, (list, tuple, np.ndarray)):
                units = [units]
            for idx in range(len(units)):
                neuron_map.append((tt, idx))

    latency_matrix = []

    for trial in T:
        # Get stim onset
        stim_time = None
        if hasattr(trial, 'EID') and hasattr(trial, 'EventT'):
            eid = np.atleast_1d(trial.EID)
            idx = np.where(eid == 118)[0]
            if len(idx) > 0:
                stim_time = trial.EventT[idx[0]]
        if stim_time is None:
            continue

        latencies = []
        for tt, unit_idx in neuron_map:
            spikes = []
            field = f'UnitT_TT{tt}'
            if hasattr(trial, field):
                units = getattr(trial, field)
                if not isinstance(units, (list, tuple, np.ndarray)):
                    units = [units]
                if unit_idx < len(units):
                    unit_spikes = np.atleast_1d(units[unit_idx])
                    if unit_spikes is not None:
                        rel_spikes = unit_spikes - stim_time
                        valid = rel_spikes[(rel_spikes >= window[0]) & (rel_spikes <= window[1])]
                        if len(valid) > 0:
                            latencies.append(np.min(valid))
                            continue
            latencies.append(np.nan)  # no spike = NaN
        latency_matrix.append(latencies)

    return np.array(latency_matrix)
