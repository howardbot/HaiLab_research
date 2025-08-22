#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverage-aware export of per-neuron spike counts and firing rates (Hz) for Pre/During/Post phases.

Rules (effective coverage):
- Pre:    window = [StimOn-0.5, StimOn], clip to trial bounds.
- During: window = [StimOn, min(StimOn+0.5, ChoiceOn if present)], clip to trial bounds.
- Post:   window = [ChoiceOn, ChoiceOn+0.5], clip to trial bounds. (requires ChoiceOn)
Average rate (Hz) = total_spikes / total_effective_seconds.
A trial contributes to a phase iff the effective window length > 0.

Outputs: ./spike_counts_and_rates_by_phase_coverage.csv
"""

import os
import numpy as np
import pandas as pd
from src.Loader import load_mat_session

# Event IDs
TRIAL_START_ID = 111
TRIAL_END_ID = 112
STIM_ON_ID = 118
SACCADE_CHOICE_ON_ID = 120

def _iter_units(units):
    """Normalize into list of 1D np.ndarrays of spike times."""
    if units is None:
        return []
    if isinstance(units, (list, tuple)):
        return [np.atleast_1d(u) for u in units]
    if isinstance(units, np.ndarray):
        if units.dtype == object:
            return [np.atleast_1d(u) for u in units]
        arr = np.atleast_1d(units)
        if arr.ndim == 1:
            return [arr]
        if arr.ndim == 2:
            return [np.atleast_1d(arr[:, i]) for i in range(arr.shape[1])]
    return [np.atleast_1d(units)]

def build_neuron_map(trial):
    """Return list of (tt, unit_idx) from first trial."""
    neuron_map = []
    for tt in range(1, 9):
        field = f"UnitT_TT{tt}"
        if hasattr(trial, field):
            units = _iter_units(getattr(trial, field))
            for i in range(len(units)):
                neuron_map.append((tt, i))
    return neuron_map

def extract_spike_times(trial, neuron_map):
    """Return absolute spike times per neuron for this trial following neuron_map layout."""
    spikes = []
    for tt, unit in neuron_map:
        field = f"UnitT_TT{tt}"
        if not hasattr(trial, field):
            spikes.append([])
            continue
        units = _iter_units(getattr(trial, field))
        spikes.append(units[unit] if unit < len(units) else [])
    return spikes

def _eids_to_int(eids):
    eids = np.atleast_1d(eids)
    try:
        return np.rint(eids.astype(float)).astype(int)
    except Exception:
        return np.array([int(str(x)) for x in eids], dtype=int)

def _event_time(trial, eid):
    """Return earliest timestamp for a given EID or None."""
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids = _eids_to_int(trial.EID)
    ts = np.atleast_1d(trial.EventT)
    if ts.size != eids.size or ts.size == 0:
        return None
    idxs = np.where(eids == int(eid))[0]
    if idxs.size == 0:
        return None
    i = idxs[np.argmin(ts[idxs])]
    return float(ts[i])

def _trial_bounds(trial):
    """Return (t_start, t_end); fallback to (min_ts, max_ts) if 111/112 absent."""
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None, None
    eids = _eids_to_int(trial.EID)
    ts = np.atleast_1d(trial.EventT)
    if ts.size != eids.size or ts.size == 0:
        return None, None
    tmin, tmax = float(np.min(ts)), float(np.max(ts))
    t_start = _event_time(trial, TRIAL_START_ID)
    t_end   = _event_time(trial, TRIAL_END_ID)
    if t_start is None: t_start = tmin
    if t_end   is None: t_end   = tmax
    return t_start, t_end

def _count_in_interval(spike_abs, t0, t1):
    if spike_abs.size == 0 or (t1 <= t0):
        return 0
    return int(np.sum((spike_abs >= t0) & (spike_abs <= t1)))

def process_session(T, fname):
    if len(T) == 0:
        return []

    neuron_map = build_neuron_map(T[0])
    if len(neuron_map) == 0:
        return []

    rows = []

    # accumulators per neuron per phase
    pre_spikes = np.zeros(len(neuron_map), dtype=int)
    dur_spikes = np.zeros(len(neuron_map), dtype=int)
    post_spikes = np.zeros(len(neuron_map), dtype=int)

    pre_sec = 0.0
    dur_sec = 0.0
    post_sec = 0.0

    pre_trials = 0
    dur_trials = 0
    post_trials = 0

    for trial in T:
        stim_t = _event_time(trial, STIM_ON_ID)
        choice_t = _event_time(trial, SACCADE_CHOICE_ON_ID)
        if stim_t is None:
            # Without 118, neither Pre nor During can be defined
            pass
        t_start, t_end = _trial_bounds(trial)
        if t_start is None or t_end is None:
            # Fallback: assume broad coverage; but still respect event existence
            t_start, t_end = -np.inf, np.inf

        # ----- Pre: [stim-0.5, stim], requires stim_t -----
        if stim_t is not None:
            pre_t0 = max(stim_t - 0.5, t_start)
            pre_t1 = min(stim_t, t_end)
            eff_pre = pre_t1 - pre_t0
            if eff_pre > 0:
                spikes = extract_spike_times(trial, neuron_map)
                for n_idx in range(len(neuron_map)):
                    s_abs = np.asarray(spikes[n_idx], dtype=float)
                    pre_spikes[n_idx] += _count_in_interval(s_abs, pre_t0, pre_t1)
                pre_sec += eff_pre
                pre_trials += 1  # contributes

        # ----- During: [stim, min(stim+0.5, choice if present)] -----
        if stim_t is not None:
            dur_t1_nominal = stim_t + 0.5
            if choice_t is not None:
                dur_t1_nominal = min(dur_t1_nominal, choice_t)
            dur_t0 = max(stim_t, t_start)
            dur_t1 = min(dur_t1_nominal, t_end)
            eff_dur = dur_t1 - dur_t0
            if eff_dur > 0:
                spikes = extract_spike_times(trial, neuron_map)
                for n_idx in range(len(neuron_map)):
                    s_abs = np.asarray(spikes[n_idx], dtype=float)
                    dur_spikes[n_idx] += _count_in_interval(s_abs, dur_t0, dur_t1)
                dur_sec += eff_dur
                dur_trials += 1

        # ----- Post: [choice, choice+0.5], requires choice_t -----
        if choice_t is not None:
            post_t0 = max(choice_t, t_start)
            post_t1 = min(choice_t + 0.5, t_end)
            eff_post = post_t1 - post_t0
            if eff_post > 0:
                spikes = extract_spike_times(trial, neuron_map)
                for n_idx in range(len(neuron_map)):
                    s_abs = np.asarray(spikes[n_idx], dtype=float)
                    post_spikes[n_idx] += _count_in_interval(s_abs, post_t0, post_t1)
                post_sec += eff_post
                post_trials += 1

    def safe_rate(total_spk, total_sec):
        return (total_spk / total_sec) if total_sec > 0 else np.nan

    for n_idx, (tt, unit_idx) in enumerate(neuron_map):
        rows.append({
            "file": fname,
            "neuron_index": n_idx,
            "TT": tt,
            "unit_idx": unit_idx,
            "trials_pre": pre_trials,
            "spikes_pre": int(pre_spikes[n_idx]),
            "covered_sec_pre": pre_sec,
            "rate_pre_hz": safe_rate(pre_spikes[n_idx], pre_sec),
            "trials_during": dur_trials,
            "spikes_during": int(dur_spikes[n_idx]),
            "covered_sec_during": dur_sec,
            "rate_during_hz": safe_rate(dur_spikes[n_idx], dur_sec),
            "trials_post": post_trials,
            "spikes_post": int(post_spikes[n_idx]),
            "covered_sec_post": post_sec,
            "rate_post_hz": safe_rate(post_spikes[n_idx], post_sec),
        })
    return rows

def main():
    data_dir = "./data"
    out_csv = "./spike_counts_and_rates_by_phase_coverage.csv"
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".mat")])
    all_rows = []
    for fname in files:
        T = load_mat_session(os.path.join(data_dir, fname))
        all_rows.extend(process_session(T, fname))
    if not all_rows:
        print("No data to export.")
        return
    df = pd.DataFrame(all_rows, columns=[
        "file","neuron_index","TT","unit_idx",
        "trials_pre","spikes_pre","covered_sec_pre","rate_pre_hz",
        "trials_during","spikes_during","covered_sec_during","rate_during_hz",
        "trials_post","spikes_post","covered_sec_post","rate_post_hz"
    ])
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()
