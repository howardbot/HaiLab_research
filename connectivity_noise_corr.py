#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
connectivity_noise_corr.py
Noise correlation during stimulus window:
1) Build neurons×trials firing-rate matrix aligned to STIM_ON (118) within WIN.
2) Build robust string condition keys (e.g., "sl=45|tilt=90|depth=57").
3) Demean per condition → residuals → Pearson correlation (noise corr).
4) Save heatmap PNG and matrix NPY per session.

You can change WIN to analyze Pre/Post windows as needed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.Loader import load_mat_session

# -------- Config --------
STIM_ON_ID = 118
WIN = (0.0, 0.5)                 # during-stim window relative to STIM_ON
OUTDIR = "connectivity_noise_corr"
MIN_TRIALS_FOR_CORR = 5          # need at least this many valid trial columns to compute corr
os.makedirs(OUTDIR, exist_ok=True)


# ---------- Utils ----------
def _eids_to_int(eids):
    """Convert EID array (float/str/int/object) → int ndarray."""
    e = np.atleast_1d(eids)
    try:
        return np.rint(e.astype(float)).astype(int)
    except Exception:
        return np.array([int(str(x)) for x in e], dtype=int)


def get_event_time(trial, eid_or_eids):
    """Return earliest time of any candidate EIDs; None if not found."""
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids = _eids_to_int(trial.EID)
    times = np.atleast_1d(trial.EventT)
    cand = np.atleast_1d(eid_or_eids)
    idx = np.where(np.isin(eids, cand))[0]
    if idx.size == 0:
        return None
    i = idx[np.argmin(times[idx])]
    return float(times[i])


def _iter_units(units):
    """Normalize mixed unit containers to a list of 1D arrays."""
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
            # each column = one unit (flip if your format differs)
            return [np.atleast_1d(arr[:, i]) for i in range(arr.shape[1])]
    return [np.atleast_1d(units)]


def build_neuron_map(trial):
    """List[(tt, unit_idx)] for TT1..TT8 present in the trial."""
    nm = []
    for tt in range(1, 9):
        f = f"UnitT_TT{tt}"
        if hasattr(trial, f):
            units = _iter_units(getattr(trial, f))
            for i in range(len(units)):
                nm.append((tt, i))
    return nm


def extract_spike_times(trial, neuron_map):
    """Return list of spike arrays per neuron (same order as neuron_map)."""
    out = []
    for tt, unit in neuron_map:
        f = f"UnitT_TT{tt}"
        if not hasattr(trial, f):
            out.append([])
            continue
        units = _iter_units(getattr(trial, f))
        out.append(units[unit] if unit < len(units) else [])
    return out


def corrcoef_safe(mat, min_trials=5):
    """
    Pearson corr on (neurons × trials) matrix using only columns where all neurons are finite.
    Returns NaN matrix if not enough columns.
    """
    if mat.size == 0:
        return np.full((0, 0), np.nan)
    col_mask = np.all(np.isfinite(mat), axis=0)
    X = mat[:, col_mask]
    if X.shape[1] < min_trials:
        return np.full((mat.shape[0], mat.shape[0]), np.nan)
    return np.corrcoef(X)


# ---------- Condition key (Option A: unified string key) ----------
def _first_attr(trial, names):
    for nm in names:
        if hasattr(trial, nm):
            return getattr(trial, nm)
    return None


def get_condition_key_str(trial):
    """
    Build a stable string key like "sl=45|tilt=90|depth=57".
    - Missing fields are omitted.
    - Numbers formatted with %.6g to stabilize 29.999999 vs 30.
    - If no fields present: "all"
    """
    sl = _first_attr(trial, ['slant', 'Slant', 'SLANT'])
    tl = _first_attr(trial, ['tilt', 'Tilt', 'TILT'])
    dp = _first_attr(trial, ['depth', 'Depth', 'DEPTH'])

    parts = []
    if sl is not None:
        parts.append(f"sl={float(sl):.6g}")
    if tl is not None:
        parts.append(f"tilt={float(tl):.6g}")
    if dp is not None:
        parts.append(f"depth={float(dp):.6g}")
    return "|".join(parts) if parts else "all"


# ---------- Core ----------
def noise_corr_one_file(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    T = load_mat_session(filepath)
    if len(T) == 0:
        print(f"[WARN] Empty session: {name}")
        return

    neuron_map = build_neuron_map(T[0])
    if len(neuron_map) < 2:
        print(f"[WARN] <2 neurons: {name}")
        return

    all_spikes, align_times, cond_keys = [], [], []
    # (optional) debug counts
    c118 = 0

    for tr in T:
        on = get_event_time(tr, STIM_ON_ID)
        if hasattr(tr, 'EID'):
            c118 += int(np.sum(_eids_to_int(tr.EID) == STIM_ON_ID))
        if on is None:
            continue
        all_spikes.append(extract_spike_times(tr, neuron_map))
        align_times.append(on)
        cond_keys.append(get_condition_key_str(tr))

    print(f"[DEBUG] {name}: EID118 count in trials (raw) = {c118}")

    if len(all_spikes) == 0:
        print(f"[WARN] No valid trials after filtering: {name}")
        return

    all_spikes = np.array(all_spikes, dtype=object)
    align_times = np.array(align_times, float)
    cond_keys = np.array(cond_keys, dtype=str)  # <-- all strings (fixes str/float compare)

    # Build rate matrix (N×T)
    N, Tn = len(neuron_map), len(all_spikes)
    dur = WIN[1] - WIN[0]
    rate = np.full((N, Tn), np.nan, float)
    for t in range(Tn):
        at = align_times[t]
        for n in range(N):
            s = np.asarray(all_spikes[t][n], float)
            if s.size:
                rel = s - at
                m = (rel >= WIN[0]) & (rel <= WIN[1])
                rate[n, t] = np.count_nonzero(m) / dur

    # Demean per condition → residuals
    R = rate.copy()
    uniq = np.unique(cond_keys)  # all strings → safe
    for u in uniq:
        cols = (cond_keys == u)
        cnt = int(np.sum(cols))
        # Helpful log
        print(f"[COND] {name}: '{u}' -> {cnt} trials")
        if cnt < 2:
            # too few trials in this condition; skip demeaning
            continue
        mu = np.nanmean(rate[:, cols], axis=1, keepdims=True)
        R[:, cols] = rate[:, cols] - mu

    # Noise correlation
    noise_corr = corrcoef_safe(R, min_trials=MIN_TRIALS_FOR_CORR)
    # Build labels: "TTx_Uy"
    labels = [f"TT{tt}_U{ui}" for (tt, ui) in neuron_map]
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(noise_corr,
                cmap='coolwarm', vmin=-1, vmax=1, square=True,
                xticklabels=labels, yticklabels=labels, cbar=True)
    plt.title(f"Noise Correlation (During {WIN[0]}–{WIN[1]}s @118) — {name}")
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"noise_corr_{name}.png"), dpi=150)
    plt.close()

    # Save matrix
    np.save(os.path.join(OUTDIR, f"noise_corr_{name}.npy"), noise_corr)
    print(f"[OK] {name}: noise-corr saved to {OUTDIR}")


def main():
    data_dir = "./data"
    files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    if not files:
        print("[WARN] No .mat files found in ./data")
    for f in files:
        noise_corr_one_file(os.path.join(data_dir, f))


if __name__ == "__main__":
    main()
