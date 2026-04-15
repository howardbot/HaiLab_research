#!/usr/bin/env python3
"""
01_prepare_spike_counts.py
===========================
Standalone data-preparation script for the conv-model ML pipeline.

Reads raw .mat sessions → extracts spike times → builds spike-count
trains (N_neurons × T_trials × K_bins) → saves as .npy files.

**Does NOT touch any existing scripts.**

Output structure
----------------
    {OUTPUT_DIR}/
        {session}/
            neuron_map_valid.csv
            PRE_stimOnAnchor/
                counts_{bin_tag}/
                    fixed_window/
                        spike_counts.npy    # (N, T_valid, K) int16
                        meta.json
            ON_stimOnAnchor/
                counts_{bin_tag}/
                    outer_0000/
                        spike_counts.npy
                        meta.json
                    outer_0001/
                        ...
                outer_index.csv
            POST_stimOffAnchor/
                counts_{bin_tag}/
                    fixed_window/
                        spike_counts.npy
                        meta.json

Usage
-----
    python 01_prepare_spike_counts.py
"""

import os
import re
import csv
import json
import sys
import numpy as np

# ---- import the shared Loader ----
# Loader lives at ../../src/Loader.py relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.Loader import load_mat_session


# =============================================================
# CONFIG
# =============================================================

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "out_spike_counts")

# Event IDs
STIM_ON_ID  = 118
STIM_OFF_IDS = [120]

# -------- spike-count bin resolutions --------
# Coarser bins than binary → counts are more informative
BIN_RESOLUTION_LIST_SEC = [
    0.002,    # 2 ms
    0.005,    # 5 ms  (recommended default for Poisson model)
    0.010,    # 10 ms
]

# -------- outer sliding windows (ON epoch) --------
OUTER_WIN_SEC  = 0.200    # 200 ms
OUTER_STEP_SEC = 0.010    # 10 ms

# -------- analysis ranges (relative to anchor) --------
PRE_RANGE_SEC  = (-0.600, 0.000)
ON_RANGE_SEC   = ( 0.000, 1.000)
POST_RANGE_SEC = ( 0.000, 0.200)

# -------- filters --------
MIN_TRIALS   = 10
MIN_RATE_HZ  = 1.0

EPS = 1e-12


# =============================================================
# UTILS
# =============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_name(s: str) -> str:
    s = os.path.splitext(os.path.basename(s))[0]
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s[:180]


def fmt_ms(x_sec: float) -> str:
    return f"{x_sec * 1000.0:.3f}ms".replace(".", "p")


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# =============================================================
# EVENT HELPERS  (same logic as the original build script)
# =============================================================

def get_event_time_first(trial, eid):
    if not hasattr(trial, "EID") or not hasattr(trial, "EventT"):
        return None
    eids = np.atleast_1d(trial.EID)
    ts   = np.atleast_1d(trial.EventT)
    idx  = np.where(eids == eid)[0]
    return float(ts[idx[0]]) if len(idx) else None


def get_event_time_first_of_many(trial, eids_list):
    for eid in eids_list:
        t = get_event_time_first(trial, eid)
        if t is not None:
            return t, eid
    return None, None


# =============================================================
# NEURON MAP / SPIKES
# =============================================================

def build_neuron_map(trial):
    neuron_map = []
    for tt in range(1, 9):
        field = f"UnitT_TT{tt}"
        if not hasattr(trial, field):
            continue
        units = getattr(trial, field)
        if isinstance(units, np.ndarray) and units.dtype != object:
            neuron_map.append((tt, 0))
        elif isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, "__len__"):
            for i in range(len(units)):
                neuron_map.append((tt, i))
        else:
            neuron_map.append((tt, 0))
    return neuron_map


def extract_spike_times(trial, neuron_map):
    spikes = []
    for tt, unit in neuron_map:
        field = f"UnitT_TT{tt}"
        if not hasattr(trial, field):
            spikes.append(np.array([], dtype=float))
            continue
        units = getattr(trial, field)
        if isinstance(units, np.ndarray) and units.dtype != object:
            spikes.append(np.asarray(units, dtype=float))
        elif isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, "__len__"):
            if unit < len(units):
                spikes.append(np.atleast_1d(units[unit]).astype(float))
            else:
                spikes.append(np.array([], dtype=float))
        else:
            if unit == 0:
                spikes.append(np.atleast_1d(units).astype(float))
            else:
                spikes.append(np.array([], dtype=float))
    return spikes


# =============================================================
# TRIAL COVERAGE
# =============================================================

def infer_trial_time_bounds_robust(trial, neuron_map=None):
    candidate_pairs = [
        ("TrialStartT", "TrialEndT"),
        ("t_start", "t_end"),
        ("StartT", "EndT"),
        ("start_time", "end_time"),
    ]
    for a, b in candidate_pairs:
        if hasattr(trial, a) and hasattr(trial, b):
            try:
                t0 = float(np.atleast_1d(getattr(trial, a))[0])
                t1 = float(np.atleast_1d(getattr(trial, b))[0])
                if np.isfinite(t0) and np.isfinite(t1) and (t1 > t0):
                    return t0, t1
            except Exception:
                pass

    tmin_ev, tmax_ev = None, None
    if hasattr(trial, "EventT"):
        ev = np.atleast_1d(trial.EventT).astype(float)
        ev = ev[np.isfinite(ev)]
        if ev.size:
            tmin_ev, tmax_ev = float(np.min(ev)), float(np.max(ev))

    if tmin_ev is None:
        if neuron_map is None:
            return None, None
        sp_list = extract_spike_times(trial, neuron_map)
        all_sp = [sp[np.isfinite(sp)] for sp in
                  (np.atleast_1d(s).astype(float) for s in sp_list)
                  if sp.size]
        if not all_sp:
            return None, None
        all_sp = np.concatenate(all_sp)
        return float(np.min(all_sp)), float(np.max(all_sp))

    t_min, t_max = tmin_ev, tmax_ev
    if neuron_map is not None:
        margin = 1.0
        sp_list = extract_spike_times(trial, neuron_map)
        for sp in sp_list:
            sp = np.atleast_1d(sp).astype(float)
            sp = sp[np.isfinite(sp)]
            sp = sp[(sp >= tmin_ev - margin) & (sp <= tmax_ev + margin)]
            if sp.size:
                t_min = min(t_min, float(np.min(sp)))
                t_max = max(t_max, float(np.max(sp)))
    return t_min, t_max


def build_coverage_mask(anchors, trial_bounds_abs, rel_window):
    mask = np.zeros(len(anchors), dtype=bool)
    for i, anchor_t in enumerate(anchors):
        t_min, t_max = trial_bounds_abs[i]
        if anchor_t is None or t_min is None or t_max is None:
            continue
        w0 = anchor_t + rel_window[0]
        w1 = anchor_t + rel_window[1]
        mask[i] = (w0 >= t_min - EPS) and (w1 <= t_max + EPS)
    return mask


# =============================================================
# FILTER
# =============================================================

def preselect_neurons_by_rate(all_spikes_trials, anchors, rel_window,
                              valid_trial_mask, min_rate_hz):
    valid_idx = np.where(valid_trial_mask)[0]
    if len(valid_idx) == 0:
        return []
    N = len(all_spikes_trials[0])
    w0, w1 = rel_window
    dur = w1 - w0
    total_time = len(valid_idx) * dur
    mean_rates = np.zeros(N, dtype=float)
    for i in range(N):
        total_sp = 0
        for tr in valid_idx:
            sp = np.asarray(all_spikes_trials[tr][i], dtype=float) - anchors[tr]
            total_sp += np.sum((sp >= w0) & (sp < w1))
        mean_rates[i] = total_sp / max(EPS, total_time)
    return [i for i in range(N) if mean_rates[i] >= min_rate_hz]


# =============================================================
# SPIKE COUNT TRAINS  (core difference from binary)
# =============================================================

def build_spike_count_trains(all_spikes_trials, anchors, rel_window,
                             bin_res_sec, neuron_idx, valid_trial_mask):
    """
    Build spike-count trains in rel_window relative to anchor.

    Unlike binary trains (clipped to 0/1), this preserves the actual
    spike count per bin — essential for Poisson models.

    Parameters
    ----------
    all_spikes_trials : list[list[ndarray]]
        Outer: trials, inner: neurons, each a 1-D array of spike times (sec).
    anchors : ndarray (n_trials,)
        Anchor times (e.g. stim-on) per trial.
    rel_window : (float, float)
        Analysis window relative to anchor, in seconds.
    bin_res_sec : float
        Bin width in seconds.
    neuron_idx : list[int]
        Which neurons to include (indices into each trial's spike list).
    valid_trial_mask : ndarray bool
        Which trials to include.

    Returns
    -------
    counts : ndarray (N, T_valid, K)  int16
        Spike counts per neuron × trial × time bin.
    bin_edges_sec : ndarray (K,)
        Left edges of each bin (relative to anchor).
    valid_trial_ids : ndarray int
        Original trial indices that were kept.
    """
    w0, w1 = rel_window
    dur = w1 - w0
    if dur <= 0:
        empty = np.zeros((len(neuron_idx), 0, 0), dtype=np.int16)
        return empty, np.array([], dtype=float), np.array([], dtype=int)

    K = int(np.floor(dur / bin_res_sec))
    if K < 2:
        empty = np.zeros((len(neuron_idx), 0, 0), dtype=np.int16)
        return empty, np.array([], dtype=float), np.array([], dtype=int)

    bin_edges = w0 + np.arange(K) * bin_res_sec
    valid_trial_ids = np.where(valid_trial_mask)[0]

    N = len(neuron_idx)
    T_valid = len(valid_trial_ids)
    counts = np.zeros((N, T_valid, K), dtype=np.int16)

    for t_valid, tr in enumerate(valid_trial_ids):
        anchor = anchors[tr]
        for ni, i in enumerate(neuron_idx):
            sp_abs = np.asarray(all_spikes_trials[tr][i], dtype=float)
            if sp_abs.size == 0:
                continue
            sp = sp_abs - anchor
            sp = sp[(sp >= w0) & (sp < w1)]
            if sp.size == 0:
                continue
            idx = np.floor((sp - w0) / bin_res_sec).astype(int)
            idx = idx[(idx >= 0) & (idx < K)]
            # KEY DIFFERENCE: count spikes, not clip to 1
            np.add.at(counts[ni, t_valid], idx, 1)

    return counts, bin_edges, valid_trial_ids


# =============================================================
# SAVE HELPERS
# =============================================================

def save_spike_counts(out_dir, counts, bin_edges_sec, valid_trial_ids,
                      session, epoch_dir, bin_res_sec, rel_window,
                      neuron_idx, outer_idx=None):
    """Save spike counts + metadata JSON."""
    ensure_dir(out_dir)

    np.save(os.path.join(out_dir, "spike_counts.npy"), counts)
    np.save(os.path.join(out_dir, "bin_edges_sec.npy"),
            bin_edges_sec.astype(np.float32))
    np.save(os.path.join(out_dir, "valid_trial_ids.npy"),
            valid_trial_ids.astype(np.int32))

    N, T, K = counts.shape
    total_spikes = int(np.sum(counts))
    nonzero_frac = float(np.mean(counts > 0))

    meta = {
        "session": session,
        "epoch_dir": epoch_dir,
        "bin_res_sec": bin_res_sec,
        "bin_res_ms": bin_res_sec * 1000.0,
        "rel_window_sec": list(rel_window),
        "outer_idx": outer_idx,
        "N_neurons": N,
        "T_trials": T,
        "K_bins": K,
        "neuron_idx": neuron_idx,
        "total_spikes": total_spikes,
        "nonzero_bin_frac": round(nonzero_frac, 6),
        "mean_count_per_bin": round(float(np.mean(counts)), 6),
        "max_count_per_bin": int(np.max(counts)) if counts.size else 0,
    }

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# =============================================================
# EPOCH ANALYSIS
# =============================================================

def analyze_fixed_epoch(session_root, session_name, epoch_dir,
                        all_spikes_trials, anchors, rel_window,
                        valid_trial_mask, valid_neurons):
    """Process a fixed-window epoch (PRE or POST)."""
    n_trials = int(np.sum(valid_trial_mask))
    if n_trials < MIN_TRIALS:
        print(f"    [SKIP] {epoch_dir}: too few covered trials ({n_trials})")
        return

    for bin_res_sec in BIN_RESOLUTION_LIST_SEC:
        res_tag = f"counts_{fmt_ms(bin_res_sec)}"
        out_dir = os.path.join(session_root, epoch_dir, res_tag, "fixed_window")

        counts, bin_edges, valid_ids = build_spike_count_trains(
            all_spikes_trials=all_spikes_trials,
            anchors=anchors,
            rel_window=rel_window,
            bin_res_sec=bin_res_sec,
            neuron_idx=valid_neurons,
            valid_trial_mask=valid_trial_mask,
        )

        if counts.shape[1] < MIN_TRIALS or counts.shape[2] < 2:
            print(f"    [SKIP] {epoch_dir}/{res_tag}: insufficient data")
            continue

        save_spike_counts(
            out_dir, counts, bin_edges, valid_ids,
            session=session_name, epoch_dir=epoch_dir,
            bin_res_sec=bin_res_sec, rel_window=rel_window,
            neuron_idx=valid_neurons,
        )

    print(f"    [OK] {epoch_dir}: {rel_window[0]:.3f}..{rel_window[1]:.3f}s "
          f"| trials={n_trials}")


def analyze_on_epoch(session_root, session_name, all_spikes_trials,
                     stim_on_anchors, on_mask, valid_neurons):
    """Process ON epoch with sliding outer windows."""
    epoch_dir = "ON_stimOnAnchor"
    epoch_root = os.path.join(session_root, epoch_dir)
    ensure_dir(epoch_root)

    t_start, t_end = ON_RANGE_SEC
    last_start = t_end - OUTER_WIN_SEC + 1e-12

    if last_start < t_start:
        print("    [SKIP] ON range shorter than outer window")
        return

    n_on = int(np.sum(on_mask))
    if n_on < MIN_TRIALS:
        print(f"    [SKIP] ON: too few covered trials ({n_on})")
        return

    outer_rows = []
    outer_idx = 0
    t0 = t_start

    while t0 <= last_start:
        rel_window = (t0, t0 + OUTER_WIN_SEC)

        for bin_res_sec in BIN_RESOLUTION_LIST_SEC:
            res_tag = f"counts_{fmt_ms(bin_res_sec)}"
            out_dir = os.path.join(
                epoch_root, res_tag, f"outer_{outer_idx:04d}"
            )

            counts, bin_edges, valid_ids = build_spike_count_trains(
                all_spikes_trials=all_spikes_trials,
                anchors=stim_on_anchors,
                rel_window=rel_window,
                bin_res_sec=bin_res_sec,
                neuron_idx=valid_neurons,
                valid_trial_mask=on_mask,
            )

            if counts.shape[1] < MIN_TRIALS or counts.shape[2] < 2:
                continue

            save_spike_counts(
                out_dir, counts, bin_edges, valid_ids,
                session=session_name, epoch_dir=epoch_dir,
                bin_res_sec=bin_res_sec, rel_window=rel_window,
                neuron_idx=valid_neurons, outer_idx=outer_idx,
            )

        outer_rows.append({
            "outer_idx": outer_idx,
            "t0_sec": t0,
            "t1_sec": t0 + OUTER_WIN_SEC,
            "W_out_sec": OUTER_WIN_SEC,
            "step_sec": OUTER_STEP_SEC,
            "n_trials": n_on,
        })

        outer_idx += 1
        t0 += OUTER_STEP_SEC

    write_csv(
        os.path.join(epoch_root, "outer_index.csv"),
        outer_rows,
        ["outer_idx", "t0_sec", "t1_sec", "W_out_sec", "step_sec", "n_trials"],
    )

    print(f"    [OK] ON: {outer_idx} outer windows | trials={n_on}")


# =============================================================
# SESSION
# =============================================================

def analyze_one_session(mat_path):
    session_name = sanitize_name(mat_path)
    print(f"\n[SESSION] {session_name}")

    T = load_mat_session(mat_path)
    if len(T) == 0:
        print("  [SKIP] empty session")
        return

    neuron_map = build_neuron_map(T[0])
    if len(neuron_map) == 0:
        print("  [SKIP] no UnitT_TT* found")
        return

    # ---- extract events & spikes from all trials ----
    all_spikes_trials = []
    stim_on_anchors   = []
    stim_off_anchors  = []
    trial_bounds_abs  = []

    for trial in T:
        stim_on = get_event_time_first(trial, STIM_ON_ID)
        if stim_on is None:
            continue
        stim_off, _ = get_event_time_first_of_many(trial, STIM_OFF_IDS)

        all_spikes_trials.append(extract_spike_times(trial, neuron_map))
        stim_on_anchors.append(stim_on)
        stim_off_anchors.append(np.nan if stim_off is None else stim_off)
        trial_bounds_abs.append(
            infer_trial_time_bounds_robust(trial, neuron_map=neuron_map)
        )

    if len(all_spikes_trials) < MIN_TRIALS:
        print(f"  [SKIP] too few trials with StimOn: {len(all_spikes_trials)}")
        return

    stim_on_anchors  = np.asarray(stim_on_anchors,  dtype=float)
    stim_off_anchors = np.asarray(stim_off_anchors, dtype=float)
    trial_bounds_abs = np.asarray(trial_bounds_abs, dtype=object)

    # ---- coverage masks ----
    pre_mask  = build_coverage_mask(stim_on_anchors,  trial_bounds_abs, PRE_RANGE_SEC)
    on_mask   = build_coverage_mask(stim_on_anchors,  trial_bounds_abs, ON_RANGE_SEC)
    post_mask = build_coverage_mask(stim_off_anchors, trial_bounds_abs, POST_RANGE_SEC)

    print(f"  covered trials | "
          f"PRE={int(np.sum(pre_mask))} "
          f"ON={int(np.sum(on_mask))} "
          f"POST={int(np.sum(post_mask))}")

    if int(np.sum(on_mask)) < MIN_TRIALS:
        print("  [SKIP] too few ON-covered trials for neuron filtering")
        return

    # ---- neuron selection (by firing rate) ----
    valid_neurons = preselect_neurons_by_rate(
        all_spikes_trials=all_spikes_trials,
        anchors=stim_on_anchors,
        rel_window=ON_RANGE_SEC,
        valid_trial_mask=on_mask,
        min_rate_hz=MIN_RATE_HZ,
    )

    if len(valid_neurons) < 2:
        print("  [SKIP] too few neurons after rate filtering")
        return

    print(f"  valid neurons: {len(valid_neurons)} / {len(neuron_map)}")

    session_root = os.path.join(OUTPUT_DIR, session_name)
    ensure_dir(session_root)

    # ---- save neuron map ----
    rows = []
    for k, orig_idx in enumerate(valid_neurons):
        tt, unit = neuron_map[orig_idx]
        rows.append({
            "k_in_valid": k,
            "orig_neuron_idx": orig_idx,
            "tt": tt,
            "unit": unit,
            "label": f"n{orig_idx}_TT{tt}u{unit}",
        })
    write_csv(
        os.path.join(session_root, "neuron_map_valid.csv"),
        rows,
        ["k_in_valid", "orig_neuron_idx", "tt", "unit", "label"],
    )

    # ---- run all epochs ----
    analyze_fixed_epoch(
        session_root, session_name, "PRE_stimOnAnchor",
        all_spikes_trials, stim_on_anchors, PRE_RANGE_SEC,
        pre_mask, valid_neurons,
    )

    analyze_on_epoch(
        session_root, session_name, all_spikes_trials,
        stim_on_anchors, on_mask, valid_neurons,
    )

    analyze_fixed_epoch(
        session_root, session_name, "POST_stimOffAnchor",
        all_spikes_trials, stim_off_anchors, POST_RANGE_SEC,
        post_mask, valid_neurons,
    )


# =============================================================
# MAIN
# =============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    mats = [f for f in os.listdir(DATA_DIR) if f.endswith(".mat")]
    if not mats:
        print(f"[WARN] no .mat files found in {DATA_DIR}")
        return

    print(f"[INFO] found {len(mats)} sessions in {DATA_DIR}")
    print(f"[INFO] bin resolutions: {[b*1000 for b in BIN_RESOLUTION_LIST_SEC]} ms")
    print(f"[INFO] output → {OUTPUT_DIR}")

    for fname in sorted(mats):
        analyze_one_session(os.path.join(DATA_DIR, fname))

    print("\n[DONE]")


if __name__ == "__main__":
    main()
