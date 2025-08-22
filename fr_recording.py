#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-phase Raster + Population Rate

Key choices:
- Pre/During are aligned to STIM_ON (118).
- Post is aligned to SACCADE_CHOICE_ON (120).
- Robust unit parsing across list/tuple/object-array/ndarray.
- Debug counters print how many times 120/130/131 appear in the session.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.Loader import load_mat_session

# -------- Event IDs --------
STIM_ON_ID = 118
SACCADE_CHOICE_ON_ID = 120          # Post alignment anchor
STIM_OFF_IDS = (130, 131)           # Only for diagnostics, not used for alignment

# =============== Robust unit parsing ===============
def _iter_units(units):
    """
    Normalize various storage formats into a list of 1D np.ndarrays (spike times):
      - list/tuple/object-array: multiple units -> list of arrays
      - numeric ndarray: 1D -> single unit; 2D -> each column is a unit
      - anything else -> treated as a single unit
    """
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
            # assume (time, unit) with units on columns
            return [np.atleast_1d(arr[:, i]) for i in range(arr.shape[1])]
    return [np.atleast_1d(units)]

def build_neuron_map(trial):
    """Create a [(tt, unit_idx), ...] map using the first trial's available TT blocks."""
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
    """Convert trial.EID to int array (tolerant to float/strings)."""
    eids = np.atleast_1d(eids)
    try:
        return np.rint(eids.astype(float)).astype(int)
    except Exception:
        return np.array([int(str(x)) for x in eids], dtype=int)

def get_first_event_time(trial, candidates):
    """
    Return the earliest timestamp among candidate EIDs for this trial; None if not present.
    """
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids_int = _eids_to_int(trial.EID)
    times = np.atleast_1d(trial.EventT)
    if times.size != eids_int.size:
        # Malformed data: lengths must match. Skip this trial.
        return None
    idxs = np.where(np.isin(eids_int, np.array(candidates, dtype=int)))[0]
    if idxs.size == 0:
        return None
    first_idx = idxs[np.argmin(times[idxs])]
    return float(times[first_idx])

# =============== Trial-wise aggregation (Hz per neuron) ===============
def compute_population_rate_nanaware(all_spikes, bin_size=0.01, window=(-0.5, 0.5)):
    """
    Inputs:
      all_spikes: list over trials, each is list over neurons -> 1D arrays of spike times (relative).
    Returns:
      time_axis: left edges of bins
      avg_rate:  average population rate per bin, in Hz per neuron
      count_trials: number of contributing trials per bin (constant across the window)
    """
    num_trials = len(all_spikes)
    num_neurons = len(all_spikes[0]) if num_trials > 0 else 0
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    num_bins = len(edges) - 1

    sum_rate = np.zeros(num_bins, dtype=float)
    total_trials = 0

    for t in range(num_trials):
        trial_hist = np.zeros(num_bins, dtype=float)
        for n in range(num_neurons):
            s = np.asarray(all_spikes[t][n], dtype=float)
            if s.size:
                s = s[(s >= window[0]) & (s <= window[1])]
                if s.size:
                    h, _ = np.histogram(s, bins=edges)  # counts per bin
                    trial_hist += h
        # Convert to Hz per neuron
        trial_rate = (trial_hist / max(num_neurons, 1)) / bin_size
        sum_rate += trial_rate
        total_trials += 1

    if total_trials == 0:
        return edges[:-1], np.full(num_bins, np.nan), np.zeros(num_bins, dtype=int)

    avg_rate = sum_rate / total_trials
    count_trials = np.full(num_bins, total_trials, dtype=int)
    return edges[:-1], avg_rate, count_trials

# =============== Plotting (pooled-by-neuron raster + population rate) ===============
def plot_raster_and_rate(all_spikes_abs, align_times, neuron_map, fname, label, window,
                         output_dir, bin_size=0.01):
    import matplotlib.gridspec as gridspec
    num_trials = len(all_spikes_abs)
    num_neurons = len(neuron_map)
    if num_trials == 0 or num_neurons == 0:
        return

    colors = plt.cm.tab20(np.linspace(0, 1, num_neurons))
    spacing, height = 1.2, 0.8
    ytick_positions = [i * spacing + 1 for i in range(num_neurons)]

    fig = plt.figure(figsize=(28, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax_raster = fig.add_subplot(gs[0])
    ax_rate = fig.add_subplot(gs[1])

    # Convert absolute to relative spike times with respect to alignment
    all_spikes_rel = []
    for trial_idx in range(num_trials):
        rel_trial = []
        for n_idx in range(num_neurons):
            abs_s = np.asarray(all_spikes_abs[trial_idx][n_idx], dtype=float)
            if abs_s.size:
                rel_s = abs_s - align_times[trial_idx]
                rel_s = rel_s[(rel_s >= window[0]) & (rel_s <= window[1])]
            else:
                rel_s = np.array([], dtype=float)
            rel_trial.append(rel_s)
        all_spikes_rel.append(rel_trial)

    # Top: pooled-by-neuron raster across trials
    for n_idx in range(num_neurons):
        color = colors[n_idx % len(colors)]
        center = ytick_positions[n_idx]
        spike_times = [t_s[n_idx] for t_s in all_spikes_rel]
        spike_flat = (np.concatenate([s for s in spike_times if s.size])
                      if any(s.size for s in spike_times) else np.array([]))
        if spike_flat.size:
            ax_raster.vlines(spike_flat, center - height/2, center + height/2,
                             color=color, linewidth=0.6)

    ax_raster.set_title(f"{label} Raster (trials pooled by neuron) — {fname}")
    ax_raster.set_ylabel("Neuron Index")
    ax_raster.set_yticks(ytick_positions)
    ax_raster.set_yticklabels([f"{i}" for i in range(num_neurons)])
    ax_raster.set_xlim(window)

    # Bottom: population rate + trial count (constant)
    time_axis, avg_rate, n_contrib = compute_population_rate_nanaware(
        all_spikes_rel, bin_size=bin_size, window=window
    )
    ax_rate.plot(time_axis, avg_rate, linewidth=1.6, color="black",
                 label=f"Avg rate (Hz / neuron), bin={bin_size*1000:.0f} ms")
    ax_rate.fill_between(time_axis, 0, np.nan_to_num(avg_rate, nan=0.0), alpha=0.25)
    ax_rate.set_ylabel("Rate (Hz / neuron)")
    ax_rate.set_xlabel(f"Time (s) from {label.lower()}")
    ax_rate.set_xlim(window)

    ax2 = ax_rate.twinx()
    ax2.plot(time_axis, n_contrib, alpha=0.6, linestyle='--', linewidth=1.0,
             label="Trials contributing (constant)")
    ax2.set_ylabel("Trials contributing")
    ax2.set_ylim(bottom=0)

    ax_rate.legend(loc="upper left")
    ax2.legend(loc="upper right")

    os.makedirs(output_dir, exist_ok=True)
    save_name = f"raster_rate_nanaware_{label.lower().replace(' ', '_')}_{os.path.splitext(fname)[0]}.png"
    plt.savefig(os.path.join(output_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close()

# =============== Optional: trial-by-trial raster for selected neurons ===============
def plot_trial_by_trial_raster(all_spikes_abs, align_times, neuron_indices,
                               fname, label, window, output_dir):
    """
    Helpful for showing colleagues that trial-wise sparsity matches the averaged Hz.
    Draws trial on the y-axis for each selected neuron.
    """
    num_trials = len(all_spikes_abs)
    for n_idx in neuron_indices:
        fig, ax = plt.subplots(figsize=(10, 6))
        for t in range(num_trials):
            s_abs = np.asarray(all_spikes_abs[t][n_idx], dtype=float)
            s_rel = s_abs - align_times[t]
            s_rel = s_rel[(s_rel >= window[0]) & (s_rel <= window[1])]
            if s_rel.size:
                ax.vlines(s_rel, t + 0.4, t + 1.4, linewidth=0.6)
        ax.set_title(f"{label} Trial-by-trial Raster — Neuron {n_idx} — {fname}")
        ax.set_ylabel("Trial")
        ax.set_xlabel(f"Time (s) from {label.lower()}")
        ax.set_xlim(window)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        out = f"trial_raster_{label.lower().replace(' ', '_')}_neuron{n_idx}_{os.path.splitext(fname)[0]}.png"
        plt.savefig(os.path.join(output_dir, out), dpi=150, bbox_inches='tight')
        plt.close()

# =============== Per-file processing (with event counters) ===============
def analyze_file(fname, window, align_to_list, label, output_dir,
                 bin_size=0.01, show_trial_rasters=False, trial_neurons=(0,)):
    print(f"\n[Processing] {fname} [{label}]")
    T = load_mat_session(os.path.join("./data", fname))
    if len(T) == 0:
        print("[WARN] Empty session.")
        return

    neuron_map = build_neuron_map(T[0])
    all_spikes_abs, align_times = [], []

    # Count occurrences of 120 / 130 / 131 across the session
    cnt_120 = cnt_130 = cnt_131 = 0

    for trial in T:
        if hasattr(trial, 'EID'):
            eids_int = _eids_to_int(trial.EID)
            cnt_120 += int(np.sum(eids_int == 120))
            cnt_130 += int(np.sum(eids_int == 130))
            cnt_131 += int(np.sum(eids_int == 131))

        align = get_first_event_time(trial, align_to_list)
        if align is None:
            continue
        spikes = extract_spike_times(trial, neuron_map)
        all_spikes_abs.append(spikes)
        align_times.append(align)

    print(f"[DEBUG] {fname}: CHOICE_ON(120)={cnt_120}, OFF(130)={cnt_130}, OFF(131)={cnt_131}")

    if len(all_spikes_abs) == 0:
        print("[WARN] No valid trials.")
        return

    all_spikes_abs = np.array(all_spikes_abs, dtype=object)
    align_times = np.array(align_times, dtype=float)

    # Main figure: pooled-by-neuron raster + population rate
    plot_raster_and_rate(all_spikes_abs, align_times, neuron_map, fname, label,
                         window, output_dir, bin_size=bin_size)

    # Optional per-trial rasters for a few representative neurons
    if show_trial_rasters and len(neuron_map) > 0:
        sel = [n for n in trial_neurons if 0 <= n < len(neuron_map)]
        if sel:
            plot_trial_by_trial_raster(all_spikes_abs, align_times, sel, fname, label,
                                       window, output_dir)

# =============== Entry point ===============
def main():
    data_dir = "./data"
    output_dir = "raster_rate_nanaware_three_phase"
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".mat")])

    for f in files:
        # Pre: align to STIM_ON (118)
        analyze_file(f, window=(-0.5, 0.0), align_to_list=[STIM_ON_ID], label="Pre Stim",
                     output_dir=output_dir, bin_size=0.01, show_trial_rasters=False)

        # During: align to STIM_ON (118)
        analyze_file(f, window=(0.0, 0.5), align_to_list=[STIM_ON_ID], label="During Stim",
                     output_dir=output_dir, bin_size=0.01, show_trial_rasters=False)

        # Post: align to SACCADE_CHOICE_ON (120)
        analyze_file(f, window=(0.0, 0.5), align_to_list=[SACCADE_CHOICE_ON_ID], label="Post Stim",
                     output_dir=output_dir, bin_size=0.01, show_trial_rasters=False)

        # Optional: if you also want to inspect OFF (130/131), uncomment below
        # analyze_file(f, window=(0.0, 0.5), align_to_list=list(STIM_OFF_IDS), label="Post (OFF 130/131)",
        #              output_dir=output_dir, bin_size=0.01, show_trial_rasters=False)

# Backward-compat alias for older naming that used "CHOISE"
SACCADE_CHOISE_ON_ID = SACCADE_CHOICE_ON_ID

if __name__ == "__main__":
    main()
