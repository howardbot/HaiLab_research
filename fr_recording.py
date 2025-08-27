#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverage-aware pooled raster + PSTH for Pre / During / Post.

Rules:
- Pre:    [StimOn-0.5, StimOn] (requires EID=118), clip to trial bounds
- During: [StimOn, min(StimOn+0.5, ChoiceOn if present)], clip to trial bounds
- Post:   [ChoiceOn, ChoiceOn+0.5] (requires EID=120), clip to trial bounds

Coverage-aware PSTH:
For each time bin, divide total spike counts by (num_neurons * total covered seconds in that bin).
This avoids downward bias when some trials don't fully cover a bin.

Generates one PNG per phase per session under ./raster_psth_coverage
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.Loader import load_mat_session

# ---- Event IDs ----
TRIAL_START_ID = 111
TRIAL_END_ID = 112
STIM_ON_ID = 118
CHOICE_ON_ID = 120

# ---- Parameters ----
BIN_SIZE = 0.01  # 10 ms bins
OUT_DIR = "./raster_psth_coverage"

def _iter_units(units):
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
    nm = []
    for tt in range(1, 9):
        f = f"UnitT_TT{tt}"
        if hasattr(trial, f):
            units = _iter_units(getattr(trial, f))
            for i in range(len(units)):
                nm.append((tt, i))
    return nm

def extract_spike_times(trial, neuron_map):
    out = []
    for tt, unit in neuron_map:
        f = f"UnitT_TT{tt}"
        if not hasattr(trial, f):
            out.append([]); continue
        units = _iter_units(getattr(trial, f))
        out.append(units[unit] if unit < len(units) else [])
    return out

def _eids_to_int(eids):
    e = np.atleast_1d(eids)
    try:
        return np.rint(e.astype(float)).astype(int)
    except Exception:
        return np.array([int(str(x)) for x in e], dtype=int)

def _event_time(trial, eid):
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
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None, None
    eids = _eids_to_int(trial.EID)
    ts = np.atleast_1d(trial.EventT)
    if ts.size != eids.size or ts.size == 0:
        return None, None
    tmin, tmax = float(np.min(ts)), float(np.max(ts))
    t_start = _event_time(trial, TRIAL_START_ID)
    t_end = _event_time(trial, TRIAL_END_ID)
    if t_start is None: t_start = tmin
    if t_end is None: t_end = tmax
    return t_start, t_end

def _compute_phase_windows(trial):
    stim = _event_time(trial, STIM_ON_ID)
    choice = _event_time(trial, CHOICE_ON_ID)
    t0, t1 = _trial_bounds(trial)
    windows = {}

    # Pre
    if stim is not None and t0 is not None:
        a0 = max(stim - 0.5, t0)
        a1 = min(stim, t1 if t1 is not None else np.inf)
        if a1 - a0 > 0:
            windows['pre'] = (a0, a1)

    # During
    if stim is not None and t0 is not None:
        nominal_end = stim + 0.5
        if choice is not None:
            nominal_end = min(nominal_end, choice)
        a0 = max(stim, t0)
        a1 = min(nominal_end, t1 if t1 is not None else np.inf)
        if a1 - a0 > 0:
            windows['during'] = (a0, a1)

    # Post
    if choice is not None and t0 is not None:
        a0 = max(choice, t0)
        a1 = min(choice + 0.5, t1 if t1 is not None else np.inf)
        if a1 - a0 > 0:
            windows['post'] = (a0, a1)

    return windows

def pooled_raster_and_psth_coverage(T, fname, bin_size=BIN_SIZE):
    if len(T) == 0:
        return

    neuron_map = build_neuron_map(T[0])
    num_neurons = len(neuron_map)
    if num_neurons == 0:
        return

    # Collect per-trial absolute spikes
    trial_spikes_abs = [extract_spike_times(tr, neuron_map) for tr in T]

    # Build phase windows per trial
    trial_windows = [ _compute_phase_windows(tr) for tr in T ]

    phases = {
        'pre':  ('Pre Stim',  -0.5, 0.0),  # relative range for plotting only
        'during': ('During Stim', 0.0, 0.5),
        'post': ('Post Stim', 0.0, 0.5),
    }

    os.makedirs(OUT_DIR, exist_ok=True)

    for key, (label, rel_lo, rel_hi) in phases.items():
        # Determine absolute plot window per trial for this phase
        # and also gather coverage-aware PSTH bins
        # Set global relative axis for figure
        edges_rel = np.arange(rel_lo, rel_hi + bin_size, bin_size)
        num_bins = len(edges_rel) - 1
        bin_spike_counts = np.zeros(num_bins, dtype=float)
        bin_covered_sec = np.zeros(num_bins, dtype=float)

        # Raster pooling: we'll pool spikes across trials per neuron in relative coords
        pooled_rel_spikes_per_neuron = [ [] for _ in range(num_neurons) ]

        any_trial = False

        for tr_idx, tr in enumerate(T):
            if key not in trial_windows[tr_idx]:
                continue
            any_trial = True
            abs_start, abs_end = trial_windows[tr_idx][key]

            # For PSTH coverage: compute overlap with each bin
            # Bin k absolute interval is [abs_anchor + edges_rel[k], abs_anchor + edges_rel[k+1]]
            # But here we don't have a single anchor; the window itself defines the absolute interval.
            # We'll map bin edges into the trial-specific absolute window by linear interpolation:
            # relative [rel_lo, rel_hi] -> absolute [abs_start, abs_end]
            # abs_t = abs_start + (rel - rel_lo) * (abs_end - abs_start) / (rel_hi - rel_lo)
            win_len_rel = (rel_hi - rel_lo)
            abs_len = abs_end - abs_start
            if win_len_rel <= 0 or abs_len <= 0:
                continue

            # Precompute mapping for edges
            abs_edges = abs_start + (edges_rel - rel_lo) * (abs_len / win_len_rel)

            # Count coverage per bin (overlap with [abs_edges[k], abs_edges[k+1]] ∩ [abs_start, abs_end])
            # which is simply abs_edges[k+1] - abs_edges[k] = bin_size * scaling
            # But due to clipping to trial bounds earlier, abs_edges are within [abs_start, abs_end] already.
            # So per-bin covered seconds contributed by this trial equals (abs_edges[k+1] - abs_edges[k]).
            per_bin_cover = abs_edges[1:] - abs_edges[:-1]
            bin_covered_sec += per_bin_cover

            # Count spikes per bin for all neurons
            spikes_abs = trial_spikes_abs[tr_idx]
            for n_idx in range(num_neurons):
                s = np.asarray(spikes_abs[n_idx], dtype=float)
                if s.size == 0:
                    continue
                # keep only spikes within the absolute window
                s = s[(s >= abs_start) & (s <= abs_end)]
                if s.size == 0:
                    continue
                # For raster pooling: convert to relative coords in [rel_lo, rel_hi] via linear map
                s_rel = rel_lo + (s - abs_start) * (win_len_rel / abs_len)
                pooled_rel_spikes_per_neuron[n_idx].append(s_rel)

                # For PSTH: histogram using abs_edges
                h, _ = np.histogram(s, bins=abs_edges)
                bin_spike_counts += h

        if not any_trial:
            continue

        # Compute coverage-aware rate per neuron (population average): divide by num_neurons and covered seconds
        # rate_pop (Hz / neuron) per bin:
        #   rate_bin = (bin_spike_counts / num_neurons) / bin_covered_sec
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_per_bin = (bin_spike_counts / max(num_neurons,1)) / bin_covered_sec
            rate_per_bin[~np.isfinite(rate_per_bin)] = np.nan

        # ---- Plot ----
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
        ax_raster = fig.add_subplot(gs[0])
        ax_psth = fig.add_subplot(gs[1])

        # Raster (pooled-by-neuron across trials)
        colors = plt.cm.tab20(np.linspace(0, 1, num_neurons))
        spacing, height = 1.2, 0.8
        yticks = []
        ytick_labels = []
        for n_idx in range(num_neurons):
            center = n_idx * spacing + 1
            yticks.append(center)
            ytick_labels.append(str(n_idx))
            if len(pooled_rel_spikes_per_neuron[n_idx]) == 0:
                continue
            s_rel = np.concatenate(pooled_rel_spikes_per_neuron[n_idx])
            if s_rel.size:
                ax_raster.vlines(s_rel, center - height/2, center + height/2,
                                 color=colors[n_idx % len(colors)], linewidth=0.5)

        ax_raster.set_title(f"{label} — coverage-aware pooled raster — {fname}")
        ax_raster.set_ylabel("Neuron Index")
        ax_raster.set_xlim(rel_lo, rel_hi)
        ax_raster.set_yticks(yticks)
        ax_raster.set_yticklabels(ytick_labels)


        # PSTH: use bar instead of line
        centers = edges_rel[:-1] + bin_size / 2
        ax_psth.bar(
            centers,
            rate_per_bin,
            width=bin_size,
            align='center',
            color='black',
            alpha=0.7,
            edgecolor='blue'
        )
        ax_psth.set_xlim(rel_lo, rel_hi)
        ax_psth.set_xlabel(f"Time (s) from {label.lower()} (relative axis)")
        ax_psth.set_ylabel("Rate (Hz / neuron)")

        ax_psth.set_xlim(rel_lo, rel_hi)
        ax_psth.set_xlabel(f"Time (s) from {label.lower()} (relative axis)")
        ax_psth.set_ylabel("Rate (Hz / neuron)")

        # Secondary axis: covered seconds per bin (useful diagnostic)
        ax2 = ax_psth.twinx()
        ax2.plot(centers, bin_covered_sec, lw=1.0, ls='--', alpha=0.7, label="Covered seconds per bin (sum over trials)")
        ax2.set_ylabel("Covered sec (sum over trials)")
        ax2.legend(loc='upper right')

        os.makedirs(OUT_DIR, exist_ok=True)
        out_name = os.path.join(OUT_DIR, f"raster_psth_coverage_{key}_{os.path.splitext(fname)[0]}.png")
        fig.savefig(out_name, dpi=150, bbox_inches='tight')
        plt.close(fig)

def main():
    data_dir = "./data"
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".mat")])
    for f in files:
        print(f"Processing {f} ...")
        T = load_mat_session(os.path.join(data_dir, f))
        pooled_raster_and_psth_coverage(T, f)
    print(f"Done. Figures saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
