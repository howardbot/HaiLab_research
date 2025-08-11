#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster + Population Rate (NaN-aware, per-trial coverage) — fixed global time axis [-0.5, 0.5]

- Raster（上图）：固定时间轴，哪里有数据就画 spike，没数据自然留白。
- Rate（下图）：每个时间 bin 只平均“本 bin 内确有记录的 trial”，没有记录的 trial 不计入分母。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.Loader import load_mat_session

# ---- Event IDs ----
STIM_ON_ID = 118
STIM_OFF_KEEP_ID = 130
STIM_OFF_BROKEN_ID = 131
SACCADE_CHOICE_ON_ID = 120  # 作为备选

# ----------------- 基础工具 -----------------
def build_neuron_map(trial):
    neuron_map = []
    for tt in range(1, 9):
        field = f"UnitT_TT{tt}"
        if hasattr(trial, field):
            units = getattr(trial, field)
            if isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, '__len__'):
                if isinstance(units, np.ndarray) and units.dtype != object:
                    neuron_map.append((tt, 0))
                else:
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
            spikes.append([])
            continue
        units = getattr(trial, field)
        if isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, '__len__'):
            if isinstance(units, np.ndarray) and units.dtype != object:
                spikes.append(np.atleast_1d(units))
            elif unit < len(units):
                spikes.append(np.atleast_1d(units[unit]))
            else:
                spikes.append([])
        else:
            spikes.append(np.atleast_1d(units) if unit == 0 else [])
    return spikes

def get_first_event_time(trial, candidates):
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids = np.atleast_1d(trial.EID)
    times = np.atleast_1d(trial.EventT)
    for eid in candidates:
        idx = np.where(eids == eid)[0]
        if len(idx) > 0:
            return times[idx[0]]
    return None

# ----------------- 逐 trial 裁剪 + NaN-aware 平均 -----------------
def compute_population_rate_nanaware(all_spikes, bin_size=0.01, window=(-0.5, 0.5)):
    num_trials = len(all_spikes)
    num_neurons = len(all_spikes[0]) if num_trials > 0 else 0

    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    num_bins = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) * 0.5

    sum_rate = np.zeros(num_bins, dtype=float)
    count_trials = np.zeros(num_bins, dtype=int)

    for t in range(num_trials):
        s_concat = []
        for n in range(num_neurons):
            s = np.asarray(all_spikes[t][n], dtype=float)
            if s.size:
                s = s[(s >= window[0]) & (s <= window[1])]
                if s.size:
                    s_concat.append(s)
        if not s_concat:
            continue

        s_all = np.concatenate(s_concat)
        cov_start = float(np.min(s_all))
        cov_end   = float(np.max(s_all))

        covered_mask = (centers >= cov_start) & (centers <= cov_end)
        if not np.any(covered_mask):
            continue

        trial_hist = np.zeros(num_bins, dtype=float)
        for n in range(num_neurons):
            s = np.asarray(all_spikes[t][n], dtype=float)
            if s.size:
                h, _ = np.histogram(s, bins=edges)
                trial_hist += h
        trial_rate = (trial_hist / max(num_neurons, 1)) / bin_size

        sum_rate[covered_mask]     += trial_rate[covered_mask]
        count_trials[covered_mask] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        avg_rate = sum_rate / count_trials
    avg_rate[count_trials == 0] = np.nan

    return edges[:-1], avg_rate, count_trials

# ----------------- 绘图 -----------------
def plot_raster_and_rate(all_spikes_abs, align_times, neuron_map, fname, label, window, output_dir, bin_size=0.01):
    import matplotlib.gridspec as gridspec

    num_trials = len(all_spikes_abs)
    num_neurons = len(neuron_map)
    colors = plt.cm.tab20(np.linspace(0, 1, num_neurons))
    spacing = 1.2
    height = 0.8
    ytick_positions = [i * spacing + 1 for i in range(num_neurons)]

    fig = plt.figure(figsize=(28, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax_raster = fig.add_subplot(gs[0])
    ax_rate = fig.add_subplot(gs[1])

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

    for n_idx in range(num_neurons):
        color = colors[n_idx % len(colors)]
        center = ytick_positions[n_idx]
        spike_times = [t_s[n_idx] for t_s in all_spikes_rel]
        spike_flat = np.concatenate([s for s in spike_times if s.size]) if any(s.size for s in spike_times) else np.array([])
        if spike_flat.size:
            ax_raster.vlines(spike_flat, center - height/2, center + height/2, color=color, linewidth=0.6)

    ax_raster.set_title(f"{label} Raster — {fname}")
    ax_raster.set_ylabel("Neuron Index")
    ax_raster.set_yticks(ytick_positions)
    ax_raster.set_yticklabels([f"{i}" for i in range(num_neurons)])
    ax_raster.set_xlim(window)

    time_axis, avg_rate, n_contrib = compute_population_rate_nanaware(all_spikes_rel, bin_size=bin_size, window=window)

    ax_rate.plot(time_axis, avg_rate, linewidth=1.6, color="black")
    ax_rate.fill_between(time_axis, 0, np.nan_to_num(avg_rate, nan=0.0), alpha=0.25)
    ax_rate.set_ylabel("Rate (Hz)")
    ax_rate.set_xlabel(f"Time (s) from {label.lower()}")
    ax_rate.set_xlim(window)

    ax2 = ax_rate.twinx()
    ax2.plot(time_axis, n_contrib, alpha=0.6, linestyle='--', linewidth=1.0)
    ax2.set_ylabel("Trials contributing")
    ax2.set_ylim(bottom=0)

    save_name = f"raster_rate_nanaware_{label.lower().replace(' ', '_')}_{os.path.splitext(fname)[0]}.png"
    plt.savefig(os.path.join(output_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close()

# ----------------- 文件级处理 -----------------
def analyze_file(fname, window, align_to_list, label, output_dir):
    print(f"\n[Processing] {fname} [{label}]")
    T = load_mat_session(os.path.join("./data", fname))
    if len(T) == 0:
        print("[WARN] Empty session.")
        return

    neuron_map = build_neuron_map(T[0])
    all_spikes_abs, align_times = [], []
    skipped = 0

    for trial in T:
        align = get_first_event_time(trial, align_to_list)
        if align is None:
            skipped += 1
            continue
        spikes = extract_spike_times(trial, neuron_map)
        all_spikes_abs.append(spikes)
        align_times.append(align)

    if skipped:
        print(f"[Info] skipped {skipped} trials (missing align event).")
    if len(all_spikes_abs) == 0:
        print("[WARN] No valid trials after alignment filtering.")
        return

    all_spikes_abs = np.array(all_spikes_abs, dtype=object)
    align_times = np.array(align_times, dtype=float)

    plot_raster_and_rate(all_spikes_abs, align_times, neuron_map, fname, label, window, output_dir, bin_size=0.01)

# ----------------- 主程序 -----------------
def main():
    data_dir = "./data"
    output_dir = "raster_rate_nanaware"
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

    if not files:
        print("[WARN] No .mat files in ./data")
        return

    window = (-0.5, 0.5)

    for f in files:
        analyze_file(f, window=window, align_to_list=[STIM_ON_ID], label="Stim On (±0.5s)", output_dir=output_dir)
        analyze_file(f, window=window, align_to_list=[STIM_OFF_KEEP_ID, STIM_OFF_BROKEN_ID, SACCADE_CHOICE_ON_ID],
                     label="Stim Off (±0.5s)", output_dir=output_dir)

if __name__ == "__main__":
    main()
