# post_pre_stim.py
import os
import numpy as np
import matplotlib.pyplot as plt
from src.Loader import load_mat_session

# Constants
STIM_ON_ID = 118
STIM_OFF_ID = 130

output_dir = "post_pre_stim"
os.makedirs(output_dir, exist_ok=True)

def extract_spike_times(trial, neuron_map):
    spikes = []
    for tt, unit_idx in neuron_map:
        tt_field = f'UnitT_TT{tt}'
        if not hasattr(trial, tt_field):
            spikes.append([])
            continue
        units = getattr(trial, tt_field)

        if isinstance(units, (np.ndarray, list, tuple)) and all(hasattr(u, '__len__') for u in units):
            if unit_idx >= len(units):
                spikes.append([])
                continue
            spikes.append(np.atleast_1d(units[unit_idx]))
        else:
            if unit_idx == 0:
                spikes.append(np.atleast_1d(units))
            else:
                spikes.append([])
    return spikes

def get_event_time(trial, eid):
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids = np.atleast_1d(trial.EID)
    times = np.atleast_1d(trial.EventT)
    idx = np.where(eids == eid)[0]
    return times[idx[0]] if len(idx) > 0 else None

def plot_raster(all_spikes, align_times, neuron_map, fname, label, window):
    plt.figure(figsize=(40, 8))
    num_trials = len(all_spikes)
    num_neurons = len(neuron_map)
    colors = plt.cm.tab20(np.linspace(0, 1, num_neurons))

    for neuron_idx in range(num_neurons):
        color = colors[neuron_idx % len(colors)]
        all_spike_times = []
        for trial_idx in range(num_trials):
            spikes = all_spikes[trial_idx][neuron_idx]
            spikes = np.atleast_1d(spikes)
            try:
                flattened = np.concatenate([np.atleast_1d(s).flatten() for s in spikes])
            except Exception:
                flattened = np.atleast_1d(spikes).flatten()
            aligned = flattened.astype(float) / 1000.0 - align_times[trial_idx]
            trial_spikes = aligned[(aligned >= window[0]) & (aligned <= window[1])]
            all_spike_times.extend(trial_spikes)

        if all_spike_times:
            plt.vlines(all_spike_times, neuron_idx + 0.5, neuron_idx + 1.5, color=color, linewidth=0.7)

    plt.title(f"{label} Raster Plot - {fname}")
    plt.xlabel(f"Time (s) from {label.lower()}")
    plt.ylabel("Neuron Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"raster_{label.lower().replace(' ', '_')}_{fname}.png"))
    plt.close()

def build_neuron_map(trial0):
    neuron_map = []
    print("\n Building neuron map:")
    for tt in range(1, 9):
        tt_field = f'UnitT_TT{tt}'
        if not hasattr(trial0, tt_field):
            continue
        units = getattr(trial0, tt_field)

        if isinstance(units, (np.ndarray, list, tuple)) and all(hasattr(u, '__len__') for u in units):
            for idx in range(len(units)):
                neuron_map.append((tt, idx))
                print(f"  Neuron {len(neuron_map)-1:02d}: TT{tt}, Unit {idx}")
        else:
            neuron_map.append((tt, 0))
            print(f"  Neuron {len(neuron_map)-1:02d}: TT{tt}, Unit 0")

    print(f" Total neurons: {len(neuron_map)}\n")
    return neuron_map

def analyze_file(fname):
    print(f"\nAnalyzing {fname}...")
    T = load_mat_session(os.path.join("./data", fname))
    if len(T) == 0:
        return

    neuron_map = build_neuron_map(T[0])

    all_spikes, stim_on_times, stim_off_times = [], [], []
    valid_trial_count = 0
    for i, trial in enumerate(T):
        spikes = extract_spike_times(trial, neuron_map)
        stim_on = get_event_time(trial, STIM_ON_ID)
        stim_off = get_event_time(trial, STIM_OFF_ID)
        print(f"  Trial {i}: stim_on={stim_on}, stim_off={stim_off}")
        if stim_on is None or stim_off is None:
            continue
        valid_trial_count += 1
        all_spikes.append(spikes)
        stim_on_times.append(stim_on)
        stim_off_times.append(stim_off)

    print(f"  → Valid trials used: {valid_trial_count}\n")
    if valid_trial_count == 0:
        return

    all_spikes = np.array(all_spikes, dtype=object)
    stim_on_times = np.array(stim_on_times)
    stim_off_times = np.array(stim_off_times)

    plot_raster(all_spikes, stim_off_times, neuron_map, fname, label="Post Stim", window=(0, 0.5))
    plot_raster(all_spikes, stim_on_times, neuron_map, fname, label="Pre Stim", window=(-0.5, 0))

def main():
    data_dir = "./data"
    files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    for f in files:
        analyze_file(f)

if __name__ == "__main__":
    main()