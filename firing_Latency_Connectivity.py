# firing_latency_connectivity.py firing rate log correlation & latency correlation）
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.Loader import load_mat_session

STIM_ON_ID = 118
STIM_OFF_ID = 120
EPS = 1e-3

output_dir = "task56_connectivity"
os.makedirs(output_dir, exist_ok=True)

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
                spikes.append(units)
            elif unit < len(units):
                spikes.append(np.atleast_1d(units[unit]))
            else:
                spikes.append([])
        else:
            spikes.append(np.atleast_1d(units) if unit == 0 else [])
    return spikes

def get_event_time(trial, eid):
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids = np.atleast_1d(trial.EID)
    times = np.atleast_1d(trial.EventT)
    idx = np.where(eids == eid)[0]
    return times[idx[0]] if len(idx) > 0 else None

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

def compute_firing_rate_matrix(spikes_by_trial, align_times, window, trial_valids):
    num_neurons = len(spikes_by_trial[0])
    num_trials = len(spikes_by_trial)
    rate_mat = np.full((num_neurons, num_trials), np.nan)
    for neuron_idx in range(num_neurons):
        for trial_idx in range(num_trials):
            if not trial_valids[trial_idx]:
                continue
            spikes = spikes_by_trial[trial_idx][neuron_idx]
            aligned = np.array(spikes) - align_times[trial_idx]
            trial_spikes = aligned[(aligned >= window[0]) & (aligned <= window[1])]
            rate_mat[neuron_idx, trial_idx] = len(trial_spikes) / (window[1] - window[0])
    return rate_mat

def compute_latency_matrix(spikes_by_trial, align_times, window):
    num_neurons = len(spikes_by_trial[0])
    num_trials = len(spikes_by_trial)
    latency_mat = np.full((num_neurons, num_trials), np.nan)
    for i in range(num_neurons):
        for j in range(num_trials):
            spikes = spikes_by_trial[j][i]
            aligned = np.array(spikes) - align_times[j]
            trial_spikes = aligned[(aligned >= window[0]) & (aligned <= window[1])]
            if len(trial_spikes) > 0:
                latency_mat[i, j] = trial_spikes[0]
    return latency_mat

def plot_matrix(matrix, title, fname):
    plt.figure(figsize=(10, 8))
    labels = [f"N{i}" for i in range(matrix.shape[0])]
    sns.heatmap(matrix, cmap='coolwarm', xticklabels=labels, yticklabels=labels, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def analyze_file(fname):
    print(f"\n[Processing] {fname} for Task 5 & 6")
    T = load_mat_session(os.path.join("./data", fname))
    if len(T) == 0:
        print("[WARN] Empty session")
        return

    neuron_map = build_neuron_map(T[0])
    all_spikes, stim_ons, stim_offs = [], [], []
    trial_valids = {"pre": [], "during": [], "post": []}

    for trial in T:
        spikes = extract_spike_times(trial, neuron_map)
        stim_on = get_event_time(trial, STIM_ON_ID)
        stim_off = get_event_time(trial, STIM_OFF_ID)
        if stim_on is None or stim_off is None:
            continue
        trial_start = 0.0
        trial_end = max([stim_on, stim_off]) + 1.0

        all_spikes.append(spikes)
        stim_ons.append(stim_on)
        stim_offs.append(stim_off)

        trial_valids["pre"].append(stim_on - 0.5 >= trial_start)
        trial_valids["during"].append(True)
        trial_valids["post"].append(stim_off + 0.5 <= trial_end)

    if len(all_spikes) == 0:
        print("[WARN] No valid trials")
        return

    all_spikes = np.array(all_spikes, dtype=object)
    stim_ons = np.array(stim_ons)
    stim_offs = np.array(stim_offs)

    # --- Task 5 ---
    for label, align, window, valids in [
        ("Pre-Stim", stim_ons, (-0.5, 0), trial_valids["pre"]),
        ("During-Stim", stim_ons, (0, 0.5), trial_valids["during"]),
        ("Post-Stim", stim_offs, (0, 0.5), trial_valids["post"]),
    ]:
        rate_mat = compute_firing_rate_matrix(all_spikes, align, window, valids)
        log_mat = np.log(rate_mat + EPS)
        if log_mat.shape[0] > 1:
            corr = np.corrcoef(np.nan_to_num(log_mat))
            plot_matrix(corr, title=f"{label} Log Rate Correlation", fname=f"log_rate_{label.lower().replace('-', '_')}_{os.path.splitext(fname)[0]}.png")

    # --- Task 6: latency correlation (only During-stim makes sense) ---
    lat_mat = compute_latency_matrix(all_spikes, stim_ons, window=(0, 0.5))
    if lat_mat.shape[0] > 1:
        lat_corr = np.corrcoef(np.nan_to_num(lat_mat))
        plot_matrix(lat_corr, title="Latency Correlation (During-Stim)", fname=f"latency_corr_{os.path.splitext(fname)[0]}.png")

def main():
    files = [f for f in os.listdir("./data") if f.endswith(".mat")]
    for f in files:
        analyze_file(f)

if __name__ == "__main__":
    main()
