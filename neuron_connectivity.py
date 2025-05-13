# latency_corr_graph.py


import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.Loader import load_mat_session

STIM_ON_ID = 118
STIM_OFF_ID = 120
EPS = 1e-3

output_dir = "task56_graph"
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

def compute_log_corr_matrix(spikes_by_trial, align_times, window):
    num_neurons = len(spikes_by_trial[0])
    num_trials = len(spikes_by_trial)
    rate_mat = np.full((num_neurons, num_trials), np.nan)
    for i in range(num_neurons):
        for j in range(num_trials):
            spikes = spikes_by_trial[j][i]
            aligned = np.array(spikes) - align_times[j]
            trial_spikes = aligned[(aligned >= window[0]) & (aligned <= window[1])]
            rate_mat[i, j] = len(trial_spikes) / (window[1] - window[0])
    log_rate = np.log(rate_mat + EPS)
    return np.corrcoef(np.nan_to_num(log_rate))

def draw_graph(neuron_count, matrix, label, fname):
    G = nx.Graph()
    for i in range(neuron_count):
        G.add_node(i)
    for i in range(neuron_count):
        for j in range(i+1, neuron_count):
            value = matrix[i, j]
            G.add_edge(i, j, weight=value)

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    plt.figure(figsize=(8, neuron_count / 1.5))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=800)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=8)
    plt.title(label)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def analyze_file(fname):
    print(f"\n[Graph View] {fname}")
    T = load_mat_session(os.path.join("./data", fname))
    if len(T) == 0:
        print("[WARN] Empty session")
        return

    neuron_map = build_neuron_map(T[0])
    all_spikes, stim_ons = [], []
    for trial in T:
        spikes = extract_spike_times(trial, neuron_map)
        stim_on = get_event_time(trial, STIM_ON_ID)
        if stim_on is None:
            continue
        all_spikes.append(spikes)
        stim_ons.append(stim_on)

    all_spikes = np.array(all_spikes, dtype=object)
    stim_ons = np.array(stim_ons)

    lat_mat = compute_latency_matrix(all_spikes, stim_ons, window=(0, 0.5))
    log_corr = compute_log_corr_matrix(all_spikes, stim_ons, window=(0, 0.5))

    mean_latency = np.nanmean(lat_mat, axis=1)
    latency_diff = np.abs(mean_latency[:, None] - mean_latency[None, :])

    neuron_count = len(mean_latency)
    draw_graph(neuron_count, latency_diff, "Latency Difference Graph", os.path.join(output_dir, f"latency_diff_graph_{os.path.splitext(fname)[0]}.png"))
    draw_graph(neuron_count, log_corr, "Log Correlation Graph", os.path.join(output_dir, f"log_corr_graph_{os.path.splitext(fname)[0]}.png"))

def main():
    files = [f for f in os.listdir("./data") if f.endswith(".mat")]
    for f in files:
        analyze_file(f)

if __name__ == "__main__":
    main()
