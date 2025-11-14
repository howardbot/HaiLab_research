import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.Loader import load_mat_session

# ===================== Constant settings ===================== #

STIM_ON_ID = 118          # Stimulus ON event ID
EPS = 1e-3

# Latency window & trial criteria
LAT_WINDOW = (0.0, 0.2)   # Time window (in s) for latency estimation
MIN_TRIALS = 5            # A neuron must have at least 5 spikes (trials with valid first spike) to be considered valid

# Connectivity analysis window (on the time axis after latency compensation)
CONN_WINDOW = (0.0, 0.2)  # e.g., after aligning to each neuron’s latency, compute firing rate in 0–200 ms

# Thresholds for directed connectivity
CORR_THRESH = 0.4         # Correlation strength threshold for defining a connection
MIN_LAT_DIFF = 0.001      # Minimal latency difference (s) to define direction, to avoid ambiguous edges

# Output directory
output_dir = "neuron_Connectivity_latencyDirected"
os.makedirs(output_dir, exist_ok=True)


# ===================== Helper: data structures ===================== #

def build_neuron_map(trial):
    """
    Build neuron_map: [(tt, unit_idx), ...]
    tt: electrode index, 1~8
    unit_idx: unit index under this tetrode (TT)
    """
    neuron_map = []
    for tt in range(1, 9):
        field = f"UnitT_TT{tt}"
        if hasattr(trial, field):
            units = getattr(trial, field)
            if isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, '__len__'):
                # Non-object ndarray: treat as a single unit
                if isinstance(units, np.ndarray) and units.dtype != object:
                    neuron_map.append((tt, 0))
                else:
                    # Object/list: each element is one unit
                    for i in range(len(units)):
                        neuron_map.append((tt, i))
            else:
                # Scalar case
                neuron_map.append((tt, 0))
    return neuron_map


def extract_spike_times(trial, neuron_map):
    """
    Extract spike times (absolute time) for each neuron according to neuron_map.
    Returns a list of length num_neurons, each element is np.array(spike_times).
    """
    spikes = []
    for tt, unit in neuron_map:
        field = f"UnitT_TT{tt}"
        if not hasattr(trial, field):
            spikes.append([])
            continue
        units = getattr(trial, field)
        if isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, '__len__'):
            # ndarray and not object: treat as a single unit's spike times
            if isinstance(units, np.ndarray) and units.dtype != object:
                spikes.append(units)
            # object / list: index by unit
            elif unit < len(units):
                spikes.append(np.atleast_1d(units[unit]))
            else:
                spikes.append([])
        else:
            # Scalar or single array: only use when unit == 0
            spikes.append(np.atleast_1d(units) if unit == 0 else [])
    return spikes


def get_event_time(trial, eid):
    """Get the time of a given event ID from trial (return None if not present)."""
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids = np.atleast_1d(trial.EID)
    times = np.atleast_1d(trial.EventT)
    idx = np.where(eids == eid)[0]
    return times[idx[0]] if len(idx) > 0 else None


# ===================== Latency-related functions ===================== #

def compute_neuron_latency(spikes_by_trial, align_times,
                           window=LAT_WINDOW, min_trials=MIN_TRIALS):
    """
    Compute typical response latency for each neuron:

    For each trial:
        - Align spike times to stim_on
        - Within the specified window, find the time of the first spike

    Across trials:
        - Take the median of "first spike times" as the neuron’s latency.

    Returns:
        latencies: shape = (num_neurons,), unit: seconds
    """
    num_trials = len(spikes_by_trial)
    if num_trials == 0:
        return np.array([])

    num_neurons = len(spikes_by_trial[0])
    latencies = np.full(num_neurons, np.nan)

    for i in range(num_neurons):
        neuron_lat_list = []
        for j in range(num_trials):
            # spikes_by_trial[j][i] is the absolute spike times before any alignment
            spikes_abs = np.asarray(spikes_by_trial[j][i])
            if spikes_abs.size == 0:
                continue
            aligned = spikes_abs - align_times[j]
            in_win = aligned[(aligned >= window[0]) & (aligned <= window[1])]
            if in_win.size > 0:
                neuron_lat_list.append(np.min(in_win))  # first spike in window

        if len(neuron_lat_list) >= min_trials:
            latencies[i] = np.median(neuron_lat_list)

    return latencies


def save_latency_csv(fname_out, latencies, neuron_map):
    """
    Save latency of each neuron to CSV.

    Columns:
        neuron_idx, TT, unit_idx, latency_s, latency_ms
    """
    rows = []
    for idx, (tt, unit_idx) in enumerate(neuron_map):
        lat = latencies[idx]
        if np.isnan(lat):
            # Could also keep NaNs; here we skip invalid neurons
            continue
        rows.append([idx, tt, unit_idx, lat, lat * 1000.0])

    if len(rows) == 0:
        print(f"[WARN] No valid latencies to save for {fname_out}")
        return

    rows = np.array(rows, dtype=float)
    header = "neuron_idx,TT,unit_idx,latency_s,latency_ms"
    np.savetxt(fname_out, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"[SAVE] Latency CSV -> {fname_out}")


# ===================== Latency compensation + connectivity ===================== #

def latency_shift_spikes_relative(spikes_by_trial, align_times, latencies):
    """
    Inputs:
        spikes_by_trial[j][i] = spike array (absolute times) of neuron i in trial j
        align_times[j]        = absolute StimON time for trial j
        latencies[i]          = latency of neuron i (relative to StimON, in seconds)

    Output:
        shifted_spikes[j][i] = spike times in a "latency-compensated" frame:
                               spike_time_shifted = (spike_abs - stim_on) - latency[i]
    """
    num_trials = len(spikes_by_trial)
    if num_trials == 0:
        return []

    num_neurons = len(spikes_by_trial[0])

    shifted = []
    for j in range(num_trials):
        trial_list = []
        for i in range(num_neurons):
            sp_abs = np.asarray(spikes_by_trial[j][i], dtype=float)
            if sp_abs.size == 0:
                trial_list.append(np.array([]))
                continue

            if np.isnan(latencies[i]):
                # If neuron has no reliable latency, only align to StimON, no neuron-level shift
                aligned = sp_abs - align_times[j]
            else:
                aligned = (sp_abs - align_times[j]) - latencies[i]

            trial_list.append(aligned)
        shifted.append(trial_list)

    return shifted


def compute_rate_corr_from_shifted(spikes_shifted, window=CONN_WINDOW):
    """
    Compute firing-rate correlation matrix on latency-compensated spikes.

    spikes_shifted[j][i] are spike times already aligned to StimON and
    shifted by neuron-specific latencies.

    Within the given window, for each neuron and trial:
        rate = spike_count / window_length

    Returns:
        corr_mat: (num_neurons, num_neurons)
    """
    num_trials = len(spikes_shifted)
    if num_trials == 0:
        return np.array([[]])

    num_neurons = len(spikes_shifted[0])

    rate_mat = np.full((num_neurons, num_trials), np.nan)
    win_len = window[1] - window[0]

    for i in range(num_neurons):
        for j in range(num_trials):
            sp = np.asarray(spikes_shifted[j][i])
            if sp.size == 0:
                rate_mat[i, j] = 0.0
                continue
            in_win = sp[(sp >= window[0]) & (sp <= window[1])]
            rate_mat[i, j] = len(in_win) / win_len

    # Avoid log(0)
    log_rate = np.log(rate_mat + EPS)
    corr_mat = np.corrcoef(np.nan_to_num(log_rate))

    return corr_mat


def build_directed_adjacency(latencies, corr_mat,
                             corr_thresh=CORR_THRESH,
                             min_lat_diff=MIN_LAT_DIFF):
    """
    Build directed adjacency matrix based on latency order and correlation.

    We define a directed edge i -> j if and only if:
        - corr_mat[i, j] >= corr_thresh
        - latencies[i] + min_lat_diff <= latencies[j]
        - Both latencies are non-NaN

    Returns:
        adj: 0/1 matrix, shape = (num_neurons, num_neurons)
    """
    num_neurons = len(latencies)
    adj = np.zeros((num_neurons, num_neurons), dtype=int)

    for i in range(num_neurons):
        for j in range(num_neurons):
            if i == j:
                continue

            li = latencies[i]
            lj = latencies[j]
            if np.isnan(li) or np.isnan(lj):
                continue

            c = corr_mat[i, j]
            if np.isnan(c):
                continue

            # Earlier neuron points to later neuron, with sufficiently high correlation
            if c >= corr_thresh and (li + min_lat_diff) <= lj:
                adj[i, j] = 1

    return adj


def save_matrix_csv(fname_out, mat, fmt="%.6f"):
    """Save any matrix to CSV."""
    np.savetxt(fname_out, mat, delimiter=",", fmt=fmt)
    print(f"[SAVE] Matrix CSV -> {fname_out}")


# ===================== Visualization: directed graph ===================== #

def draw_directed_graph(neuron_map, latencies, corr_mat, adj_mat,
                        title, fname, corr_thresh=CORR_THRESH):
    """
    Plot a directed graph:
        - Nodes: neurons
        - Layout: arranged by tetrode (TT) layers
        - Node color: latency (ms)
        - Edges: where adj_mat[i, j] == 1
        - Edge width: proportional to correlation strength
    """
    num_neurons = len(neuron_map)
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_neurons):
        G.add_node(i)

    # Group neurons by tetrode (electrode) layer
    electrode_layers = {}
    for idx, (tt, _) in enumerate(neuron_map):
        electrode_layers.setdefault(tt, []).append(idx)

    # Compute node positions: each TT forms one horizontal layer
    pos = {}
    sorted_electrodes = sorted(electrode_layers.keys())
    layer_y = {}
    for layer_idx, tt in enumerate(sorted_electrodes):
        neurons = electrode_layers[tt]
        x_start = - (len(neurons) - 1) / 2
        y = -layer_idx
        layer_y[tt] = y
        for i, nid in enumerate(neurons):
            x = x_start + i
            pos[nid] = (x, y)

    # Add directed edges
    edge_weights = []
    edge_list = []
    for i in range(num_neurons):
        for j in range(num_neurons):
            if adj_mat[i, j] == 1:
                edge_list.append((i, j))
                w = corr_mat[i, j]
                # Map corr_thresh~1.0 to line width 0.5~5
                edge_weights.append(0.5 + 4.5 * (w - corr_thresh) / max(1e-6, 1.0 - corr_thresh))

    # Node color: latency in ms
    lat_ms = np.array([lat * 1000.0 if not np.isnan(lat) else np.nan for lat in latencies])
    vmin = np.nanmin(lat_ms) if np.any(~np.isnan(lat_ms)) else 0.0
    vmax = np.nanmax(lat_ms) if np.any(~np.isnan(lat_ms)) else 1.0
    # Nodes with NaN latency will be drawn in gray
    node_colors = []
    for val in lat_ms:
        if np.isnan(val):
            node_colors.append(vmax + 10)  # put them outside colorbar range; recolor later
        else:
            node_colors.append(val)

    plt.figure(figsize=(12, max(5, num_neurons / 2.5)))

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap='viridis',
        node_size=800,
        vmin=vmin,
        vmax=vmax
    )

    # Manually redraw NaN-latency nodes as gray
    ax = plt.gca()
    for idx, (node, color_val) in enumerate(zip(G.nodes(), node_colors)):
        if color_val > vmax:
            x, y = pos[node]
            ax.scatter([x], [y], s=800, c='lightgray', zorder=3)

    nx.draw_networkx_labels(G, pos)

    if edge_list:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_list,
            width=edge_weights,
            arrowstyle='-|>',
            arrowsize=15,
            connectionstyle='arc3,rad=0.1'
        )

    # Add tetrode (TT) labels on the left
    for tt in sorted_electrodes:
        plt.text(-5, layer_y[tt], f"TT{tt}",
                 fontsize=10,
                 verticalalignment='center',
                 horizontalalignment='right')

    # Add colorbar for latency
    if np.any(~np.isnan(lat_ms)):
        cbar = plt.colorbar(nodes, ax=ax)
        cbar.set_label("Latency (ms)")

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"[SAVE] Directed graph -> {fname}")


# ===================== Main pipeline ===================== #

def analyze_file(fname):
    print(f"\n[Latency-Directed Connectivity] {fname}")
    T = load_mat_session(os.path.join("./data", fname))
    if len(T) == 0:
        print("[WARN] Empty session")
        return

    # Build neuron_map using the first trial
    neuron_map = build_neuron_map(T[0])
    if len(neuron_map) == 0:
        print("[WARN] No neurons found in this session.")
        return

    # Collect spike times (absolute) and StimON times for all trials
    all_spikes_abs, stim_ons = [], []
    for trial in T:
        stim_on = get_event_time(trial, STIM_ON_ID)
        if stim_on is None:
            continue
        spikes = extract_spike_times(trial, neuron_map)
        all_spikes_abs.append(spikes)
        stim_ons.append(stim_on)

    if len(all_spikes_abs) == 0:
        print("[WARN] No trials with STIM_ON event for this session.")
        return

    all_spikes_abs = np.array(all_spikes_abs, dtype=object)
    stim_ons = np.array(stim_ons)

    num_trials = len(all_spikes_abs)
    num_neurons = len(neuron_map)
    print(f"  trials used: {num_trials}")
    print(f"  neurons: {num_neurons}")

    # === Step 1: compute latency for each neuron === #
    latencies = compute_neuron_latency(
        all_spikes_abs, stim_ons,
        window=LAT_WINDOW,
        min_trials=MIN_TRIALS
    )

    print("  [Latency per neuron] (ms, NaN = could not be estimated)")
    for idx, (tt, unit_idx) in enumerate(neuron_map):
        lat_ms = latencies[idx] * 1000 if not np.isnan(latencies[idx]) else np.nan
        print(f"    neuron {idx:02d} (TT{tt}, unit {unit_idx}): {lat_ms:.2f} ms")

    # Save latency CSV
    basename = os.path.splitext(fname)[0]
    out_latency_csv = os.path.join(output_dir, f"latency_{basename}.csv")
    save_latency_csv(out_latency_csv, latencies, neuron_map)

    # === Step 2: latency-compensated alignment of spikes === #
    spikes_shifted = latency_shift_spikes_relative(all_spikes_abs, stim_ons, latencies)

    # === Step 3: compute connectivity (rate correlation) on compensated data === #
    corr_mat = compute_rate_corr_from_shifted(spikes_shifted, window=CONN_WINDOW)

    out_corr_csv = os.path.join(output_dir, f"corr_latencyCorrected_{basename}.csv")
    save_matrix_csv(out_corr_csv, corr_mat, fmt="%.6f")

    # === Step 4: build directed adjacency matrix (core of Method C) === #
    adj_mat = build_directed_adjacency(latencies, corr_mat,
                                       corr_thresh=CORR_THRESH,
                                       min_lat_diff=MIN_LAT_DIFF)

    out_adj_csv = os.path.join(output_dir, f"adj_latencyDirected_{basename}.csv")
    save_matrix_csv(out_adj_csv, adj_mat, fmt="%d")

    # === Step 5: plot directed graph === #
    title = (
        f"Latency-directed connectivity (corr>={CORR_THRESH}, "
        f"Δlat>={MIN_LAT_DIFF*1000:.1f} ms)\n{basename}"
    )
    out_fig = os.path.join(output_dir, f"graph_latencyDirected_{basename}.png")
    draw_directed_graph(neuron_map, latencies, corr_mat, adj_mat, title, out_fig)


def main():
    files = [f for f in os.listdir("./data") if f.endswith(".mat")]
    if not files:
        print("[WARN] No .mat files found in ./data")
    for f in files:
        analyze_file(f)


if __name__ == "__main__":
    main()