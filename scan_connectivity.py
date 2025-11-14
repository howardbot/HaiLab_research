"""
scan_connectivity.py

Parameter scan for latency-compensated directed connectivity:

For each .mat session in ./data:
    1) Build neuron map (TT, unit).
    2) Extract spikes and StimOn times.
    3) For each parameter combination:
        - Compute neuron latencies.
        - Apply latency compensation.
        - Compute connectivity window firing-rate correlation.
        - Build directed adjacency: early neuron -> late neuron.
        - Evaluate connectivity with a scalar score.
    4) Return all results (no plots, no per-parameter files).

At the end:
    - Concatenate results from all sessions.
    - Save a single CSV: scan_connectivity_results.csv
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.Loader import load_mat_session

# ===================== Global constants ===================== #

STIM_ON_ID = 118        # Event ID for stimulus onset
EPS = 1e-3
MIN_TRIALS = 5          # Min trials with spikes to accept a latency estimate

# Parameter grids for scanning
# You can adjust these lists to enlarge or shrink the search space.
LAT_WINDOW = (0.0, 0.2)  # Window used to estimate neuron latency (fixed here)

CONN_BIN_SIZES = [0.10, 0.20]      # seconds, e.g. 100 ms, 200 ms
CONN_START_TIMES = [0.0, 0.05]     # window start relative to (latency-compensated) time axis

CORR_THRESH_LIST = [0.3, 0.4, 0.5, 0.6]
LAT_DIFF_LIST = [0.003, 0.005, 0.010]  # minimum latency difference for direction (3, 5, 10 ms)

# Output file
SCAN_OUTPUT_CSV = "scan_connectivity_results.csv"


# ===================== Basic helpers: neuron map, spikes, events ===================== #

def build_neuron_map(trial):
    """
    Build neuron_map: [(tt, unit_idx), ...]
    tt: 1~8 tetrode index (assumed to reflect depth ordering).
    unit_idx: unit index on that tetrode.
    """
    neuron_map = []
    for tt in range(1, 9):
        field = f"UnitT_TT{tt}"
        if hasattr(trial, field):
            units = getattr(trial, field)
            if isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, '__len__'):
                # Non-object numpy array -> treat as a single unit
                if isinstance(units, np.ndarray) and units.dtype != object:
                    neuron_map.append((tt, 0))
                else:
                    # list / object array -> one unit per element
                    for i in range(len(units)):
                        neuron_map.append((tt, i))
            else:
                # Scalar or other types -> single unit
                neuron_map.append((tt, 0))
    return neuron_map


def extract_spike_times(trial, neuron_map):
    """
    Extract spike times (absolute time) for each neuron defined in neuron_map.

    Returns:
        spikes: list of length num_neurons
                spikes[i] is a 1D numpy array of spike times (absolute).
    """
    spikes = []
    for tt, unit in neuron_map:
        field = f"UnitT_TT{tt}"
        if not hasattr(trial, field):
            spikes.append(np.array([]))
            continue

        units = getattr(trial, field)
        if isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, '__len__'):
            # Non-object numpy array: a single unit's spike times
            if isinstance(units, np.ndarray) and units.dtype != object:
                spikes.append(np.asarray(units))
            # Object array / list: pick the unit-th element
            elif unit < len(units):
                spikes.append(np.atleast_1d(units[unit]))
            else:
                spikes.append(np.array([]))
        else:
            # Scalar / single array: only used if unit == 0
            spikes.append(np.atleast_1d(units) if unit == 0 else np.array([]))

    return spikes


def get_event_time(trial, eid):
    """
    Get time of a specific event ID from a trial.
    Returns None if not found.
    """
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None

    eids = np.atleast_1d(trial.EID)
    times = np.atleast_1d(trial.EventT)
    idx = np.where(eids == eid)[0]
    return times[idx[0]] if len(idx) > 0 else None


# ===================== Latency-related helpers ===================== #

def compute_neuron_latency(
    spikes_by_trial_abs,
    align_times,
    window=LAT_WINDOW,
    min_trials=MIN_TRIALS
):
    """
    Compute typical response latency for each neuron.

    For each neuron i:
        - For each trial j:
            - Align spikes to StimOn: t_aligned = spike_abs - align_times[j].
            - Consider only spikes within "window".
            - Take the first spike in that window.
        - Compute the median first-spike time across trials (if >= min_trials available).

    Args:
        spikes_by_trial_abs: list of length num_trials
                             each element is a list of length num_neurons with absolute spike times.
        align_times        : array of shape (num_trials,)
        window             : (start, end) in seconds relative to StimOn
        min_trials         : minimum number of trials with spikes to accept a latency

    Returns:
        latencies: numpy array of shape (num_neurons,), in seconds relative to StimOn.
                   NaN where latency estimation is not reliable.
    """
    num_trials = len(spikes_by_trial_abs)
    if num_trials == 0:
        return np.array([])

    num_neurons = len(spikes_by_trial_abs[0])
    latencies = np.full(num_neurons, np.nan)

    for i in range(num_neurons):
        lat_list = []
        for j in range(num_trials):
            spikes_abs = np.asarray(spikes_by_trial_abs[j][i])
            if spikes_abs.size == 0:
                continue
            aligned = spikes_abs - align_times[j]
            in_win = aligned[(aligned >= window[0]) & (aligned <= window[1])]
            if in_win.size > 0:
                lat_list.append(np.min(in_win))  # first spike

        if len(lat_list) >= min_trials:
            latencies[i] = np.median(lat_list)

    return latencies


def latency_shift_spikes_relative(spikes_by_trial_abs, align_times, latencies):
    """
    Apply per-neuron latency compensation.

    For each trial j and neuron i:
        spike_time_shifted = (spike_abs - StimOn_j) - latency[i]

    Neurons with NaN latency are only aligned to StimOn (no latency shift).

    Args:
        spikes_by_trial_abs: list of length num_trials, each a list of length num_neurons
        align_times        : array of length num_trials
        latencies          : array of length num_neurons

    Returns:
        shifted_spikes: list of length num_trials,
                        each a list of length num_neurons with shifted spike times.
    """
    num_trials = len(spikes_by_trial_abs)
    if num_trials == 0:
        return []

    num_neurons = len(spikes_by_trial_abs[0])
    shifted = []

    for j in range(num_trials):
        trial_list = []
        for i in range(num_neurons):
            sp_abs = np.asarray(spikes_by_trial_abs[j][i], dtype=float)
            if sp_abs.size == 0:
                trial_list.append(np.array([]))
                continue

            if np.isnan(latencies[i]):
                aligned = sp_abs - align_times[j]
            else:
                aligned = (sp_abs - align_times[j]) - latencies[i]

            trial_list.append(aligned)
        shifted.append(trial_list)

    return shifted


# ===================== Connectivity calculation ===================== #

def compute_rate_corr_from_shifted(spikes_shifted, window):
    """
    Compute log-rate correlation matrix from latency-compensated spike trains.

    For each neuron i and trial j:
        - Count spikes in the given "window" (relative to compensated time).
        - Convert to firing rate = spike_count / window_length.
        - Apply log(rate + EPS).
    Then compute Pearson correlation over neurons (corrcoef on log-rate).

    Args:
        spikes_shifted: list of length num_trials, each a list of length num_neurons
        window        : (start, end) in seconds

    Returns:
        corr_mat: numpy array (num_neurons, num_neurons)
    """
    num_trials = len(spikes_shifted)
    if num_trials == 0:
        return np.array([[]])

    num_neurons = len(spikes_shifted[0])
    win_len = window[1] - window[0]
    rate_mat = np.full((num_neurons, num_trials), np.nan)

    for i in range(num_neurons):
        for j in range(num_trials):
            sp = np.asarray(spikes_shifted[j][i])
            if sp.size == 0:
                rate_mat[i, j] = 0.0
                continue
            in_win = sp[(sp >= window[0]) & (sp <= window[1])]
            rate_mat[i, j] = len(in_win) / win_len

    log_rate = np.log(rate_mat + EPS)
    corr_mat = np.corrcoef(np.nan_to_num(log_rate))
    return corr_mat


def build_directed_adjacency(latencies, corr_mat, corr_thresh, min_lat_diff):
    """
    Build a directed adjacency matrix based on:
        - correlation strength
        - latency ordering

    We set a directed edge i -> j if:
        - corr_mat[i, j] >= corr_thresh
        - latencies[i] + min_lat_diff <= latencies[j]
        - neither latency is NaN

    Args:
        latencies    : array (num_neurons,)
        corr_mat     : array (num_neurons, num_neurons)
        corr_thresh  : scalar threshold for correlation
        min_lat_diff : minimum latency difference (seconds) to assign direction

    Returns:
        adj: integer array (num_neurons, num_neurons), 0/1
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

            if c >= corr_thresh and (li + min_lat_diff) <= lj:
                adj[i, j] = 1

    return adj


# ===================== Connectivity evaluation (scoring) ===================== #

def evaluate_connectivity(latencies, corr_mat, adj_mat, neuron_map):
    """
    Compute a few simple metrics to evaluate how "good" a connectivity result is.

    Current metrics:
        - num_edges          : total number of directed edges
        - density            : edge density
        - mean_edge_corr     : mean |correlation| for edges
        - feedforward_ratio  : fraction of edges that go "deep -> superficial",
                               assuming higher TT index = deeper.
        - score              : combined scalar score

    The score is a heuristic:
        score = 0.5 * feedforward_ratio
              + 0.3 * mean_edge_corr
              + 0.2 * density_score

    where density_score is a Gaussian penalty around a "target" density
    (we don't want networks that are too sparse or too dense).
    """
    num_neurons = len(neuron_map)
    if num_neurons == 0:
        return dict(
            score=0.0,
            num_edges=0,
            density=0.0,
            feedforward_ratio=0.0,
            mean_edge_corr=np.nan
        )

    num_edges = int(adj_mat.sum())
    max_edges = num_neurons * (num_neurons - 1)
    density = num_edges / max_edges if max_edges > 0 else 0.0

    if num_edges == 0:
        return dict(
            score=0.0,
            num_edges=0,
            density=density,
            feedforward_ratio=0.0,
            mean_edge_corr=np.nan
        )

    feedforward = 0
    edge_corr_vals = []

    for i in range(num_neurons):
        for j in range(num_neurons):
            if adj_mat[i, j] == 1:
                tt_i, _ = neuron_map[i]
                tt_j, _ = neuron_map[j]
                # "Deep -> superficial": higher TT index assumed deeper
                if tt_i > tt_j:
                    feedforward += 1
                edge_corr_vals.append(corr_mat[i, j])

    feedforward_ratio = feedforward / num_edges
    mean_edge_corr = float(np.nanmean(np.abs(edge_corr_vals)))

    # Density target and penalty: prefer moderate density (e.g., around 0.15).
    target_density = 0.15
    sigma_density = 0.10
    density_score = np.exp(-((density - target_density) ** 2) / (2 * sigma_density ** 2))

    score = (
        0.5 * feedforward_ratio +
        0.3 * mean_edge_corr +
        0.2 * density_score
    )

    return dict(
        score=float(score),
        num_edges=num_edges,
        density=float(density),
        feedforward_ratio=float(feedforward_ratio),
        mean_edge_corr=mean_edge_corr
    )


# ===================== Single-parameter-run wrapper ===================== #

def run_connectivity_once(
    spikes_by_trial_abs,
    stim_ons,
    neuron_map,
    lat_window,
    conn_window,
    corr_thresh,
    min_lat_diff
):
    """
    Run one full connectivity analysis with a given parameter set,
    without plotting or writing intermediate files.

    Returns:
        metrics: dict with score, metrics and all parameter values.
    """
    # 1) Latency estimation
    latencies = compute_neuron_latency(
        spikes_by_trial_abs,
        stim_ons,
        window=lat_window,
        min_trials=MIN_TRIALS
    )

    # 2) Latency compensation
    spikes_shifted = latency_shift_spikes_relative(
        spikes_by_trial_abs,
        stim_ons,
        latencies
    )

    # 3) Connectivity window correlation
    corr_mat = compute_rate_corr_from_shifted(
        spikes_shifted,
        window=conn_window
    )

    # 4) Directed adjacency
    adj_mat = build_directed_adjacency(
        latencies,
        corr_mat,
        corr_thresh=corr_thresh,
        min_lat_diff=min_lat_diff
    )

    # 5) Evaluate
    metrics = evaluate_connectivity(latencies, corr_mat, adj_mat, neuron_map)

    # Attach parameter values for later inspection
    metrics.update(dict(
        lat_window_start=lat_window[0],
        lat_window_end=lat_window[1],
        conn_window_start=conn_window[0],
        conn_window_end=conn_window[1],
        conn_bin_size=conn_window[1] - conn_window[0],
        corr_thresh=corr_thresh,
        min_lat_diff=min_lat_diff
    ))
    return metrics


# ===================== Session-level processing (for multiprocessing) ===================== #

def process_session(mat_fname):
    """
    Process a single .mat session:
        - Load data
        - Build neuron_map
        - Extract spikes and StimOn times
        - Scan parameter grid
        - Return a list of metrics dicts
    """
    session_name = os.path.splitext(mat_fname)[0]
    session_results = []

    print(f"[SCAN] Session: {mat_fname}")
    T = load_mat_session(os.path.join("./data", mat_fname))
    if len(T) == 0:
        print(f"[WARN] Empty session: {mat_fname}")
        return session_results

    neuron_map = build_neuron_map(T[0])
    if len(neuron_map) == 0:
        print(f"[WARN] No neurons found in session: {mat_fname}")
        return session_results

    # Collect spikes (absolute time) and StimOn times for all trials
    all_spikes_abs = []
    stim_ons = []
    for trial in T:
        stim_on = get_event_time(trial, STIM_ON_ID)
        if stim_on is None:
            continue
        spikes = extract_spike_times(trial, neuron_map)
        all_spikes_abs.append(spikes)
        stim_ons.append(stim_on)

    if len(all_spikes_abs) == 0:
        print(f"[WARN] No valid trials with StimOn in session: {mat_fname}")
        return session_results

    all_spikes_abs = np.array(all_spikes_abs, dtype=object)
    stim_ons = np.array(stim_ons)

    # Build parameter grid for this session
    param_grid = []
    for bin_size in CONN_BIN_SIZES:
        for start_t in CONN_START_TIMES:
            conn_window = (start_t, start_t + bin_size)
            for ct in CORR_THRESH_LIST:
                for ld in LAT_DIFF_LIST:
                    param_grid.append(dict(
                        conn_window=conn_window,
                        corr_thresh=ct,
                        min_lat_diff=ld
                    ))

    # Scan all parameter combinations
    for params in param_grid:
        metrics = run_connectivity_once(
            spikes_by_trial_abs=all_spikes_abs,
            stim_ons=stim_ons,
            neuron_map=neuron_map,
            lat_window=LAT_WINDOW,
            conn_window=params["conn_window"],
            corr_thresh=params["corr_thresh"],
            min_lat_diff=params["min_lat_diff"]
        )
        metrics["session"] = session_name
        session_results.append(metrics)

    print(f"[DONE] Session {mat_fname}: {len(session_results)} parameter sets evaluated.")
    return session_results


# ===================== Main: multi-process over sessions ===================== #

def main():
    # Find all .mat files in ./data
    mat_files = [f for f in os.listdir("./data") if f.endswith(".mat")]
    if not mat_files:
        print("[WARN] No .mat files found in ./data")
        return

    print(f"[INFO] Found {len(mat_files)} sessions: {mat_files}")

    all_results = []

    # Use multiprocessing over sessions (each session loads its .mat only once).
    # Adjust max_workers according to your CPU cores.
    max_workers = min(len(mat_files), os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_session, mf): mf for mf in mat_files}
        for fut in as_completed(futures):
            mf = futures[fut]
            try:
                session_res = fut.result()
                all_results.extend(session_res)
            except Exception as e:
                print(f"[ERROR] Session {mf} failed with error: {e}")

    if not all_results:
        print("[WARN] No results collected from any session.")
        return

    # Aggregate all results into a single DataFrame
    df = pd.DataFrame(all_results)
    df = df.sort_values(by=["score"], ascending=False)

    df.to_csv(SCAN_OUTPUT_CSV, index=False)
    print(f"[SAVE] Scan results saved to {SCAN_OUTPUT_CSV}")
    print(df.head(10))  # show top parameter sets


if __name__ == "__main__":
    main()