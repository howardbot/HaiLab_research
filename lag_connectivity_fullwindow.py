import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from src.Loader import load_mat_session  # your existing loader


# ===================== Basic helpers (same style as your code) ===================== #

def build_neuron_map(trial):
    """
    Build neuron_map: list of (tt, unit_idx) for this session.
    tt: electrode index (1..8 or similar)
    unit_idx: unit index under this tetrode (0-based)
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
    """
    Get the time of a given event ID from trial (return None if not present).
    """
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids = np.atleast_1d(trial.EID)
    times = np.atleast_1d(trial.EventT)
    idx = np.where(eids == eid)[0]
    return times[idx[0]] if len(idx) > 0 else None


# ===================== Global constants for Method A (early window) ===================== #

STIM_ON_ID   = 118          # Stimulus ON

EARLY_WINDOW = (0.0, 0.2)   # fixed analysis window: 0–200 ms after StimOn

BIN_LIST     = [0.003, 0.005, 0.010]  # bin sizes in seconds: 3, 5, 10 ms

TAU_MAX_SEC  = 0.05         # scan lags in [-50 ms, +50 ms]

CORR_THRESH  = 0.15          # minimal correlation to define a connection
TOPK_OUT     = 3            # at most K outgoing edges per neuron (None = no limit)

MIN_RATE_HZ  = 1.0          # neuron must have mean firing rate >= 1 Hz in EARLY_WINDOW
MIN_TRIALS   = 10           # minimal #trials with StimOn to attempt connectivity

# density range considered "reasonable"
DENSITY_MIN  = 0.02         # 2%
DENSITY_MAX  = 0.25         # 25%

EPS          = 1e-12

OUTPUT_DIR   = "neuron_Connectivity_lagged_Early"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 1. Pre-select neurons by firing rate in EARLY_WINDOW ===================== #

def preselect_neurons_by_rate_window(all_spikes_abs, stim_ons,
                                     window=EARLY_WINDOW,
                                     min_rate_hz=MIN_RATE_HZ):
    """
    Pre-select neurons based on mean firing rate within a fixed window
    relative to StimOn, across all trials.

    all_spikes_abs[j][i] : absolute spike times of neuron i in trial j
    stim_ons[j]          : absolute StimOn time in trial j

    Returns:
        valid_neuron_idx : list[int] of neuron indices that pass the rate filter
        mean_rates_hz    : array[num_neurons] of mean firing rates (Hz)
    """
    num_trials = len(all_spikes_abs)
    if num_trials == 0:
        return [], np.array([])

    num_neurons = len(all_spikes_abs[0])
    total_time = num_trials * (window[1] - window[0])
    mean_rates = np.zeros(num_neurons, dtype=float)

    for i in range(num_neurons):
        total_spikes = 0
        for j in range(num_trials):
            sp = np.asarray(all_spikes_abs[j][i]) - stim_ons[j]
            in_win = (sp >= window[0]) & (sp <= window[1])
            total_spikes += in_win.sum()
        mean_rates[i] = total_spikes / max(EPS, total_time)

    valid_idx = [i for i in range(num_neurons) if mean_rates[i] >= min_rate_hz]
    return valid_idx, mean_rates


# ===================== 2. Build binned spike-count tensor in EARLY_WINDOW ===================== #

def build_binned_rates_window(all_spikes_abs, stim_ons,
                              window=EARLY_WINDOW,
                              bin_size=0.005,
                              neuron_idx=None):
    """
    Convert spike times into a 3D tensor of spike counts (neuron, trial, time_bin)
    over a fixed window [window[0], window[1]] relative to StimOn.

    Inputs:
        all_spikes_abs : list[trials][neurons] -> absolute spike times
        stim_ons       : array[trials] -> absolute StimOn times
        window         : (t_start, t_end) relative to StimOn
        bin_size       : bin size in seconds
        neuron_idx     : if not None, only use these neuron indices

    Returns:
        rates        : shape (N, T, B) spike counts
        t_edges      : array of bin boundaries of length B+1
        used_neurons : list of neuron indices corresponding to axis 0 of 'rates'
    """
    num_trials = len(all_spikes_abs)
    if num_trials == 0:
        return np.zeros((0, 0, 0)), np.array([]), []

    num_neurons_total = len(all_spikes_abs[0])
    if neuron_idx is None:
        neuron_idx = list(range(num_neurons_total))

    # time bins from window[0] to window[1]
    t_edges = np.arange(window[0],
                        window[1] + bin_size + 1e-9,
                        bin_size)
    num_bins = len(t_edges) - 1

    N = len(neuron_idx)
    rates = np.zeros((N, num_trials, num_bins), dtype=float)

    for ni, i in enumerate(neuron_idx):
        for j in range(num_trials):
            sp = np.asarray(all_spikes_abs[j][i]) - stim_ons[j]
            # restrict to the analysis window
            sp = sp[(sp >= window[0]) & (sp <= window[1])]
            if sp.size == 0:
                continue
            counts, _ = np.histogram(sp, bins=t_edges)
            rates[ni, j, :] = counts

    return rates, t_edges, neuron_idx


# ===================== 3. Lagged correlation in EARLY_WINDOW ===================== #

def corr_at_lag_flat(Xi, Xj, lag_bins):
    """
    Compute Pearson correlation between Xi and Xj at a given lag (in bins).

    Xi, Xj : shape (num_trials, num_bins)
    lag_bins : integer lag ( >0 means Xj is shifted forward in time )

    Returns:
        corr (float) or np.nan if not defined.
    """
    num_trials, B = Xi.shape
    if abs(lag_bins) >= B:
        return np.nan

    if lag_bins >= 0:
        X1 = Xi[:, :B - lag_bins]
        X2 = Xj[:, lag_bins:]
    else:
        L = -lag_bins
        X1 = Xi[:, L:]
        X2 = Xj[:, :B - L]

    x = X1.reshape(-1)
    y = X2.reshape(-1)

    if x.std() < EPS or y.std() < EPS:
        return np.nan

    c = np.corrcoef(x, y)[0, 1]
    return c


def scan_lagged_corr_window(rates, bin_size, tau_max_sec=TAU_MAX_SEC):
    """
    For each neuron pair, scan a set of lags across the fixed window,
    and find the best correlation and corresponding lag.

    The lag grid is chosen as integer multiples of bin_size:
        tau_k = k * bin_size, where |tau_k| <= tau_max_sec.

    Inputs:
        rates       : shape (N, T, B) spike-count tensor
        bin_size    : bin size in seconds
        tau_max_sec : maximum absolute lag to scan (seconds)

    Returns:
        best_corr : shape (N, N) maximum correlation for each pair (i, j)
        best_tau  : shape (N, N) lag (in seconds) at which best_corr is achieved
    """
    N, T, B = rates.shape
    if B <= 1:
        return np.full((N, N), np.nan), np.zeros((N, N))

    # maximum lag in bins
    max_lag_bins = int(np.floor(tau_max_sec / bin_size))
    if max_lag_bins < 1:
        return np.full((N, N), np.nan), np.zeros((N, N))

    lag_bins_list = np.arange(-max_lag_bins, max_lag_bins + 1, 1)
    lag_secs_list = lag_bins_list * bin_size

    best_corr = np.full((N, N), np.nan, dtype=float)
    best_tau = np.zeros((N, N), dtype=float)

    for i in range(N):
        Xi = rates[i]  # shape (T, B)
        if Xi.sum() == 0:
            continue
        for j in range(N):
            if i == j:
                continue
            Xj = rates[j]
            if Xj.sum() == 0:
                continue

            best_c = -np.inf
            best_t = 0.0

            for lag_bins, lag_sec in zip(lag_bins_list, lag_secs_list):
                c = corr_at_lag_flat(Xi, Xj, lag_bins)
                if np.isnan(c):
                    continue
                if c > best_c:
                    best_c = c
                    best_t = lag_sec

            if best_c > -np.inf:
                best_corr[i, j] = best_c
                best_tau[i, j] = best_t

    return best_corr, best_tau


# ===================== 4. Build directed adjacency with bin-dependent lag threshold ===================== #

def build_adjacency_from_best(best_corr,
                              best_tau,
                              corr_thresh,
                              lag_thresh_sec,
                              topk_out=TOPK_OUT):
    """
    Build a directed adjacency matrix from best_corr and best_tau (single window).

    Rule:
        For each unordered pair {i, j} (i < j):
            - Consider (i, j) and (j, i) entries.
            - Pick the direction with higher correlation.
            - If corr >= corr_thresh and |tau| >= lag_thresh_sec, keep a directed edge.

        Direction:
            Assuming best_corr[i, j] corresponds to Corr( Xi(t), Xj(t+tau) ):
            - If tau > 0: i leads j  => i -> j
            - If tau < 0: j leads i  => j -> i

        If topk_out is not None:
            - For each neuron i, keep at most topk_out strongest outgoing edges.

    Returns:
        adj      : shape (N, N), 0/1 adjacency matrix
        weight   : shape (N, N), edge weights (= correlation), 0 if no edge
        dir_tau  : shape (N, N), effective delay (s) for each directed edge
    """
    N = best_corr.shape[0]
    adj = np.zeros((N, N), dtype=int)
    weight = np.zeros((N, N), dtype=float)
    dir_tau = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i + 1, N):
            c_ij = best_corr[i, j]
            t_ij = best_tau[i, j]
            c_ji = best_corr[j, i]
            t_ji = best_tau[j, i]

            # If both are NaN, skip
            if np.isnan(c_ij) and np.isnan(c_ji):
                continue

            # Choose the direction with larger correlation
            if np.isnan(c_ij):
                c_sel, t_sel, src, dst = c_ji, t_ji, j, i
            elif np.isnan(c_ji):
                c_sel, t_sel, src, dst = c_ij, t_ij, i, j
            else:
                if c_ij >= c_ji:
                    c_sel, t_sel, src, dst = c_ij, t_ij, i, j
                else:
                    c_sel, t_sel, src, dst = c_ji, t_ji, j, i

            if c_sel < corr_thresh or abs(t_sel) < lag_thresh_sec:
                continue

            # Decide actual direction based on sign of tau
            if t_sel > 0:
                eff_src, eff_dst = src, dst
                eff_tau = t_sel  # positive delay
            else:
                eff_src, eff_dst = dst, src
                eff_tau = -t_sel  # store positive magnitude

            adj[eff_src, eff_dst] = 1
            weight[eff_src, eff_dst] = c_sel
            dir_tau[eff_src, eff_dst] = eff_tau

    # Optional: limit out-degree to topk_out strongest edges per neuron
    if topk_out is not None and topk_out > 0:
        for i in range(N):
            out_idxs = np.where(adj[i, :] == 1)[0]
            if len(out_idxs) <= topk_out:
                continue
            wts = weight[i, out_idxs]
            order = np.argsort(-wts)  # descending
            keep = out_idxs[order[:topk_out]]
            drop = set(out_idxs) - set(keep)
            for j in drop:
                adj[i, j] = 0
                weight[i, j] = 0.0
                dir_tau[i, j] = 0.0

    return adj, weight, dir_tau


# ===================== 5. Draw directed graph (vertical by original neuron index) ===================== #

def draw_directed_graph_by_index(neuron_map_valid,
                                 used_neurons,
                                 adj_mat,
                                 weight_mat,
                                 title,
                                 fname):
    """
    Professional-style neural connectivity graph:

        - spring_layout with larger repulsion (spread out nodes)
        - arrowheads avoid overlapping nodes (min_*_margin)
        - curved edges for clarity
        - nodes colored by TT index
        - edges标注 corr 值
    """
    N = len(neuron_map_valid)
    if N == 0:
        print("[WARN] No valid neurons to draw.")
        return

    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i)

    # Collect edges & weights
    edges = []
    weights = []
    for i in range(N):
        for j in range(N):
            if adj_mat[i, j] == 1:
                edges.append((i, j))
                weights.append(weight_mat[i, j])

    # === 1. NODE LAYOUT ===
    k = 2.0 / np.sqrt(max(N, 1))  # repulsion strength
    pos = nx.spring_layout(G, k=k, iterations=300, seed=42)

    # === 2. Node color: by TT ===
    tt_list = [tt for (tt, unit) in neuron_map_valid]
    unique_tt = np.unique(tt_list)
    cmap = plt.get_cmap("tab10")
    tt_to_idx = {tt: i for i, tt in enumerate(unique_tt)}
    node_colors = [cmap(tt_to_idx[tt] % cmap.N) for tt in tt_list]

    # === 3. Node labels ===
    labels = {}
    for i in range(N):
        tt, unit = neuron_map_valid[i]
        orig = used_neurons[i]
        labels[i] = f"n{orig}\nTT{tt}u{unit}"

    plt.figure(figsize=(8, max(6, N * 0.6)))
    ax = plt.gca()

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=900,
        edgecolors="black",
        linewidths=0.8,
        ax=ax
    )

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=9,
        font_weight="bold",
        ax=ax
    )

    # === 4. Draw edges & widths ===
    if edges:
        widths = []
        for w in weights:
            w_clipped = max(CORR_THRESH, w)
            widths.append(0.6 + 3.0 * (w_clipped - CORR_THRESH) / (1 - CORR_THRESH))

        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            width=widths,
            arrowstyle='-|>',
            arrowsize=18,
            ax=ax,
            connectionstyle="arc3,rad=0.2",
            min_source_margin=25,
            min_target_margin=25
        )

        # === 5. Edge labels: corr values ===
        edge_labels = {}
        for (i, j) in edges:
            c = weight_mat[i, j]
            edge_labels[(i, j)] = f"{c:.2f}"  # 保留两位小数

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,  # 中点
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"),
            ax=ax
        )
    else:
        print("[INFO] No edges to draw for this graph.")

    # === 6. Legend for TT colors ===
    from matplotlib.lines import Line2D
    legend_elems = []
    for tt, idx in tt_to_idx.items():
        legend_elems.append(Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=10,
            markerfacecolor=cmap(idx % cmap.N),
            markeredgecolor="black",
            label=f"TT{tt}"
        ))
    if legend_elems:
        ax.legend(handles=legend_elems,
                  title="Tetrode",
                  fontsize=8,
                  loc="upper right")

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[SAVE] Connectivity graph -> {fname}")



# ===================== 6. Evaluate network & score ===================== #

def evaluate_network(adj, weight):
    """
    Compute simple metrics for a directed network:

        adj    : (N, N) 0/1 adjacency matrix
        weight : (N, N) edge weights (correlation)

    Returns:
        E      : number of edges
        density: edge density
        W_mean : mean edge weight (over existing edges), or 0 if no edges
    """
    N = adj.shape[0]
    if N <= 1:
        return 0, 0.0, 0.0

    E = int(adj.sum())
    density = E / (N * (N - 1)) if N > 1 else 0.0

    if E > 0:
        W_mean = weight[adj == 1].mean()
    else:
        W_mean = 0.0

    return E, density, W_mean


def compute_score(E, density, W_mean,
                  density_min=DENSITY_MIN,
                  density_max=DENSITY_MAX):
    """
    Simple scoring function for a candidate network:

        - If density is outside [density_min, density_max], return -inf.
        - Otherwise, score = W_mean * log(E + 1).

    Encourages:
        - Reasonable edge count (not too sparse, not too dense).
        - Strong average correlations.
    """
    if E == 0:
        return -np.inf
    ### DEBUG
    #if density < density_min or density > density_max:
     ##   return -np.inf
    return float(W_mean * np.log(E + 1.0))


# ===================== 7. Full pipeline for one session (fixed early window) ===================== #

def analyze_early_lagged_for_session(all_spikes_abs,
                                     stim_ons,
                                     neuron_map,
                                     session_name,
                                     output_dir=OUTPUT_DIR):
    """
    Full pipeline for a single session, using a fixed early window [0, 200 ms]:

        1. Use all trials that have StimOn.
        2. Pre-select neurons by mean firing rate in EARLY_WINDOW.
        3. For each candidate bin size:
             a) Build binned spike-count tensor.
             b) Run lagged correlation.
             c) Build directed adjacency with:
                    - corr_thresh = CORR_THRESH
                    - lag_thresh_sec = 1 * bin_size
             d) Evaluate network and compute score.
        4. Choose the bin size with the highest score.
        5. Draw and save only that best graph for this session.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_trials = len(all_spikes_abs)
    num_neurons = len(neuron_map)
    print(f"\n[Early Lagged Connectivity] {session_name}")
    print(f"  trials with StimOn: {num_trials}")
    print(f"  neurons: {num_neurons}")

    if num_trials < MIN_TRIALS:
        print(f"[WARN] Too few trials (<{MIN_TRIALS}), skip session.")
        return

    # 1) Pre-select neurons by firing rate in EARLY_WINDOW
    valid_neuron_idx, mean_rates = preselect_neurons_by_rate_window(
        all_spikes_abs, stim_ons,
        window=EARLY_WINDOW,
        min_rate_hz=MIN_RATE_HZ
    )
    print(f"  Valid neurons after rate filter: {len(valid_neuron_idx)} / {num_neurons}")

    if len(valid_neuron_idx) < 2:
        print("[WARN] Too few neurons after filtering, skip session.")
        return

    best_score = -np.inf
    best_result = None  # (bin_size, adj, weight, used_neurons)

    for bin_size in BIN_LIST:
        print(f"    [BIN SCAN] bin_size = {bin_size*1000:.1f} ms")

        # 3a) Build binned spike-count tensor
        rates, t_edges, used_neurons = build_binned_rates_window(
            all_spikes_abs,
            stim_ons,
            window=EARLY_WINDOW,
            bin_size=bin_size,
            neuron_idx=valid_neuron_idx
        )

        N, T, B = rates.shape
        print(f"      rates shape: N={N}, T={T}, B={B}")
        if B <= 1:
            print("      [SKIP] Not enough time bins for lagged corr.")
            continue

        # 3b) Run lagged correlation
        best_corr, best_tau = scan_lagged_corr_window(
            rates,
            bin_size,
            tau_max_sec=TAU_MAX_SEC
        )

        # 3c) Build directed adjacency.
        #     Here lag_thresh_sec = 1 * bin_size: at least one bin offset to claim direction.
        lag_thresh_sec = bin_size
        adj, weight, dir_tau = build_adjacency_from_best(
            best_corr,
            best_tau,
            corr_thresh=CORR_THRESH,
            lag_thresh_sec=lag_thresh_sec,
            topk_out=TOPK_OUT
        )

        # 3d) Evaluate network and compute score
        E, density, W_mean = evaluate_network(adj, weight)
        score = compute_score(E, density, W_mean,
                              density_min=DENSITY_MIN,
                              density_max=DENSITY_MAX)

        print(f"      edges={E}, density={density:.4f}, W_mean={W_mean:.3f}, score={score:.4f}")

        if score > best_score:
            best_score = score
            best_result = (bin_size, adj, weight, used_neurons)

    if best_result is None or best_score == -np.inf:
        print("[WARN] No suitable network found for any bin size, skip drawing.")
        return

    # Unpack best result
    best_bin_size, best_adj, best_weight, used_neurons = best_result
    print(f"  >>> Best bin_size = {best_bin_size*1000:.1f} ms, score = {best_score:.4f}")

    neuron_map_valid = [neuron_map[i] for i in used_neurons]

    base = os.path.splitext(session_name)[0]
    fig_path = os.path.join(output_dir,
                            f"graph_EarlyLagged_best_{base}.png")
    title = (f"Early lagged connectivity "
             f"({EARLY_WINDOW[0]*1000:.0f}–{EARLY_WINDOW[1]*1000:.0f} ms, "
             f"bin={best_bin_size*1000:.1f} ms, |τ|≤{TAU_MAX_SEC*1000:.0f} ms)\n{base}")
    draw_directed_graph_by_index(
        neuron_map_valid,
        used_neurons,
        best_adj,
        best_weight,
        title,
        fig_path
    )


# ===================== 8. Session-level wrapper over your .mat files ===================== #

def analyze_file_early(fname):
    """
    High-level wrapper for one .mat session:

        - Load session.
        - Build neuron_map from the first trial.
        - Extract spike times and StimOn(118) times across all trials.
        - Run the early-window lagged-connectivity pipeline.
    """
    print(f"\n[Analyze Early-lagged connectivity] {fname}")
    T = load_mat_session(os.path.join("./data", fname))
    if len(T) == 0:
        print("[WARN] Empty session")
        return

    neuron_map = build_neuron_map(T[0])
    if len(neuron_map) == 0:
        print("[WARN] No neurons found in this session.")
        return

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

    analyze_early_lagged_for_session(
        all_spikes_abs,
        stim_ons,
        neuron_map,
        session_name=fname,
        output_dir=OUTPUT_DIR
    )


def main():
    files = [f for f in os.listdir("./data") if f.endswith(".mat")]
    if not files:
        print("[WARN] No .mat files found in ./data")
        return
    for f in files:
        analyze_file_early(f)


if __name__ == "__main__":
    main()
