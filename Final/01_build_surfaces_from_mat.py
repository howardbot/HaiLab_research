import os
import re
import csv
import numpy as np

from src.Loader import load_mat_session

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "./data"
OUTPUT_DIR = "./out_binary_outer_surfaces_ON_only"

# Event IDs
STIM_ON_ID = 118
STIM_OFF_IDS = [120]

# -------- binary tiny-bin resolution --------
# this replaces the old inner-window binning scale
BINARY_RESOLUTION_LIST_SEC = [
    0.0005,   # 0.5 ms
    0.0010,   # 1.0 ms
    0.0020,   # 2.0 ms
]

# -------- lag range --------
# tau step will follow binary resolution
TAU_RANGE_SEC = (-0.05, 0.05)   # +-50 ms

# -------- outer sliding windows --------
OUTER_WIN_SEC = 0.200           # 200 ms
OUTER_STEP_SEC = 0.010          # 10 ms

# -------- analysis ranges --------
PRE_RANGE_SEC = (-0.600, 0.000)
ON_RANGE_SEC = (0.000, 1.000)
POST_RANGE_SEC = (0.000, 0.200)

# -------- filters --------
MIN_TRIALS = 10
MIN_RATE_HZ = 1.0
USE_UPPER_TRIANGLE = True
WRITE_NEURON_MAP_VALID = True

EPS = 1e-12


# =========================================================
# UTILS
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_name(s: str) -> str:
    s = os.path.splitext(os.path.basename(s))[0]
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s[:180] if len(s) > 180 else s


def fmt_ms(x_sec: float) -> str:
    return f"{x_sec * 1000.0:.3f}ms".replace(".", "p")


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =========================================================
# EVENT HELPERS
# =========================================================

def get_event_time_first(trial, eid):
    if not hasattr(trial, "EID") or not hasattr(trial, "EventT"):
        return None
    eids = np.atleast_1d(trial.EID)
    ts = np.atleast_1d(trial.EventT)
    idx = np.where(eids == eid)[0]
    if len(idx) == 0:
        return None
    return float(ts[idx[0]])


def get_event_time_first_of_many(trial, eids_list):
    for eid in eids_list:
        t = get_event_time_first(trial, eid)
        if t is not None:
            return t, eid
    return None, None


# =========================================================
# NEURON MAP / SPIKES
# =========================================================

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


# =========================================================
# TRIAL COVERAGE
# =========================================================

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
            tmin_ev = float(np.min(ev))
            tmax_ev = float(np.max(ev))

    if tmin_ev is None or tmax_ev is None:
        if neuron_map is None:
            return None, None
        sp_list = extract_spike_times(trial, neuron_map)
        all_sp = []
        for sp in sp_list:
            sp = np.atleast_1d(sp).astype(float)
            sp = sp[np.isfinite(sp)]
            if sp.size:
                all_sp.append(sp)
        if not all_sp:
            return None, None
        all_sp = np.concatenate(all_sp)
        return float(np.min(all_sp)), float(np.max(all_sp))

    t_min, t_max = tmin_ev, tmax_ev
    if neuron_map is not None:
        margin = 1.0
        lo = tmin_ev - margin
        hi = tmax_ev + margin
        sp_list = extract_spike_times(trial, neuron_map)
        sp_keep = []
        for sp in sp_list:
            sp = np.atleast_1d(sp).astype(float)
            sp = sp[np.isfinite(sp)]
            if sp.size:
                sp = sp[(sp >= lo) & (sp <= hi)]
                if sp.size:
                    sp_keep.append(sp)
        if sp_keep:
            all_sp = np.concatenate(sp_keep)
            t_min = float(min(t_min, np.min(all_sp)))
            t_max = float(max(t_max, np.max(all_sp)))
    return t_min, t_max


def window_span_abs(anchor_t, rel_window):
    return anchor_t + rel_window[0], anchor_t + rel_window[1]


def window_is_covered_strict(t_min, t_max, anchor_t, rel_window):
    if anchor_t is None or t_min is None or t_max is None:
        return False
    w0, w1 = window_span_abs(anchor_t, rel_window)
    return (w0 >= t_min - EPS) and (w1 <= t_max + EPS)


def build_coverage_mask(anchors, trial_bounds_abs, rel_window):
    mask = np.zeros(len(anchors), dtype=bool)
    for i, anchor_t in enumerate(anchors):
        t_min, t_max = trial_bounds_abs[i]
        mask[i] = window_is_covered_strict(t_min, t_max, anchor_t, rel_window)
    return mask


# =========================================================
# FILTER
# =========================================================

def preselect_neurons_by_rate(all_spikes_trials, anchors, rel_window, valid_trial_mask, min_rate_hz):
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


# =========================================================
# BINARY TRAINS
# =========================================================

def build_binary_trains(all_spikes_trials, anchors, rel_window, bin_res_sec, neuron_idx, valid_trial_mask):
    """
    Build binary trains in rel_window relative to anchor.

    Returns
    -------
    trains : shape (N, T_valid, K)
    starts : shape (K,)
    valid_trial_ids : original trial ids kept
    """
    w0, w1 = rel_window
    dur = w1 - w0
    if dur <= 0:
        return np.zeros((len(neuron_idx), 0, 0), dtype=np.float32), np.array([], dtype=float), np.array([], dtype=int)

    K = int(np.floor(dur / bin_res_sec))
    if K < 2:
        return np.zeros((len(neuron_idx), 0, 0), dtype=np.float32), np.array([], dtype=float), np.array([], dtype=int)

    starts = w0 + np.arange(K) * bin_res_sec
    valid_trial_ids = np.where(valid_trial_mask)[0]

    N = len(neuron_idx)
    T_valid = len(valid_trial_ids)
    trains = np.zeros((N, T_valid, K), dtype=np.float32)

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
            trains[ni, t_valid, idx] = 1.0

    return trains, starts, valid_trial_ids


# =========================================================
# CORRELATION
# =========================================================

def tau_grid(tau_range_sec, step_sec):
    t0, t1 = tau_range_sec
    if t0 > t1:
        t0, t1 = t1, t0
    k0 = int(np.ceil(t0 / step_sec))
    k1 = int(np.floor(t1 / step_sec))
    ks = np.arange(k0, k1 + 1, dtype=int)
    return ks * step_sec


def nan_corrcoef_flat(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return np.nan

    xx = x[mask]
    yy = y[mask]

    if np.std(xx) < EPS or np.std(yy) < EPS:
        return np.nan

    return float(np.corrcoef(xx, yy)[0, 1])


def corr_pair_at_lag(binary_i, binary_j, lag_steps):
    """
    binary_i, binary_j: shape (T_valid, K)
    lag > 0 means corr(i[t], j[t+lag])
    """
    _, K = binary_i.shape
    if abs(lag_steps) >= K:
        return np.nan

    if lag_steps > 0:
        a = binary_i[:, :K - lag_steps]
        b = binary_j[:, lag_steps:]
    elif lag_steps < 0:
        lag_abs = -lag_steps
        a = binary_i[:, lag_abs:]
        b = binary_j[:, :K - lag_abs]
    else:
        a = binary_i
        b = binary_j

    return nan_corrcoef_flat(a.reshape(-1), b.reshape(-1))


def build_pair_ij(N, upper_only=True):
    if upper_only:
        iu = np.triu_indices(N, k=1)
        return np.vstack([iu[0], iu[1]]).T.astype(int)
    ij = [(i, j) for i in range(N) for j in range(N) if i != j]
    return np.array(ij, dtype=int)


def compute_Z_from_binary(binary_trains, pair_ij, taus_sec, bin_res_sec):
    """
    binary_trains: (N, T_valid, K)
    Z: (P, Ntau)
    """
    N, _, K = binary_trains.shape
    P = pair_ij.shape[0]
    Z = np.full((P, len(taus_sec)), np.nan, dtype=np.float32)

    views = [binary_trains[i] for i in range(N)]

    for ti, tau in enumerate(taus_sec):
        lag_steps = int(np.round(tau / bin_res_sec))
        if abs(lag_steps) >= K:
            continue

        for p in range(P):
            i, j = pair_ij[p]
            Z[p, ti] = corr_pair_at_lag(views[i], views[j], lag_steps)

    return Z


# =========================================================
# SAVE
# =========================================================

def save_surface_out(out_dir, Z, tau_ms, pair_ij):
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "Z.npy"), Z.astype(np.float32))
    np.save(os.path.join(out_dir, "tau_ms.npy"), tau_ms.astype(np.float32))
    np.save(os.path.join(out_dir, "pair_ij.npy"), pair_ij.astype(np.int32))


def save_fixed_window_meta(epoch_root, rel_window, anchor_name, n_trials):
    rows = [{
        "anchor": anchor_name,
        "t0_sec": rel_window[0],
        "t1_sec": rel_window[1],
        "window_sec": rel_window[1] - rel_window[0],
        "n_trials": n_trials,
    }]
    write_csv(
        os.path.join(epoch_root, "fixed_window.csv"),
        rows,
        ["anchor", "t0_sec", "t1_sec", "window_sec", "n_trials"],
    )


# =========================================================
# ANALYSIS
# =========================================================

def analyze_fixed_epoch(
    session_root,
    epoch_dir,
    anchor_name,
    all_spikes_trials,
    anchors,
    rel_window,
    valid_trial_mask,
    valid_neurons,
    pair_ij,
):
    epoch_root = os.path.join(session_root, epoch_dir)
    ensure_dir(epoch_root)

    n_trials = int(np.sum(valid_trial_mask))
    if n_trials < MIN_TRIALS:
        print(f"    [SKIP] {epoch_dir}: too few covered trials ({n_trials})")
        return

    save_fixed_window_meta(epoch_root, rel_window, anchor_name, n_trials)

    for bin_res_sec in BINARY_RESOLUTION_LIST_SEC:
        res_tag = f"binary_{fmt_ms(bin_res_sec)}"
        out_dir = os.path.join(epoch_root, res_tag, "fixed_window", "surface_out")

        binary_trains, _, _ = build_binary_trains(
            all_spikes_trials=all_spikes_trials,
            anchors=anchors,
            rel_window=rel_window,
            bin_res_sec=bin_res_sec,
            neuron_idx=valid_neurons,
            valid_trial_mask=valid_trial_mask,
        )

        if binary_trains.shape[1] < MIN_TRIALS or binary_trains.shape[2] < 2:
            print(f"    [SKIP] {epoch_dir}/{res_tag}: insufficient trials or bins")
            continue

        taus_sec = tau_grid(TAU_RANGE_SEC, bin_res_sec)
        tau_ms = taus_sec * 1000.0
        Z = compute_Z_from_binary(binary_trains, pair_ij, taus_sec, bin_res_sec)
        save_surface_out(out_dir, Z, tau_ms, pair_ij)

    print(f"    [OK] {epoch_dir}: fixed window {rel_window[0]:.3f}..{rel_window[1]:.3f}s | trials={n_trials}")


def analyze_on_epoch(
    session_root,
    all_spikes_trials,
    stim_on_anchors,
    on_mask,
    valid_neurons,
    pair_ij,
):
    epoch_root = os.path.join(session_root, "ON_stimOnAnchor")
    ensure_dir(epoch_root)

    t_start, t_end = ON_RANGE_SEC
    last_start = t_end - OUTER_WIN_SEC + 1e-12

    if last_start < t_start:
        print("    [SKIP] ON range shorter than outer window")
        return

    if int(np.sum(on_mask)) < MIN_TRIALS:
        print(f"    [SKIP] ON: too few covered trials ({int(np.sum(on_mask))})")
        return

    outer_rows = []
    outer_idx = 0
    t0 = t_start

    while t0 <= last_start:
        rel_window = (t0, t0 + OUTER_WIN_SEC)

        for bin_res_sec in BINARY_RESOLUTION_LIST_SEC:
            res_tag = f"binary_{fmt_ms(bin_res_sec)}"
            out_dir = os.path.join(
                epoch_root,
                res_tag,
                f"outer_{outer_idx:04d}",
                "surface_out"
            )

            binary_trains, _, _ = build_binary_trains(
                all_spikes_trials=all_spikes_trials,
                anchors=stim_on_anchors,
                rel_window=rel_window,
                bin_res_sec=bin_res_sec,
                neuron_idx=valid_neurons,
                valid_trial_mask=on_mask,
            )

            if binary_trains.shape[1] < MIN_TRIALS or binary_trains.shape[2] < 2:
                continue

            taus_sec = tau_grid(TAU_RANGE_SEC, bin_res_sec)
            tau_ms = taus_sec * 1000.0
            Z = compute_Z_from_binary(binary_trains, pair_ij, taus_sec, bin_res_sec)

            save_surface_out(out_dir, Z, tau_ms, pair_ij)

        outer_rows.append({
            "outer_idx": outer_idx,
            "t0_sec": t0,
            "t1_sec": t0 + OUTER_WIN_SEC,
            "W_out_sec": OUTER_WIN_SEC,
            "step_sec": OUTER_STEP_SEC,
            "n_trials": int(np.sum(on_mask)),
        })

        outer_idx += 1
        t0 += OUTER_STEP_SEC

    write_csv(
        os.path.join(epoch_root, "outer_index.csv"),
        outer_rows,
        ["outer_idx", "t0_sec", "t1_sec", "W_out_sec", "step_sec", "n_trials"]
    )

    print(f"    [OK] ON: {outer_idx} outer windows | trials={int(np.sum(on_mask))}")


# =========================================================
# SESSION ANALYSIS
# =========================================================

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

    all_spikes_trials = []
    stim_on_anchors = []
    stim_off_anchors = []
    trial_bounds_abs = []

    for trial in T:
        stim_on = get_event_time_first(trial, STIM_ON_ID)
        if stim_on is None:
            continue

        stim_off, _ = get_event_time_first_of_many(trial, STIM_OFF_IDS)
        all_spikes_trials.append(extract_spike_times(trial, neuron_map))
        stim_on_anchors.append(stim_on)
        stim_off_anchors.append(np.nan if stim_off is None else stim_off)
        trial_bounds_abs.append(infer_trial_time_bounds_robust(trial, neuron_map=neuron_map))

    if len(all_spikes_trials) < MIN_TRIALS:
        print(f"  [SKIP] too few trials with StimOn: {len(all_spikes_trials)}")
        return

    stim_on_anchors = np.asarray(stim_on_anchors, dtype=float)
    stim_off_anchors = np.asarray(stim_off_anchors, dtype=float)
    trial_bounds_abs = np.asarray(trial_bounds_abs, dtype=object)

    pre_mask = build_coverage_mask(stim_on_anchors, trial_bounds_abs, PRE_RANGE_SEC)
    on_mask = build_coverage_mask(stim_on_anchors, trial_bounds_abs, ON_RANGE_SEC)
    post_mask = build_coverage_mask(stim_off_anchors, trial_bounds_abs, POST_RANGE_SEC)

    print(
        "  covered trials | "
        f"PRE={int(np.sum(pre_mask))} "
        f"ON={int(np.sum(on_mask))} "
        f"POST={int(np.sum(post_mask))}"
    )

    if int(np.sum(on_mask)) < MIN_TRIALS:
        print("  [SKIP] too few ON-covered trials for neuron filtering")
        return

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

    session_root = os.path.join(OUTPUT_DIR, session_name)
    ensure_dir(session_root)

    if WRITE_NEURON_MAP_VALID:
        rows = []
        for k, orig_idx in enumerate(valid_neurons):
            tt, unit = neuron_map[orig_idx]
            rows.append({
                "k_in_valid": k,
                "orig_neuron_idx": orig_idx,
                "tt": tt,
                "unit": unit,
                "label": f"n{orig_idx}_TT{tt}u{unit}"
            })

        write_csv(
            os.path.join(session_root, "neuron_map_valid.csv"),
            rows,
            ["k_in_valid", "orig_neuron_idx", "tt", "unit", "label"]
        )

    N = len(valid_neurons)
    pair_ij = build_pair_ij(N, upper_only=USE_UPPER_TRIANGLE)

    analyze_fixed_epoch(
        session_root=session_root,
        epoch_dir="PRE_stimOnAnchor",
        anchor_name="StimOn(118)",
        all_spikes_trials=all_spikes_trials,
        anchors=stim_on_anchors,
        rel_window=PRE_RANGE_SEC,
        valid_trial_mask=pre_mask,
        valid_neurons=valid_neurons,
        pair_ij=pair_ij,
    )

    analyze_on_epoch(
        session_root=session_root,
        all_spikes_trials=all_spikes_trials,
        stim_on_anchors=stim_on_anchors,
        on_mask=on_mask,
        valid_neurons=valid_neurons,
        pair_ij=pair_ij,
    )

    analyze_fixed_epoch(
        session_root=session_root,
        epoch_dir="POST_stimOffAnchor",
        anchor_name="StimOff(120)",
        all_spikes_trials=all_spikes_trials,
        anchors=stim_off_anchors,
        rel_window=POST_RANGE_SEC,
        valid_trial_mask=post_mask,
        valid_neurons=valid_neurons,
        pair_ij=pair_ij,
    )


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dir(OUTPUT_DIR)

    mats = [f for f in os.listdir(DATA_DIR) if f.endswith(".mat")]
    if not mats:
        print(f"[WARN] no .mat files found in {DATA_DIR}")
        return

    print(f"[INFO] found {len(mats)} sessions")
    for fname in sorted(mats):
        analyze_one_session(os.path.join(DATA_DIR, fname))

    print("\n[DONE]")


if __name__ == "__main__":
    main()
