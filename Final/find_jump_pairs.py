import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

ROOT_DIR = r"C:\Users\14478\Downloads\res\out_surface"
OUT_ROOT = os.path.join(ROOT_DIR, "_jump_analysis_all")

# session name filter
AREA_KEYS = ["CIP", "V3A"]

# bin folders to include; if None -> auto-detect all bin_xxxms
BIN_FOLDERS = None

# score mode
TAU_MODE = "near_zero"   # "near_zero" or "all"
TAU_ABS_MS = 10.0        # only used if TAU_MODE == "near_zero"

# normalization
USE_GLOBAL_RESIDUAL = True

# filtering thresholds
MIN_PEAK_AFTER = 0.10        # after jump peak（residual score）
MIN_MEAN_AFTER_3 = 0.05      # peaks after jump avg
MIN_PERSIST_COUNT = 2        # number of after jump corr pass threshold
PERSIST_THRESH = 0.05        # persist threshold on residual score

# top results
TOP_K_GLOBAL = 200
TOP_K_PER_COMBO = 30

# plotting
PLOT_TOP_PER_COMBO = 5

# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_csv_dict(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

def safe_int(x, default=-1):
    try:
        return int(x)
    except:
        return default

def sanitize_name(s):
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)

def detect_sessions(root_dir):
    sessions = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue
        up = name.upper()
        if any(k.upper() in up for k in AREA_KEYS):
            sessions.append(name)
    return sessions

def detect_bins(on_root):
    bins = []
    for name in sorted(os.listdir(on_root)):
        p = os.path.join(on_root, name)
        if os.path.isdir(p) and name.startswith("bin_"):
            bins.append(name)
    return bins

def load_outer_index(path):
    rows = []
    for row in load_csv_dict(path):
        rows.append({
            "outer_idx": safe_int(row["outer_idx"]),
            "t0_sec": safe_float(row["t0_sec"]),
            "t1_sec": safe_float(row["t1_sec"]),
            "W_out_sec": safe_float(row.get("W_out_sec", np.nan)),
            "step_sec": safe_float(row.get("step_sec", np.nan)),
        })
    return rows

def load_neuron_map(path):
    if not os.path.exists(path):
        return None
    return load_csv_dict(path)

def safe_label(neuron_map, idx):
    if neuron_map is None:
        return f"n{idx}"
    if idx < 0 or idx >= len(neuron_map):
        return f"n{idx}"
    return neuron_map[idx].get("label", f"n{idx}")

def get_first_valid_surface(bin_root, outer_table):
    for row in outer_table:
        idx = row["outer_idx"]
        sdir = os.path.join(bin_root, f"outer_{idx:04d}", "surface_out")
        z_path = os.path.join(sdir, "Z.npy")
        tau_path = os.path.join(sdir, "tau_ms.npy")
        pair_path = os.path.join(sdir, "pair_ij.npy")
        if os.path.exists(z_path) and os.path.exists(tau_path) and os.path.exists(pair_path):
            return sdir
    return None

def contiguous_run_length_from(mask, start_idx):
    """Count consecutive True starting at start_idx."""
    n = 0
    for k in range(start_idx, len(mask)):
        if mask[k]:
            n += 1
        else:
            break
    return n

# =========================================================
# CORE ANALYSIS FOR ONE (SESSION, BIN)
# =========================================================

def analyze_one_bin(session_name, session_root, bin_name):
    on_root = os.path.join(session_root, "on_1000ms")
    bin_root = os.path.join(on_root, bin_name)
    outer_index_path = os.path.join(on_root, "outer_index.csv")
    neuron_map_path = os.path.join(session_root, "neuron_map_valid.csv")

    if not os.path.exists(outer_index_path):
        print(f"[WARN] Missing outer_index.csv: {outer_index_path}")
        return None

    outer_table = load_outer_index(outer_index_path)
    neuron_map = load_neuron_map(neuron_map_path)

    first_surface = get_first_valid_surface(bin_root, outer_table)
    if first_surface is None:
        print(f"[WARN] No valid surface found for {session_name} | {bin_name}")
        return None

    tau_ms = np.load(os.path.join(first_surface, "tau_ms.npy"))
    pair_ij = np.load(os.path.join(first_surface, "pair_ij.npy"))

    if TAU_MODE == "near_zero":
        tau_mask = np.abs(tau_ms) <= TAU_ABS_MS
    else:
        tau_mask = np.ones_like(tau_ms, dtype=bool)

    if np.sum(tau_mask) == 0:
        print(f"[WARN] Empty tau mask for {session_name} | {bin_name}")
        return None

    # Build score matrix S[pair, outer]
    S_list = []
    valid_outer = []

    for row in outer_table:
        idx = row["outer_idx"]
        z_path = os.path.join(bin_root, f"outer_{idx:04d}", "surface_out", "Z.npy")
        if not os.path.exists(z_path):
            continue

        Z = np.load(z_path)
        if Z.ndim != 2:
            continue
        if Z.shape[0] != pair_ij.shape[0]:
            continue

        # score: max abs corr across selected tau range
        score = np.nanmax(np.abs(Z[:, tau_mask]), axis=1)
        S_list.append(score)
        valid_outer.append(row)

    if len(S_list) < 3:
        print(f"[WARN] Too few valid outer windows for {session_name} | {bin_name}")
        return None

    S = np.stack(S_list, axis=1)  # (P, W)
    P, W = S.shape

    global_mean = np.nanmean(S, axis=0, keepdims=True)
    if USE_GLOBAL_RESIDUAL:
        S_used = S - global_mean
    else:
        S_used = S.copy()

    dS = np.diff(S_used, axis=1)   # (P, W-1)
    dS_raw = np.diff(S, axis=1)

    # metrics
    max_up = np.nanmax(dS, axis=1)
    max_down = np.nanmin(dS, axis=1)
    max_abs_jump = np.nanmax(np.abs(dS), axis=1)
    max_abs_jump_raw = np.nanmax(np.abs(dS_raw), axis=1)

    arg_up = np.nanargmax(dS, axis=1)
    arg_down = np.nanargmin(dS, axis=1)
    arg_abs = np.nanargmax(np.abs(dS), axis=1)

    time_centers = np.array([(r["t0_sec"] + r["t1_sec"]) / 2.0 for r in valid_outer])

    summary_rows = []

    for p in range(P):
        i, j = pair_ij[p]
        k_abs = int(arg_abs[p])  # jump occurs between k_abs and k_abs+1

        # post-jump features
        after_start = k_abs + 1
        after_vals = S_used[p, after_start:]

        if len(after_vals) == 0:
            peak_after = np.nan
            mean_after_3 = np.nan
            persist_count = 0
            persist_run = 0
        else:
            peak_after = float(np.nanmax(after_vals))
            mean_after_3 = float(np.nanmean(after_vals[:min(3, len(after_vals))]))
            above = after_vals > PERSIST_THRESH
            persist_count = int(np.sum(above))
            persist_run = int(contiguous_run_length_from(above, 0))

        isolated_noise_flag = int(
            (persist_run < MIN_PERSIST_COUNT) or
            (not np.isfinite(peak_after)) or
            (peak_after < MIN_PEAK_AFTER) or
            (not np.isfinite(mean_after_3)) or
            (mean_after_3 < MIN_MEAN_AFTER_3)
        )

        summary_rows.append({
            "session": session_name,
            "area": "CIP" if "CIP" in session_name.upper() else ("V3A" if "V3A" in session_name.upper() else "UNKNOWN"),
            "bin": bin_name,
            "pair_idx": int(p),
            "i_valid": int(i),
            "j_valid": int(j),
            "i_label": safe_label(neuron_map, int(i)),
            "j_label": safe_label(neuron_map, int(j)),
            "max_up": float(max_up[p]),
            "max_down": float(max_down[p]),
            "max_abs_jump": float(max_abs_jump[p]),
            "max_abs_jump_raw": float(max_abs_jump_raw[p]),
            "best_transition": f"{k_abs}->{k_abs+1}",
            "jump_from_center_sec": float(time_centers[k_abs]),
            "jump_to_center_sec": float(time_centers[k_abs+1]),
            "peak_after_jump": peak_after,
            "mean_after_3wins": mean_after_3,
            "persist_count": persist_count,
            "persist_run": persist_run,
            "isolated_noise_flag": isolated_noise_flag,
        })

    # save per-combo summary
    combo_out = os.path.join(OUT_ROOT, session_name, bin_name)
    ensure_dir(combo_out)

    summary_csv = os.path.join(combo_out, "pair_jump_summary.csv")
    fieldnames = list(summary_rows[0].keys())
    write_csv(summary_csv, summary_rows, fieldnames)

    # filtered ranking
    filtered = [
        r for r in summary_rows
        if r["isolated_noise_flag"] == 0
    ]

    filtered_sorted = sorted(
        filtered,
        key=lambda r: (r["max_abs_jump"], r["peak_after_jump"], r["persist_run"]),
        reverse=True
    )

    top_combo = filtered_sorted[:TOP_K_PER_COMBO]
    top_csv = os.path.join(combo_out, f"top_{TOP_K_PER_COMBO}_filtered_jump_pairs.csv")
    if len(top_combo) > 0:
        write_csv(top_csv, top_combo, fieldnames)
    else:
        write_csv(top_csv, [], fieldnames)

    # plots
    try:
        global_plot = os.path.join(combo_out, "global_mean_score.png")
        plt.figure(figsize=(10, 5))
        plt.plot(time_centers, np.nanmean(S, axis=0), marker="o")
        plt.xlabel("Outer window center time (sec)")
        plt.ylabel("Mean pair score")
        plt.title(f"Global mean score | {session_name} | {bin_name}")
        plt.tight_layout()
        plt.savefig(global_plot, dpi=150)
        plt.close()

        plot_dir = os.path.join(combo_out, "top_pair_plots")
        ensure_dir(plot_dir)

        for rank, row in enumerate(top_combo[:PLOT_TOP_PER_COMBO], start=1):
            p = int(row["pair_idx"])
            k = int(arg_abs[p])

            plt.figure(figsize=(10, 5))
            plt.plot(time_centers, S[p], marker="o", label="raw score")
            plt.plot(time_centers, S_used[p], marker="o", label="residual score")
            plt.axvline(time_centers[k], linestyle="--")
            plt.axvline(time_centers[k+1], linestyle="--")
            plt.xlabel("Outer window center time (sec)")
            plt.ylabel("Score")
            plt.title(
                f"{session_name} | {bin_name} | rank {rank}\n"
                f"{row['i_label']} vs {row['j_label']}"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"rank_{rank:02d}_pair_{p}.png"), dpi=150)
            plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed for {session_name} | {bin_name}: {e}")

    print(f"[OK] {session_name} | {bin_name} | total_pairs={len(summary_rows)} | kept={len(filtered)}")

    return summary_rows, filtered_sorted

# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dir(OUT_ROOT)

    sessions = detect_sessions(ROOT_DIR)
    if len(sessions) == 0:
        print("[WARN] No session folders found.")
        return

    print(f"[INFO] Found sessions: {sessions}")

    all_rows = []
    global_filtered_rows = []

    for session_name in sessions:
        session_root = os.path.join(ROOT_DIR, session_name)
        on_root = os.path.join(session_root, "on_1000ms")
        if not os.path.exists(on_root):
            print(f"[WARN] Missing on_1000ms: {session_name}")
            continue

        bins = BIN_FOLDERS if BIN_FOLDERS is not None else detect_bins(on_root)
        if len(bins) == 0:
            print(f"[WARN] No bin folders in: {on_root}")
            continue

        for bin_name in bins:
            result = analyze_one_bin(session_name, session_root, bin_name)
            if result is None:
                continue

            summary_rows, filtered_sorted = result
            all_rows.extend(summary_rows)
            global_filtered_rows.extend(filtered_sorted)

    if len(all_rows) == 0:
        print("[WARN] No valid results.")
        return

    # save merged all rows
    merged_csv = os.path.join(OUT_ROOT, "ALL_pair_jump_summary.csv")
    fieldnames = list(all_rows[0].keys())
    write_csv(merged_csv, all_rows, fieldnames)

    # global ranking after filtering
    global_sorted = sorted(
        global_filtered_rows,
        key=lambda r: (r["max_abs_jump"], r["peak_after_jump"], r["persist_run"]),
        reverse=True
    )

    global_top = global_sorted[:TOP_K_GLOBAL]
    global_top_csv = os.path.join(OUT_ROOT, f"GLOBAL_top_{TOP_K_GLOBAL}_filtered_jump_pairs.csv")
    write_csv(global_top_csv, global_top, fieldnames)

    # area-specific rankings
    for area in ["CIP", "V3A"]:
        area_rows = [r for r in global_sorted if r["area"] == area]
        area_top_csv = os.path.join(OUT_ROOT, f"{area}_top_{TOP_K_GLOBAL}_filtered_jump_pairs.csv")
        write_csv(area_top_csv, area_rows[:TOP_K_GLOBAL], fieldnames)

    print("\n[DONE]")
    print(f"All summary: {merged_csv}")
    print(f"Global top:   {global_top_csv}")

if __name__ == "__main__":
    main()