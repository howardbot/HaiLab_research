#!/usr/bin/env python3
"""
04_train_perceptron_tau_scan.py
================================
Perceptron-based connectivity inference with explicit tau scanning.

Based on: Vareberg et al. 2022 (Hai Lab)
  "Inference of Presynaptic Connectivity from Temporally Blurry
   Spike Trains by Supervised Learning"

Model (per target j, per source i, per lag tau):
    ŷ_j(t)    = w_{ij}(τ) · s_i(t-τ) + b
    w update  = w + lr * Δ * s_i(t-τ)      ← Widrow-Hoff / Eq.4 in paper
    Δ         = s_j(t) - ŷ_j(t)            ← bin-level error

Output: coupling_tau[i, j, tau_idx]  shape (N, N, n_tau)
        tau_ms[tau_idx]               shape (n_tau,)

Upper triangle → (n_pairs, n_tau): directly comparable to Z(pair, τ)
from 01_build_surfaces_from_mat.py.

Usage
-----
    python 04_train_perceptron_tau_scan.py
    python 04_train_perceptron_tau_scan.py --tau_range_ms 50 --tau_step_ms 5
    python 04_train_perceptron_tau_scan.py --sessions CIP_1 CIP_2
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch


# =============================================================
# CONFIG
# =============================================================

@dataclass
class TrainConfig:
    # ---- data ----
    data_dir:  str  = "D:/Research-Python/HaiLab_research/Final/ML/out_spike_counts"
    sessions:  list = field(default_factory=lambda: [
        "CIP_1", "CIP_2", "CIP_3", "CIP_4",
        "V3A_1", "V3A_2", "V3A_3", "V3A_4",
    ])
    bin_tag:   str  = "counts_5p000ms"           # active bin (set per-iteration in main)
    bin_tags:  list = field(default_factory=lambda: [
        "counts_0p500ms",   # 0.5 ms — mirrors 01
        "counts_1p000ms",   # 1   ms — mirrors 01
        "counts_2p000ms",   # 2   ms — mirrors 01
    ])
    epoch_dir: str  = "ON_stimOnAnchor"          # active epoch (set per-iteration in main)
    epoch_dirs: list = field(default_factory=lambda: [
        "PRE_stimOnAnchor",      # fixed_window
        "ON_stimOnAnchor",       # outer scan
        "POST_stimOffAnchor",    # fixed_window
    ])
    # Which epochs use a single fixed_window vs sliding outer windows
    fixed_window_epochs: list = field(default_factory=lambda: [
        "PRE_stimOnAnchor", "POST_stimOffAnchor",
    ])

    # ---- tau scan ----
    tau_range_ms: float = 50.0   # scan ±tau_range_ms
    tau_step_ms:  float = 0.0    # 0 = auto-match bin resolution (mirrors 01)

    # ---- perceptron training ----
    lr:         float = 1e-3
    n_epochs:   int   = 200
    train_frac: float = 0.8
    seed:       int   = 42

    # ---- output ----
    output_dir: str = "D:/Research-Python/HaiLab_research/Final/ML/out_perceptron_tau"
    device:     str = "auto"          # "auto" | "cuda" | "cpu"

    # ---- outer-window mode ----
    outer_idx: Optional[str] = "all"


def resolve_device(cfg) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


# =============================================================
# DATA LOADING  (same as 04_train_conv_alltoone.py)
# =============================================================

def load_spike_counts(cfg, session, outer_idx=None):
    sub  = f"outer_{outer_idx:04d}" if outer_idx is not None else "fixed_window"
    base = os.path.join(cfg.data_dir, session, cfg.epoch_dir, cfg.bin_tag, sub)
    counts_path = os.path.join(base, "spike_counts.npy")
    meta_path   = os.path.join(base, "meta.json")

    if not os.path.isfile(counts_path):
        return None, None

    counts = np.load(counts_path).astype(np.float32)   # (N, T, K)
    meta   = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    return counts, meta


def list_outer_indices(cfg, session):
    root = os.path.join(cfg.data_dir, session, cfg.epoch_dir, cfg.bin_tag)
    if not os.path.isdir(root):
        return []
    indices = []
    for d in os.listdir(root):
        if d.startswith("outer_"):
            try:
                indices.append(int(d.split("_")[1]))
            except ValueError:
                pass
    return sorted(indices)


# =============================================================
# TAU GRID
# =============================================================

def build_tau_grid(cfg, bin_res_ms):
    """
    Return tau values in ms and in bins.
    Positive tau  → source leads target  (s_i at t-τ predicts s_j at t)
    Negative tau  → target leads source
    """
    # tau_step_ms == 0  →  auto-match bin resolution (mirrors 01)
    step_ms = cfg.tau_step_ms if cfg.tau_step_ms > 0 else bin_res_ms
    rng_ms  = cfg.tau_range_ms

    # round to nearest bin
    step_bins = max(1, int(round(step_ms / bin_res_ms)))
    rng_bins  = int(round(rng_ms  / bin_res_ms))

    tau_bins = np.arange(-rng_bins, rng_bins + 1, step_bins)   # e.g. -10..+10
    tau_ms   = tau_bins * bin_res_ms
    return tau_bins.tolist(), tau_ms.tolist()


# =============================================================
# VECTORIZED PERCEPTRON  (closed-form least squares, all pairs at once)
# =============================================================
#
# Widrow-Hoff (paper Eq.4) converges to the least-squares solution.
# For univariate regression this has a closed form:
#
#     w = cov(x, y) / var(x)
#     b = mean(y) - w * mean(x)
#
# We batch this over ALL (source i, target j) pairs simultaneously
# via one matrix multiplication, on GPU.
# =============================================================

def shift_align_torch(counts_NTK, tau_bins):
    """
    Align source vs target along the K-bin axis with a lag of tau_bins.

    tau_bins > 0 : source at t-τ, target at t   (source leads)
    tau_bins < 0 : source at t+|τ|, target at t (target leads)

    Returns:
        src_NM : (N, T*(K-|τ|))   — source view, flattened
        tgt_NM : (N, T*(K-|τ|))   — target view, flattened
    """
    N, T, K = counts_NTK.shape
    ab      = abs(tau_bins)

    if ab == 0:
        src = counts_NTK
        tgt = counts_NTK
    elif tau_bins > 0:
        src = counts_NTK[:, :, :K - ab]      # earlier bins
        tgt = counts_NTK[:, :, ab:]          # later bins
    else:
        src = counts_NTK[:, :, ab:]
        tgt = counts_NTK[:, :, :K - ab]

    M = src.shape[1] * src.shape[2]
    return src.reshape(N, M), tgt.reshape(N, M)


def fit_pairs_one_tau(train_NTK, val_NTK, tau_bins):
    """
    Fit Widrow-Hoff perceptron for ALL (source, target) pairs at one tau.

    train_NTK, val_NTK : torch.Tensor on GPU, shape (N, T_split, K)

    Returns
    -------
    W        : (N, N)  W[i,j] = i→j coupling weight at this tau
    val_corr : (N, N)  Pearson r between predicted and true target on val set
    """
    Xtr, Ytr = shift_align_torch(train_NTK, tau_bins)   # (N, M_tr)
    Xva, Yva = shift_align_torch(val_NTK,   tau_bins)   # (N, M_va)

    Mtr = Xtr.shape[1]
    if Mtr < 5:
        N = train_NTK.shape[0]
        z = torch.zeros((N, N), device=train_NTK.device)
        return z, z

    # ---- closed-form fit on training set ----
    x_mean = Xtr.mean(dim=1, keepdim=True)            # (N, 1)
    y_mean = Ytr.mean(dim=1, keepdim=True)            # (N, 1)
    Xc     = Xtr - x_mean
    Yc     = Ytr - y_mean

    # cov[i,j] = <Xc[i], Yc[j]> / Mtr
    cov   = (Xc @ Yc.T) / Mtr                         # (N_src, N_tgt)
    var_x = (Xc * Xc).sum(dim=1) / Mtr                # (N,)
    W     = cov / (var_x.unsqueeze(1) + 1e-12)        # (N, N)  W[i,j]
    B     = y_mean.T - W * x_mean                     # (N, N)  B[i,j]

    # ---- validation correlation per (i, j) ----
    # y_hat[i, j, t] = W[i,j] * Xva[i, t] + B[i,j]
    # corr( y_hat[i,j,:], Yva[j,:] )
    Mva    = Xva.shape[1]
    yhat   = W.unsqueeze(2) * Xva.unsqueeze(1) + B.unsqueeze(2)   # (N_src, N_tgt, M_va)

    yh_mean = yhat.mean(dim=2, keepdim=True)
    yv_mean = Yva.mean(dim=1, keepdim=True).unsqueeze(0)          # (1, N_tgt, 1)

    yh_c    = yhat - yh_mean
    yv_c    = Yva.unsqueeze(0) - yv_mean                          # (1, N_tgt, M_va)

    num     = (yh_c * yv_c).sum(dim=2)
    den     = torch.sqrt((yh_c * yh_c).sum(dim=2) *
                         (yv_c * yv_c).sum(dim=2) + 1e-24)
    corr    = num / den

    # zero out i==j (self-coupling)
    N = W.shape[0]
    eye = torch.eye(N, device=W.device, dtype=torch.bool)
    W    = W.masked_fill(eye, 0.0)
    corr = corr.masked_fill(eye, 0.0)

    return W, corr


# =============================================================
# SESSION PIPELINE
# =============================================================

def run_session(cfg, session, outer_idx=None, device=None):
    if device is None:
        device = resolve_device(cfg)

    counts, meta = load_spike_counts(cfg, session, outer_idx)
    if counts is None:
        raise FileNotFoundError(
            f"spike_counts.npy not found for {session}. "
            f"Run 01_prepare_spike_counts.py first."
        )

    N, T, K    = counts.shape
    bin_res_ms = meta.get("bin_res_ms", 5.0)

    if N < 2 or T < 5 or K < 4:
        print(f"  [SKIP] insufficient data  N={N} T={T} K={K}")
        return None

    # ---- tau grid ----
    tau_bins_list, tau_ms_list = build_tau_grid(cfg, bin_res_ms)
    n_tau = len(tau_bins_list)

    # ---- train/val split (over trials) ----
    rng     = np.random.RandomState(cfg.seed)
    perm    = rng.permutation(T)
    n_train = int(T * cfg.train_frac)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    # ---- move data to GPU once ----
    counts_t  = torch.from_numpy(counts).to(device)             # (N, T, K)
    train_NTK = counts_t[:, train_idx, :]
    val_NTK   = counts_t[:, val_idx,   :]

    # ---- output tensors on GPU, then move back at the end ----
    W_all  = torch.zeros((N, N, n_tau), device=device)
    R_all  = torch.zeros((N, N, n_tau), device=device)

    t0 = time.time()

    for ti, tau_b in enumerate(tau_bins_list):
        W, R = fit_pairs_one_tau(train_NTK, val_NTK, tau_b)
        W_all[:, :, ti] = W
        R_all[:, :, ti] = R

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"  [{session}] N={N} T={T} K={K} bin={bin_res_ms}ms "
          f"outer={outer_idx} | {n_tau} tau steps in {elapsed:.2f}s")

    return {
        "coupling_tau": W_all.cpu().numpy().astype(np.float32),
        "val_corr_tau": R_all.cpu().numpy().astype(np.float32),
        "tau_ms":       np.array(tau_ms_list, dtype=np.float32),
        "tau_bins":     np.array(tau_bins_list, dtype=np.int32),
        "N": N, "T": T, "K": K,
        "bin_res_ms":   bin_res_ms,
        "meta":         meta,
    }


# =============================================================
# SAVE RESULTS
# =============================================================

def save_results(cfg, session, outer_idx, result):
    tag     = f"outer_{outer_idx:04d}" if outer_idx is not None else "fixed_window"
    out_dir = os.path.join(cfg.output_dir, session, cfg.epoch_dir,
                           cfg.bin_tag, tag)
    os.makedirs(out_dir, exist_ok=True)

    C   = result["coupling_tau"]   # (N, N, n_tau)
    N   = C.shape[0]
    tau = result["tau_ms"]

    # ---- full NxNxT coupling ----
    np.save(os.path.join(out_dir, "coupling_tau_NxNxT.npy"), C)
    np.save(os.path.join(out_dir, "tau_ms.npy"), tau)
    np.save(os.path.join(out_dir, "val_corr_tau_NxNxT.npy"), result["val_corr_tau"])

    # ---- upper-triangle surface: (n_pairs, n_tau) ----
    # directly comparable to Z(pair, tau) from 01_build_surfaces_from_mat.py
    ii, jj         = np.triu_indices(N, k=1)
    coupling_pairs = C[ii, jj, :]              # (n_pairs, n_tau)
    # asymmetric: also save reverse direction
    coupling_rev   = C[jj, ii, :]              # j→i direction
    np.save(os.path.join(out_dir, "coupling_tau_pairs_fwd.npy"), coupling_pairs)
    np.save(os.path.join(out_dir, "coupling_tau_pairs_rev.npy"), coupling_rev)

    # ---- metadata ----
    out_meta = {
        "session":      session,
        "outer_idx":    outer_idx,
        "bin_res_ms":   float(result["bin_res_ms"]),
        "tau_ms":       tau.tolist(),
        "n_tau":        len(tau),
        "N":            N,
        "T":            result["T"],
        "K":            result["K"],
        "n_pairs":      len(ii),
        "mean_val_corr": float(np.nanmean(result["val_corr_tau"])),
        "config": {k: v for k, v in asdict(cfg).items()
                   if not isinstance(v, list)},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(out_meta, f, indent=2)

    return out_dir


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Perceptron tau-scan connectivity (Vareberg et al. 2022)"
    )
    parser.add_argument("--sessions",      nargs="+", default=None)
    parser.add_argument("--bin_tags",      nargs="+", default=None,
                        help="Subset of bin tags to run, e.g. "
                             "--bin_tags counts_2p000ms counts_5p000ms")
    parser.add_argument("--bin_tag",       default=None,
                        help="(deprecated) single bin tag; use --bin_tags")
    parser.add_argument("--tau_range_ms",  type=float, default=50.0)
    parser.add_argument("--tau_step_ms",   type=float, default=0.0,
                        help="0 = auto-match bin resolution (mirrors 01)")
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--n_epochs",      type=int,   default=200)
    parser.add_argument("--outer_idx",     default="all",
                        help="Only used to override behavior for the ON epoch. "
                             "PRE/POST are always run as fixed_window.")
    parser.add_argument("--epochs",        nargs="+", default=None,
                        help="Subset of epochs to run, e.g. "
                             "--epochs PRE_stimOnAnchor ON_stimOnAnchor")
    parser.add_argument("--data_dir",      default=None)
    parser.add_argument("--device",        default="auto")
    args = parser.parse_args()

    cfg = TrainConfig(
        tau_range_ms = args.tau_range_ms,
        tau_step_ms  = args.tau_step_ms,
        lr           = args.lr,
        n_epochs     = args.n_epochs,
        device       = args.device,
    )
    if args.sessions:  cfg.sessions   = args.sessions
    if args.data_dir:  cfg.data_dir   = args.data_dir
    if args.epochs:    cfg.epoch_dirs = args.epochs
    if args.bin_tags:  cfg.bin_tags   = args.bin_tags
    elif args.bin_tag: cfg.bin_tags   = [args.bin_tag]

    device = resolve_device(cfg)

    print("=" * 60)
    print("  Perceptron Tau-Scan  (Vareberg et al. 2022)  [GPU vectorized]")
    print("=" * 60)
    print(f"  device={device}")
    if cfg.tau_step_ms > 0:
        n_tau = int(2 * cfg.tau_range_ms / cfg.tau_step_ms) + 1
        print(f"  tau: ±{cfg.tau_range_ms}ms  step={cfg.tau_step_ms}ms  "
              f"→ {n_tau} tau values")
    else:
        print(f"  tau: ±{cfg.tau_range_ms}ms  step=AUTO (= bin resolution)")
    print(f"  bin_tags={cfg.bin_tags}")
    print(f"  sessions={cfg.sessions}")
    print(f"  epochs={cfg.epoch_dirs}")
    print()

    total_t0 = time.time()

    # outer-most loop: bin resolution
    for bin_tag in cfg.bin_tags:
        cfg.bin_tag = bin_tag
        print(f"\n{'@'*60}")
        print(f"@ BIN: {bin_tag}")
        print(f"{'@'*60}")

        for epoch in cfg.epoch_dirs:
            cfg.epoch_dir = epoch
            is_fixed = epoch in cfg.fixed_window_epochs

            print(f"\n{'#'*60}")
            print(f"# EPOCH: {epoch}   "
                  f"({'fixed_window' if is_fixed else 'outer scan'})")
            print(f"{'#'*60}")

            for session in cfg.sessions:
                print(f"\n{'='*60}")
                print(f"[RUN] {session} | {bin_tag} | {epoch}")
                print(f"{'='*60}")

                # PRE / POST → single fixed_window
                if is_fixed:
                    try:
                        result = run_session(cfg, session, device=device,
                                             outer_idx=None)
                        if result:
                            save_results(cfg, session, None, result)
                    except FileNotFoundError as e:
                        print(f"  [ERROR] {e}")
                    continue

                # ON → honor --outer_idx (default 'all')
                if args.outer_idx in (None, "None"):
                    try:
                        result = run_session(cfg, session, device=device,
                                             outer_idx=None)
                        if result:
                            save_results(cfg, session, None, result)
                    except FileNotFoundError as e:
                        print(f"  [ERROR] {e}")

                elif args.outer_idx == "all":
                    indices = list_outer_indices(cfg, session)
                    if not indices:
                        print(f"  [SKIP] no outer windows found")
                        continue
                    print(f"  {len(indices)} outer windows")
                    sess_t0 = time.time()

                    for oi in indices:
                        try:
                            result = run_session(cfg, session, device=device,
                                                 outer_idx=oi)
                            if result:
                                save_results(cfg, session, oi, result)
                        except FileNotFoundError as e:
                            print(f"  [ERROR] outer {oi}: {e}")

                    print(f"  [{session}/{bin_tag}/{epoch}] "
                          f"ALL done in {time.time()-sess_t0:.0f}s")

                else:
                    oi = int(args.outer_idx)
                    try:
                        result = run_session(cfg, session, device=device,
                                             outer_idx=oi)
                        if result:
                            save_results(cfg, session, oi, result)
                    except FileNotFoundError as e:
                        print(f"  [ERROR] {e}")

    total = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"[DONE] total {total:.0f}s ({total/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
