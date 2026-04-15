#!/usr/bin/env python3
"""
04_train_conv_alltoone.py
=========================
All-to-one 1D-conv predictive model for spike-train connectivity.

Model (GLM-inspired):
    For each target neuron n:
        source_counts (N-1, K) → 1D causal conv (learned kernel)
                               → weighted sum over sources (coupling)
                               → log(firing_rate)
        Loss = Poisson NLL(predicted_rate, actual_count)

This is essentially a convolutional GLM:
    log λ_n(t) = Σ_m  w_m · (h * s_m)(t) + b
    where h = learned temporal kernel, w_m = coupling weight, s_m = source counts

OPTIMIZED: data stays on GPU, no DataLoader overhead, full-batch training.

Reads spike_counts.npy produced by 01_prepare_spike_counts.py.

Usage
-----
    python 04_train_conv_alltoone.py                      # default
    python 04_train_conv_alltoone.py --device cuda         # GPU
    python 04_train_conv_alltoone.py --bin_tag 5p000ms     # specific resolution
    python 04_train_conv_alltoone.py --shared_kernel 0     # per-neuron kernels
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
import torch.nn as nn
import torch.nn.functional as F


# =============================================================
# CONFIG
# =============================================================

@dataclass
class TrainConfig:
    # ---- data ----
    data_dir: str = "D:/Research-Python/HaiLab_research/Final/ML/out_spike_counts"
    sessions: list = field(default_factory=lambda: [
        "CIP_1", "CIP_2", "CIP_3", "CIP_4",
        "V3A_1", "V3A_2", "V3A_3", "V3A_4",
    ])
    bin_tag: str = "counts_5p000ms"
    epoch_dir: str = "ON_stimOnAnchor"

    # ---- model ----
    kernel_size_ms: float = 50.0
    shared_kernel: bool = True
    causal: bool = True
    bias: bool = False

    # ---- training ----
    epochs: int = 500
    lr: float = 5e-4
    weight_decay: float = 1e-4
    l1_kernel: float = 1e-4
    l1_coupling: float = 1e-3
    train_frac: float = 0.8
    seed: int = 42
    patience: int = 50

    # ---- output ----
    output_dir: str = "D:/Research-Python/HaiLab_research/Final/ML/out_conv_poisson"
    device: str = "auto"

    # ---- outer-window mode ----
    outer_idx: Optional[str] = "all"


def resolve_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


# =============================================================
# DATA LOADING
# =============================================================

def load_spike_counts(cfg, session, outer_idx=None):
    if outer_idx is not None:
        sub = f"outer_{outer_idx:04d}"
    else:
        sub = "fixed_window"

    base = os.path.join(cfg.data_dir, session, cfg.epoch_dir, cfg.bin_tag, sub)
    counts_path = os.path.join(base, "spike_counts.npy")
    meta_path   = os.path.join(base, "meta.json")

    if not os.path.isfile(counts_path):
        return None, None

    counts = np.load(counts_path)
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return counts, meta


def list_outer_indices(cfg, session):
    epoch_root = os.path.join(cfg.data_dir, session, cfg.epoch_dir, cfg.bin_tag)
    if not os.path.isdir(epoch_root):
        return []
    indices = []
    for d in os.listdir(epoch_root):
        if d.startswith("outer_"):
            try:
                indices.append(int(d.split("_")[1]))
            except ValueError:
                pass
    return sorted(indices)


# =============================================================
# MODEL
# =============================================================

class AllToOneConvPoisson(nn.Module):
    """
    GLM-inspired all-to-one model with Poisson output.

    source_counts (B, N-1, K) → Conv1D → weighted sum → log_rate (B, K)
    """

    def __init__(self, n_sources: int, kernel_size: int,
                 shared_kernel: bool = True, causal: bool = True,
                 bias: bool = False):
        super().__init__()
        self.n_sources = n_sources
        self.kernel_size = kernel_size
        self.shared_kernel = shared_kernel
        self.causal = causal

        if shared_kernel:
            self.conv = nn.Conv1d(1, 1, kernel_size, padding=0, bias=bias)
        else:
            self.conv = nn.Conv1d(
                n_sources, n_sources, kernel_size,
                padding=0, groups=n_sources, bias=bias,
            )

        self.coupling = nn.Parameter(torch.zeros(n_sources))
        self.out_bias = nn.Parameter(torch.tensor(-3.0))

        nn.init.xavier_uniform_(self.conv.weight, gain=0.01)
        nn.init.zeros_(self.coupling)

    def forward(self, source_counts: torch.Tensor) -> torch.Tensor:
        B, Ns, K = source_counts.shape

        if self.causal:
            pad = self.kernel_size - 1
            if self.shared_kernel:
                x = source_counts.reshape(B * Ns, 1, K)
                x = F.pad(x, (pad, 0))
                x = self.conv(x).reshape(B, Ns, K)
            else:
                x = F.pad(source_counts, (pad, 0))
                x = self.conv(x)
        else:
            pad = self.kernel_size // 2
            if self.shared_kernel:
                x = source_counts.reshape(B * Ns, 1, K)
                x = F.pad(x, (pad, pad))
                x = self.conv(x)[:, :, :K].reshape(B, Ns, K)
            else:
                x = F.pad(source_counts, (pad, pad))
                x = self.conv(x)[:, :, :K]

        w = self.coupling.unsqueeze(0).unsqueeze(-1)
        log_rate = (x * w).sum(dim=1) + self.out_bias
        return log_rate


# =============================================================
# TRAINING  (optimized: full-batch, data on GPU, no DataLoader)
# =============================================================

def train_one_target(target_idx: int, train_data: torch.Tensor,
                     val_data: torch.Tensor, cfg: TrainConfig,
                     device: torch.device, kernel_size: int):
    """
    Train one target neuron.

    train_data / val_data: already on GPU, shape (T_split, N, K)
    """
    N = train_data.shape[1]
    K = train_data.shape[2]
    n_sources = N - 1

    src_idx = [i for i in range(N) if i != target_idx]

    # Pre-slice source and target — stays on GPU, no copy
    train_src = train_data[:, src_idx, :]     # (T_train, N-1, K)
    train_tgt = train_data[:, target_idx, :]  # (T_train, K)
    val_src   = val_data[:, src_idx, :]
    val_tgt   = val_data[:, target_idx, :]

    model = AllToOneConvPoisson(
        n_sources=n_sources, kernel_size=kernel_size,
        shared_kernel=cfg.shared_kernel, causal=cfg.causal, bias=cfg.bias,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )

    best_val_loss = float("inf")
    best_state = None
    train_losses = []
    val_losses = []
    no_improve = 0

    for epoch in range(cfg.epochs):
        # ---- train (full batch, no DataLoader) ----
        model.train()
        log_rate = model(train_src)
        nll = torch.mean(torch.exp(log_rate) - train_tgt * log_rate)

        l1_k = cfg.l1_kernel * model.conv.weight.abs().mean()
        l1_c = cfg.l1_coupling * model.coupling.abs().mean()
        loss = nll + l1_k + l1_c

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(nll.item())

        # ---- validate (full batch) ----
        model.eval()
        with torch.no_grad():
            val_log_rate = model(val_src)
            val_nll = torch.mean(torch.exp(val_log_rate) - val_tgt * val_log_rate)
        val_losses.append(val_nll.item())

        if val_nll.item() < best_val_loss:
            best_val_loss = val_nll.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            break

    # ---- extract best model ----
    model.load_state_dict(best_state)
    model.eval()

    # Final correlation
    with torch.no_grad():
        pred_rate = torch.exp(model(val_src)).cpu().numpy().reshape(-1)
        true_count = val_tgt.cpu().numpy().reshape(-1)

    if np.std(pred_rate) > 1e-10 and np.std(true_count) > 1e-10:
        corr = float(np.corrcoef(pred_rate, true_count)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    return {
        "target_idx": target_idx,
        "src_idx": src_idx,
        "kernel_weights": model.conv.weight.detach().cpu().numpy(),
        "coupling_weights": model.coupling.detach().cpu().numpy(),
        "out_bias": model.out_bias.item(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "final_val_corr": corr,
        "kernel_size": kernel_size,
        "n_epochs_trained": len(train_losses),
    }


# =============================================================
# SESSION PIPELINE
# =============================================================

def run_session(cfg, session, outer_idx=None, device=None):
    if device is None:
        device = resolve_device(cfg)

    counts, meta = load_spike_counts(cfg, session, outer_idx)
    if counts is None:
        raise FileNotFoundError(
            f"spike_counts.npy not found for {session}/{cfg.bin_tag}. "
            f"Run 01_prepare_spike_counts.py first."
        )

    N, T, K = counts.shape
    bin_res_ms = meta.get("bin_res_ms", 5.0)

    print(f"  [{session}] N={N}, T={T}, K={K} (bin={bin_res_ms}ms) | outer={outer_idx}")

    if N < 2 or T < 7 or K < 10:
        print(f"  [SKIP] insufficient data")
        return None

    kernel_size = max(3, int(cfg.kernel_size_ms / bin_res_ms) + 1)
    kernel_size = min(kernel_size, K - 1)

    # ---- put ALL data on GPU once ----
    # counts: (N, T, K) → transpose to (T, N, K) for batch dim
    all_data = torch.from_numpy(counts.astype(np.float32)).permute(1, 0, 2).to(device)
    # all_data: (T, N, K)

    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(T)
    n_train = int(T * cfg.train_frac)
    train_data = all_data[perm[:n_train]]   # (T_train, N, K) on GPU
    val_data   = all_data[perm[n_train:]]   # (T_val, N, K)   on GPU

    if train_data.shape[0] < 5 or val_data.shape[0] < 2:
        print(f"  [SKIP] insufficient trials after split")
        return None

    # ---- train all targets ----
    results = []
    t0 = time.time()

    for n in range(N):
        res = train_one_target(n, train_data, val_data, cfg, device, kernel_size)
        if res is not None:
            results.append(res)
            if (n + 1) % 5 == 0 or n == N - 1:
                elapsed = time.time() - t0
                print(f"    target {n+1}/{N} | "
                      f"val={res['best_val_loss']:.4f} "
                      f"r={res['final_val_corr']:.3f} "
                      f"ep={res['n_epochs_trained']} | "
                      f"{elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  [{session}] {len(results)}/{N} targets in {elapsed:.1f}s")

    return results


def save_results(cfg, session, outer_idx, results, meta):
    if outer_idx is not None:
        tag = f"outer_{outer_idx:04d}"
    else:
        tag = "fixed_window"

    out_dir = os.path.join(cfg.output_dir, session, cfg.epoch_dir,
                           cfg.bin_tag, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Kernels
    np.savez_compressed(
        os.path.join(out_dir, "kernels.npz"),
        **{f"target_{r['target_idx']}": r["kernel_weights"] for r in results}
    )

    # Coupling matrix (sparse)
    max_src = max(len(r["coupling_weights"]) for r in results)
    coupling_mat = np.full((len(results), max_src), np.nan, dtype=np.float32)
    for i, r in enumerate(results):
        coupling_mat[i, :len(r["coupling_weights"])] = r["coupling_weights"]
    np.save(os.path.join(out_dir, "coupling_matrix.npy"), coupling_mat)

    # Full NxN coupling
    if results:
        N = max(r["target_idx"] for r in results) + 1
        coupling_full = np.zeros((N, N), dtype=np.float32)
        for r in results:
            tid = r["target_idx"]
            for si, src in enumerate(r["src_idx"]):
                coupling_full[src, tid] = r["coupling_weights"][si]
        np.save(os.path.join(out_dir, "coupling_full_NxN.npy"), coupling_full)

    # Loss curves
    loss_data = {}
    for r in results:
        tid = r["target_idx"]
        loss_data[f"train_{tid}"] = np.array(r["train_losses"], dtype=np.float32)
        loss_data[f"val_{tid}"]   = np.array(r["val_losses"],   dtype=np.float32)
    np.savez_compressed(os.path.join(out_dir, "loss_curves.npz"), **loss_data)

    # Metadata
    targets_meta = [{
        "target_idx": r["target_idx"],
        "src_idx": r["src_idx"],
        "best_val_loss": float(r["best_val_loss"]),
        "final_val_corr": float(r["final_val_corr"]),
        "out_bias": float(r["out_bias"]),
        "kernel_size": r["kernel_size"],
        "n_epochs_trained": r["n_epochs_trained"],
    } for r in results]

    all_meta = {
        "config": {k: v for k, v in asdict(cfg).items() if not isinstance(v, list)},
        "config_lists": {k: v for k, v in asdict(cfg).items() if isinstance(v, list)},
        "session": session,
        "bin_tag": cfg.bin_tag,
        "outer_idx": outer_idx,
        "data_meta": meta,
        "n_targets_trained": len(results),
        "mean_val_loss": float(np.mean([r["best_val_loss"] for r in results])),
        "mean_val_corr": float(np.mean([r["final_val_corr"] for r in results])),
        "targets": targets_meta,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(all_meta, f, indent=2)

    return out_dir


# =============================================================
# VISUALIZATION
# =============================================================

def plot_results(out_dir, results, session, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Kernel + Coupling
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    for r in results:
        kw = r["kernel_weights"].squeeze()
        ks = len(kw)
        t_bins = np.arange(ks) - (ks - 1) if cfg.causal else np.arange(ks) - ks // 2
        axes[0].plot(t_bins, kw, alpha=0.4, linewidth=0.8)

    axes[0].set_xlabel("Lag (bins)")
    axes[0].set_ylabel("Kernel weight")
    axes[0].set_title(f"Temporal kernels - {session}")
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[0].axvline(0, color="r", linewidth=0.5, linestyle="--", alpha=0.5)

    if results:
        N = max(r["target_idx"] for r in results) + 1
        coupling_full = np.zeros((N, N), dtype=np.float32)
        for r in results:
            tid = r["target_idx"]
            for si, src in enumerate(r["src_idx"]):
                coupling_full[src, tid] = r["coupling_weights"][si]
        vmax = np.max(np.abs(coupling_full)) + 1e-8
        im = axes[1].imshow(coupling_full, cmap="RdBu_r", aspect="auto",
                            vmin=-vmax, vmax=vmax)
        axes[1].set_xlabel("Target neuron")
        axes[1].set_ylabel("Source neuron")
        axes[1].set_title("Coupling weights")
        plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "kernel_and_coupling.png"), dpi=200)
    plt.close(fig)

    # 2. Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in results:
        axes[0].plot(r["train_losses"], alpha=0.3, linewidth=0.7)
        axes[1].plot(r["val_losses"],   alpha=0.3, linewidth=0.7)
    axes[0].set_title("Train (Poisson NLL)")
    axes[1].set_title("Val (Poisson NLL)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "loss_curves.png"), dpi=200)
    plt.close(fig)

    # 3. Summary
    corrs = [r["final_val_corr"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(corrs)), corrs, color="steelblue", alpha=0.8)
    ax.set_xlabel("Target neuron")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Prediction quality - {session}")
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "prediction_summary.png"), dpi=200)
    plt.close(fig)


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="All-to-one conv (Poisson)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--shared_kernel", type=int, default=1)
    parser.add_argument("--causal", type=int, default=1)
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--bin_tag", default="counts_5p000ms")
    parser.add_argument("--outer_idx", default="all")
    parser.add_argument("--plot", type=int, default=1)
    parser.add_argument("--kernel_size_ms", type=float, default=50.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        device=args.device, epochs=args.epochs, lr=args.lr,
        shared_kernel=bool(args.shared_kernel), causal=bool(args.causal),
        kernel_size_ms=args.kernel_size_ms, bin_tag=args.bin_tag,
        patience=args.patience,
    )
    if args.sessions:
        cfg.sessions = args.sessions
    if args.data_dir:
        cfg.data_dir = args.data_dir

    device = resolve_device(cfg)
    print(f"{'='*60}")
    print(f"  All-to-One Conv - Poisson NLL  (optimized)")
    print(f"{'='*60}")
    print(f"  device={device} | epochs={cfg.epochs} (patience={cfg.patience})")
    print(f"  kernel={cfg.kernel_size_ms}ms shared={cfg.shared_kernel} causal={cfg.causal}")
    print(f"  bin_tag={cfg.bin_tag}")
    print(f"  sessions={cfg.sessions}")
    print()

    total_t0 = time.time()

    for session in cfg.sessions:
        print(f"\n{'='*60}")
        print(f"[RUN] {session}")
        print(f"{'='*60}")

        if args.outer_idx is None or args.outer_idx == "None":
            try:
                results = run_session(cfg, session, outer_idx=None, device=device)
                if results:
                    _, meta = load_spike_counts(cfg, session, outer_idx=None)
                    out_dir = save_results(cfg, session, None, results, meta or {})
                    if args.plot:
                        plot_results(out_dir, results, session, cfg)
            except FileNotFoundError as e:
                print(f"  [ERROR] {e}")

        elif args.outer_idx == "all":
            indices = list_outer_indices(cfg, session)
            if not indices:
                print(f"  [SKIP] no outer windows found")
                continue

            print(f"  {len(indices)} outer windows")
            session_t0 = time.time()

            for oi in indices:
                try:
                    results = run_session(cfg, session, outer_idx=oi, device=device)
                    if results:
                        _, meta = load_spike_counts(cfg, session, outer_idx=oi)
                        out_dir = save_results(cfg, session, oi, results, meta or {})
                        if args.plot:
                            plot_results(out_dir, results, session, cfg)
                except FileNotFoundError as e:
                    print(f"  [ERROR] outer {oi}: {e}")

            print(f"  [{session}] ALL outer windows done in "
                  f"{time.time()-session_t0:.0f}s")

        else:
            oi = int(args.outer_idx)
            try:
                results = run_session(cfg, session, outer_idx=oi, device=device)
                if results:
                    _, meta = load_spike_counts(cfg, session, outer_idx=oi)
                    out_dir = save_results(cfg, session, oi, results, meta or {})
                    if args.plot:
                        plot_results(out_dir, results, session, cfg)
            except FileNotFoundError as e:
                print(f"  [ERROR] {e}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"[DONE] total {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
