"""
08_make_baseline_dual_4up.py
=============================
PRE / POST static 4-up PNGs with DUAL encoding:
    height = cross-correlation (from 01)
    color  = ML perceptron coupling weight (from ML/04, fwd direction)

Output naming matches what 07_make_videos_dual_encode.py expects:
    ./final_prepost_dual/REGION_PRE_stimOnAnchor_binary_<bin>ms_dual_4up.png
    ./final_prepost_dual/REGION_POST_stimOffAnchor_binary_<bin>ms_dual_4up.png
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors as mcolors


# ------------------------------------------------------------------ #
CORR_DIR  = "./out_binary_outer_surfaces_ON_only"
ML_DIR    = "./ML/out_perceptron_tau"
FINAL_DIR = "./final_prepost_dual"
os.makedirs(FINAL_DIR, exist_ok=True)

REGION_SESSIONS = {
    "CIP": ["CIP_1", "CIP_2", "CIP_3", "CIP_4"],
    "V3A": ["V3A_1", "V3A_2", "V3A_3", "V3A_4"],
}

# (corr_tag, ml_tag)
BIN_PAIRS = [
    ("binary_0p500ms", "counts_0p500ms"),
    ("binary_1p000ms", "counts_1p000ms"),
    ("binary_2p000ms", "counts_2p000ms"),
]

EPOCH_SPECS = [
    {"epoch_dir": "PRE_stimOnAnchor",    "label": "PRE fixed"},
    {"epoch_dir": "POST_stimOffAnchor",  "label": "POST fixed"},
]

ML_DIRECTION = "fwd"

DPI               = 160
VIEW_ELEV         = 30
VIEW_AZIM         = -120
MAX_PAIRS_TO_PLOT = 6000
CMAP_COLOR        = "RdBu_r"
PERCENTILE        = 98
GAMMA             = 0.7


# ------------------------------------------------------------------ #
def load_corr(session, epoch_dir, corr_tag):
    base = os.path.join(CORR_DIR, session, epoch_dir, corr_tag,
                        "fixed_window", "surface_out")
    Z   = np.load(os.path.join(base, "Z.npy"))
    tau = np.load(os.path.join(base, "tau_ms.npy"))
    return Z, tau


def load_ml_fwd(session, epoch_dir, ml_tag):
    base = os.path.join(ML_DIR, session, epoch_dir, ml_tag, "fixed_window")
    return np.load(os.path.join(base, f"coupling_tau_pairs_{ML_DIRECTION}.npy"))


def load_fixed_window_meta(session, epoch_dir):
    path = os.path.join(CORR_DIR, session, epoch_dir, "fixed_window.csv")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return {
        "anchor":     row["anchor"],
        "t0_sec":     float(row["t0_sec"]),
        "t1_sec":     float(row["t1_sec"]),
        "window_sec": float(row["window_sec"]),
        "n_trials":   int(float(row["n_trials"])),
    }


def available_for_all(sessions, epoch_dir, corr_tag, ml_tag):
    for s in sessions:
        c = os.path.join(CORR_DIR, s, epoch_dir, corr_tag,
                         "fixed_window", "surface_out")
        m = os.path.join(ML_DIR,   s, epoch_dir, ml_tag, "fixed_window")
        meta = os.path.join(CORR_DIR, s, epoch_dir, "fixed_window.csv")
        if not (os.path.isdir(c) and os.path.isdir(m) and os.path.isfile(meta)):
            return False
    return True


def compute_global_scales(sessions, epoch_dir, corr_tag, ml_tag):
    zs, ws = [], []
    for s in sessions:
        zs.append(np.abs(load_corr(s, epoch_dir, corr_tag)[0]).ravel())
        ws.append(np.abs(load_ml_fwd(s, epoch_dir, ml_tag)).ravel())
    z_max = max(np.percentile(np.concatenate(zs), PERCENTILE), 1e-6) ** GAMMA
    w_max = max(np.percentile(np.concatenate(ws), PERCENTILE), 1e-6) ** GAMMA
    return z_max, w_max


# ------------------------------------------------------------------ #
def plot_dual_surface_to_png(Z, W, tau_ms, out_png, title,
                             z_max_vis, w_max_vis):
    pair_count = Z.shape[0]
    stride     = 1 if pair_count <= MAX_PAIRS_TO_PLOT \
                   else int(np.ceil(pair_count / MAX_PAIRS_TO_PLOT))

    Zp, Wp = Z[::stride, :], W[::stride, :]
    Z_vis  = np.sign(Zp) * (np.abs(Zp) ** GAMMA)
    W_vis  = np.sign(Wp) * (np.abs(Wp) ** GAMMA)

    pair_axis = np.arange(Zp.shape[0])
    Xg, Yg    = np.meshgrid(tau_ms, pair_axis)

    norm  = mcolors.Normalize(vmin=-w_max_vis, vmax=w_max_vis)
    cmap  = cm.get_cmap(CMAP_COLOR)
    facec = cmap(norm(W_vis))

    fig = plt.figure(figsize=(7.2, 5.0))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        Xg, Yg, Z_vis,
        facecolors=facec,
        linewidth=0, antialiased=True, shade=False,
    )

    ax.set_xlabel("Tau (ms)")
    ax.set_ylabel("Pair idx")
    ax.set_zlabel("Corr (height)")
    # NOTE: do NOT set_zlim — let matplotlib auto-fit (matches 02/03).
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label(f"ML w ({ML_DIRECTION})")

    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.06, top=0.90)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def make_grid_frame(pngs, grid_png, suptitle):
    fig = plt.figure(figsize=(12, 8))
    for k, p in enumerate(pngs[:4]):
        img = plt.imread(p)
        ax  = fig.add_subplot(2, 2, k + 1)
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.90,
                        wspace=0.04, hspace=0.08)
    fig.savefig(grid_png, dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ #
def render_region_epoch(region_name, sessions, epoch_dir, label,
                        corr_tag, ml_tag):
    if not available_for_all(sessions, epoch_dir, corr_tag, ml_tag):
        print(f"[SKIP] missing data for {region_name}/{epoch_dir}/{corr_tag}")
        return

    z_max, w_max = compute_global_scales(sessions, epoch_dir, corr_tag, ml_tag)

    tmp_dir = os.path.join(FINAL_DIR,
                           f"_tmp_{region_name}_{epoch_dir}_{corr_tag}")
    os.makedirs(tmp_dir, exist_ok=True)

    metas, pngs = [], []
    for s in sessions:
        meta = load_fixed_window_meta(s, epoch_dir)
        Z, tau = load_corr(s, epoch_dir, corr_tag)
        W      = load_ml_fwd(s, epoch_dir, ml_tag)

        metas.append(meta)
        out_png = os.path.join(tmp_dir, f"{s}_{epoch_dir}_{corr_tag}.png")
        title   = (f"{s} | {region_name} | {label} | {corr_tag}\n"
                   f"height=corr, color=ML({ML_DIRECTION}) | "
                   f"[{meta['t0_sec']:.3f}, {meta['t1_sec']:.3f}) s")
        plot_dual_surface_to_png(Z, W, tau, out_png, title, z_max, w_max)
        pngs.append(out_png)

    ref = metas[0]
    grid_png = os.path.join(
        FINAL_DIR,
        f"{region_name}_{epoch_dir}_{corr_tag}_dual_4up.png"
    )
    make_grid_frame(
        pngs, grid_png,
        f"{region_name} | {label} | {corr_tag} | "
        f"height=corr, color=ML({ML_DIRECTION}) | "
        f"{ref['anchor']} | window [{ref['t0_sec']:.3f}, {ref['t1_sec']:.3f}) s",
    )
    print(f"[OK] {grid_png}")


def main():
    for region_name, sessions in REGION_SESSIONS.items():
        for spec in EPOCH_SPECS:
            for corr_tag, ml_tag in BIN_PAIRS:
                render_region_epoch(
                    region_name, sessions,
                    spec["epoch_dir"], spec["label"],
                    corr_tag, ml_tag,
                )


if __name__ == "__main__":
    main()
