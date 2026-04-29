"""
07_make_videos_dual_encode.py
==============================
4-up sliding video: height = cross-correlation (from 01),
                    color  = ML perceptron coupling weight (from ML/04).

Reads:
    ./out_binary_outer_surfaces_ON_only/SESSION/ON_stimOnAnchor/
        binary_<bin>ms/outer_XXXX/surface_out/{Z, tau_ms, pair_ij}.npy
    ./ML/out_perceptron_tau/SESSION/ON_stimOnAnchor/
        counts_<bin>ms/outer_XXXX/coupling_tau_pairs_fwd.npy

Writes:
    ./final_videos_dual/REGION_ON_dual_<bin>ms.mp4
    (with PRE / POST static panels from 08 held at start/end)
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors as mcolors
from moviepy.editor import ImageSequenceClip


# ------------------------------------------------------------------ #
CORR_DIR    = "./out_binary_outer_surfaces_ON_only"
ML_DIR      = "./ML/out_perceptron_tau"
FINAL_DIR   = "./final_videos_dual"
PREPOST_DIR = "./final_prepost_dual"          # written by 08
os.makedirs(FINAL_DIR, exist_ok=True)

REGION_SESSIONS = {
    "CIP": ["CIP_1", "CIP_2", "CIP_3", "CIP_4"],
    "V3A": ["V3A_1", "V3A_2", "V3A_3", "V3A_4"],
}

EPOCH_DIR = "ON_stimOnAnchor"

# (corr_tag, ml_tag) pairs
BIN_PAIRS = [
    ("binary_0p500ms", "counts_0p500ms"),
    ("binary_1p000ms", "counts_1p000ms"),
    ("binary_2p000ms", "counts_2p000ms"),
]

ML_DIRECTION = "fwd"                          # i→j perceptron weight

FPS               = 5
DPI               = 160
VIEW_ELEV         = 30
VIEW_AZIM         = -120
MAX_PAIRS_TO_PLOT = 6000
CMAP_COLOR        = "RdBu_r"                  # diverging for ML (signed)
PERCENTILE        = 98
GAMMA             = 0.7

PRE_HOLD_SEC  = 2.0
POST_HOLD_SEC = 2.0


# ------------------------------------------------------------------ #
def load_corr(session, bin_tag, outer_idx):
    base = os.path.join(CORR_DIR, session, EPOCH_DIR, bin_tag,
                        f"outer_{outer_idx:04d}", "surface_out")
    Z   = np.load(os.path.join(base, "Z.npy"))
    tau = np.load(os.path.join(base, "tau_ms.npy"))
    return Z, tau


def load_ml_fwd(session, bin_tag, outer_idx):
    base = os.path.join(ML_DIR, session, EPOCH_DIR, bin_tag,
                        f"outer_{outer_idx:04d}")
    W = np.load(os.path.join(base, f"coupling_tau_pairs_{ML_DIRECTION}.npy"))
    return W


def load_outer_index(session):
    path = os.path.join(CORR_DIR, session, EPOCH_DIR, "outer_index.csv")
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "outer_idx": int(row["outer_idx"]),
                "t0_sec":    float(row["t0_sec"]),
                "t1_sec":    float(row["t1_sec"]),
            })
    return rows


def find_common_outer_rows(sessions, corr_tag, ml_tag):
    per_session = []
    for session in sessions:
        c_base = os.path.join(CORR_DIR, session, EPOCH_DIR, corr_tag)
        m_base = os.path.join(ML_DIR,   session, EPOCH_DIR, ml_tag)
        if not (os.path.isdir(c_base) and os.path.isdir(m_base)):
            print(f"  [WARN] missing dirs for {session}: {c_base} | {m_base}")
            return []
        rows = load_outer_index(session)
        c_count = len([d for d in os.listdir(c_base) if d.startswith("outer_")])
        m_count = len([d for d in os.listdir(m_base) if d.startswith("outer_")])
        per_session.append(rows[:min(c_count, m_count)])
    if not per_session:
        return []
    min_len = min(len(r) for r in per_session)
    return per_session[0][:min_len]


# ------------------------------------------------------------------ #
def compute_global_scales(sessions, corr_tag, ml_tag, outer_rows):
    z_vals, w_vals = [], []
    for row in outer_rows:
        oi = row["outer_idx"]
        for session in sessions:
            Z = load_corr(session, corr_tag, oi)[0]
            W = load_ml_fwd(session, ml_tag, oi)
            z_vals.append(np.abs(Z).ravel())
            w_vals.append(np.abs(W).ravel())
    z_max = max(np.percentile(np.concatenate(z_vals), PERCENTILE), 1e-6) ** GAMMA
    w_max = max(np.percentile(np.concatenate(w_vals), PERCENTILE), 1e-6) ** GAMMA
    return z_max, w_max


# ------------------------------------------------------------------ #
def plot_dual_surface_to_png(Z, W, tau_ms, out_png, title,
                             z_max_vis, w_max_vis):
    """
    Height = Z (corr), Color = W (ML fwd).
    Both undergo sign × |·|^GAMMA visual compression.
    """
    pair_count = Z.shape[0]
    stride     = 1 if pair_count <= MAX_PAIRS_TO_PLOT \
                   else int(np.ceil(pair_count / MAX_PAIRS_TO_PLOT))

    Zp = Z[::stride, :]
    Wp = W[::stride, :]

    Z_vis = np.sign(Zp) * (np.abs(Zp) ** GAMMA)
    W_vis = np.sign(Wp) * (np.abs(Wp) ** GAMMA)

    pair_axis = np.arange(Zp.shape[0])
    Xg, Yg    = np.meshgrid(tau_ms, pair_axis)

    # color map from W
    norm  = mcolors.Normalize(vmin=-w_max_vis, vmax=w_max_vis)
    cmap  = cm.get_cmap(CMAP_COLOR)
    facec = cmap(norm(W_vis))                      # (n_pair, n_tau, 4) RGBA

    fig = plt.figure(figsize=(7.2, 5.0))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        Xg, Yg, Z_vis,
        facecolors=facec,
        linewidth=0, antialiased=True,
        shade=False,
    )

    ax.set_xlabel("Tau (ms)")
    ax.set_ylabel("Pair idx")
    ax.set_zlabel("Corr (height)")
    # NOTE: do NOT set_zlim — let matplotlib auto-fit so tall peaks
    # stay inside the plot box (matches 02/03 behavior).
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    # colorbar for the color channel (ML)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label(f"ML w ({ML_DIRECTION})")

    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.06, top=0.90)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def make_grid_frame(pngs, grid_png, suptitle):
    fig = plt.figure(figsize=(12, 8))
    for k, png_path in enumerate(pngs[:4]):
        img = plt.imread(png_path)
        ax  = fig.add_subplot(2, 2, k + 1)
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.90,
                        wspace=0.04, hspace=0.08)
    fig.savefig(grid_png, dpi=DPI)
    plt.close(fig)


def make_hold_frame(panel_png, out_png):
    img = plt.imread(panel_png)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def get_prepost_panel_path(region_name, corr_tag, which):
    """Match 08's naming: REGION_<EPOCH>_<corr_tag>_dual_4up.png"""
    epoch = "PRE_stimOnAnchor" if which == "pre" else "POST_stimOffAnchor"
    fname = f"{region_name}_{epoch}_{corr_tag}_dual_4up.png"
    path  = os.path.join(PREPOST_DIR, fname)
    if not os.path.isfile(path):
        print(f"  [WARN] missing {which} panel: {path}")
        return None
    return path


# ------------------------------------------------------------------ #
def render_region_video(region_name, sessions, corr_tag, ml_tag):
    outer_rows = find_common_outer_rows(sessions, corr_tag, ml_tag)
    if not outer_rows:
        print(f"[SKIP] {region_name} / {corr_tag}: no common outer windows")
        return

    print(f"  [{region_name} / {corr_tag}] computing global scales …")
    z_max, w_max = compute_global_scales(sessions, corr_tag, ml_tag, outer_rows)
    print(f"    z_max_vis={z_max:.4f}  w_max_vis={w_max:.4f}  "
          f"({len(outer_rows)} windows)")

    tmp_dir = os.path.join(FINAL_DIR, f"_tmp_{region_name}_{corr_tag}")
    os.makedirs(tmp_dir, exist_ok=True)

    frame_paths = []

    # PRE hold
    pre_panel = get_prepost_panel_path(region_name, corr_tag, "pre")
    if pre_panel:
        pre_png = os.path.join(tmp_dir, "intro_pre.png")
        make_hold_frame(pre_panel, pre_png)
        frame_paths.extend([pre_png] * max(1, int(round(PRE_HOLD_SEC * FPS))))

    # ON sliding
    total = len(outer_rows)
    for fi, row in enumerate(outer_rows):
        oi = row["outer_idx"]
        pngs = []

        for si, session in enumerate(sessions):
            Z, tau = load_corr(session, corr_tag, oi)
            W       = load_ml_fwd(session, ml_tag, oi)
            out_png = os.path.join(tmp_dir, f"frame_{oi:04d}_s{si}.png")
            title   = (f"{session} | {region_name} | ON | {corr_tag}\n"
                       f"height=corr | color=ML({ML_DIRECTION}) | "
                       f"[{row['t0_sec']:.3f}, {row['t1_sec']:.3f}) s")
            plot_dual_surface_to_png(Z, W, tau, out_png, title, z_max, w_max)
            pngs.append(out_png)

        grid_png = os.path.join(tmp_dir, f"grid_{oi:04d}.png")
        make_grid_frame(
            pngs, grid_png,
            f"{region_name} | ON outer={oi} | {corr_tag} | "
            f"height=corr, color=ML({ML_DIRECTION})  "
            f"[{row['t0_sec']:.3f}, {row['t1_sec']:.3f}) s",
        )
        frame_paths.append(grid_png)
        print(f"  {region_name} {corr_tag}: {fi+1}/{total}", end="\r")

    # POST hold
    post_panel = get_prepost_panel_path(region_name, corr_tag, "post")
    if post_panel:
        post_png = os.path.join(tmp_dir, "outro_post.png")
        make_hold_frame(post_panel, post_png)
        frame_paths.extend([post_png] * max(1, int(round(POST_HOLD_SEC * FPS))))

    if not frame_paths:
        print(f"\n[SKIP] {region_name}: no frames produced")
        return

    out_mp4 = os.path.join(FINAL_DIR,
                           f"{region_name}_ON_dual_{corr_tag}.mp4")
    clip = ImageSequenceClip(frame_paths, fps=FPS)
    try:
        clip.write_videofile(out_mp4, codec="libx264", audio=False,
                             verbose=False, logger="bar")
    except Exception:
        print(f"\n  [INFO] libx264 unavailable, falling back to mpeg4")
        clip.write_videofile(out_mp4, codec="mpeg4", audio=False,
                             verbose=False, logger="bar",
                             ffmpeg_params=["-q:v", "5"])
    print(f"\n[OK] {out_mp4}")


def main():
    for region_name, sessions in REGION_SESSIONS.items():
        if len(sessions) != 4:
            print(f"[SKIP] {region_name}: need 4 sessions")
            continue
        for corr_tag, ml_tag in BIN_PAIRS:
            render_region_video(region_name, sessions, corr_tag, ml_tag)


if __name__ == "__main__":
    main()
