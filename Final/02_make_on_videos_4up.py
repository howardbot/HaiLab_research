import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip

SURFACE_DIR = "./out_binary_outer_surfaces_ON_only"
FINAL_DIR = "./final_videos_with_prepost"
PREPOST_DIR = "./final_prepost_frames_like02"
os.makedirs(FINAL_DIR, exist_ok=True)

REGION_SESSIONS = {
    "CIP": ["CIP_1", "CIP_2", "CIP_3", "CIP_4"],
    "V3A": ["V3A_1", "V3A_2", "V3A_3", "V3A_4"],
}

EPOCH_DIR = "ON_stimOnAnchor"

BINARY_TAGS = [
    "binary_0p500ms",
    "binary_1p000ms",
    "binary_2p000ms",
]

FPS = 5
DPI = 160
VIEW_ELEV = 30
VIEW_AZIM = -120
MAX_PAIRS_TO_PLOT = 6000
CMAP = "RdBu_r"
PERCENTILE = 98
GAMMA = 0.7

# how long PRE / POST static panels stay on screen
PRE_HOLD_SEC = 2.0
POST_HOLD_SEC = 2.0


def load_surface(session, binary_tag, outer_idx):
    base = os.path.join(
        SURFACE_DIR,
        session,
        EPOCH_DIR,
        binary_tag,
        f"outer_{outer_idx:04d}",
        "surface_out",
    )
    Z = np.load(os.path.join(base, "Z.npy"))
    tau = np.load(os.path.join(base, "tau_ms.npy"))
    return Z, tau


def load_outer_index(session):
    path = os.path.join(SURFACE_DIR, session, EPOCH_DIR, "outer_index.csv")
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "outer_idx": int(row["outer_idx"]),
                    "t0_sec": float(row["t0_sec"]),
                    "t1_sec": float(row["t1_sec"]),
                }
            )
    return rows


def find_common_outer_rows(sessions, binary_tag):
    per_session_rows = []
    for session in sessions:
        base = os.path.join(SURFACE_DIR, session, EPOCH_DIR, binary_tag)
        if not os.path.isdir(base):
            return []
        rows = load_outer_index(session)
        count = len([d for d in os.listdir(base) if d.startswith("outer_")])
        per_session_rows.append(rows[:count])

    if not per_session_rows:
        return []

    min_len = min(len(rows) for rows in per_session_rows)
    return per_session_rows[0][:min_len]


def compute_global_scale(sessions, binary_tag, outer_rows):
    vals = []
    for row in outer_rows:
        outer_idx = row["outer_idx"]
        for session in sessions:
            Z, _ = load_surface(session, binary_tag, outer_idx)
            vals.append(np.abs(Z).ravel())

    if not vals:
        return 1e-6

    abs_max = np.percentile(np.concatenate(vals), PERCENTILE)
    if abs_max < 1e-6:
        abs_max = 1e-6

    abs_max_vis = abs_max ** GAMMA
    if abs_max_vis < 1e-6:
        abs_max_vis = 1e-6

    return abs_max_vis


def plot_surface_to_png(Z, tau_ms, out_png, title, abs_max_vis):
    pair_count, _ = Z.shape
    stride = 1
    if pair_count > MAX_PAIRS_TO_PLOT:
        stride = int(np.ceil(pair_count / MAX_PAIRS_TO_PLOT))

    Zp = Z[::stride, :]
    Zp_vis = np.sign(Zp) * (np.abs(Zp) ** GAMMA)

    pair_axis = np.arange(Zp.shape[0])
    Xg, Yg = np.meshgrid(tau_ms, pair_axis)

    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Xg,
        Yg,
        Zp_vis,
        cmap=CMAP,
        vmin=-abs_max_vis,
        vmax=abs_max_vis,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel("Tau (ms)")
    ax.set_ylabel("Pair idx")
    ax.set_zlabel("Corr")
    ax.set_title(title)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.06, top=0.90)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def make_grid_frame_from_session_pngs(pngs, grid_png, suptitle):
    fig = plt.figure(figsize=(12, 8))

    for k, png_path in enumerate(pngs[:4]):
        img = plt.imread(png_path)
        ax = fig.add_subplot(2, 2, k + 1)
        ax.imshow(img)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.90, wspace=0.04, hspace=0.08)
    fig.savefig(grid_png, dpi=DPI)
    plt.close(fig)


def get_prepost_panel_path(region_name, binary_tag, which):
    """
    Match new baseline (like02) naming:
      CIP_PRE_stimOnAnchor_binary_xxx_4up.png
      CIP_POST_stimOffAnchor_binary_xxx_4up.png
      V3A_PRE_stimOnAnchor_binary_xxx_4up.png
      V3A_POST_stimOffAnchor_binary_xxx_4up.png
    """
    if which == "pre":
        filename = f"{region_name}_PRE_stimOnAnchor_{binary_tag}_4up.png"
    elif which == "post":
        filename = f"{region_name}_POST_stimOffAnchor_{binary_tag}_4up.png"
    else:
        raise ValueError("which must be 'pre' or 'post'")

    path = os.path.join(PREPOST_DIR, filename)

    if not os.path.isfile(path):
        print(f"[WARN] missing {which.upper()} panel for {region_name} / {binary_tag}: {path}")
        return None

    return path


def make_hold_frame(panel_png, out_png, title):
    img = plt.imread(panel_png)

    fig = plt.figure(figsize=(12, 8))
    # 使用 add_axes([left, bottom, width, height])
    # [0, 0, 1, 1] 表示让这张图片 100% 铺满整个 12x8 的画布，不留任何边距
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")

    # 只有当 title 真的是字符串时，才用 fig.text 手动加上去
    if title is not None:
        fig.text(0.5, 0.95, title, ha='center', va='top', fontsize=14)

    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def render_region_video(region_name, sessions, binary_tag):
    outer_rows = find_common_outer_rows(sessions, binary_tag)
    if not outer_rows:
        print(f"[SKIP] missing epoch/binary outputs for {region_name} / {binary_tag}")
        return

    abs_max_vis = compute_global_scale(sessions, binary_tag, outer_rows)

    tmp_dir = os.path.join(FINAL_DIR, f"_tmp_frames_{region_name}_{binary_tag}")
    os.makedirs(tmp_dir, exist_ok=True)

    frame_paths = []
    total_frames = len(outer_rows)

    # ---------- PRE intro ----------
    pre_panel = get_prepost_panel_path(region_name, binary_tag, "pre")
    if pre_panel is not None:
        pre_hold_png = os.path.join(tmp_dir, "intro_pre.png")
        make_hold_frame(
            pre_panel,
            pre_hold_png,
            None,
        )
        pre_hold_n = max(1, int(round(PRE_HOLD_SEC * FPS)))
        frame_paths.extend([pre_hold_png] * pre_hold_n)

    # ---------- ON sliding frames ----------
    for frame_idx, row in enumerate(outer_rows):
        outer_idx = row["outer_idx"]
        pngs = []

        for session_idx, session in enumerate(sessions):
            Z, tau = load_surface(session, binary_tag, outer_idx)
            out_png = os.path.join(tmp_dir, f"frame_{outer_idx:04d}_s{session_idx}.png")
            title = (
                f"{session} | {region_name} | ON | {binary_tag} | "
                f"[{row['t0_sec']:.3f}, {row['t1_sec']:.3f}) s"
            )
            plot_surface_to_png(Z, tau, out_png, title, abs_max_vis)
            pngs.append(out_png)

        grid_png = os.path.join(tmp_dir, f"grid_{outer_idx:04d}.png")
        make_grid_frame_from_session_pngs(
            pngs,
            grid_png,
            (
                f"{region_name} | ON outer slide | {binary_tag} | "
                f"[{row['t0_sec']:.3f}, {row['t1_sec']:.3f}) s"
            ),
        )

        frame_paths.append(grid_png)
        print(f"  {region_name} {binary_tag}: frame {frame_idx + 1}/{total_frames} done", end="\r")

    # ---------- POST outro ----------
    post_panel = get_prepost_panel_path(region_name, binary_tag, "post")
    if post_panel is not None:
        post_hold_png = os.path.join(tmp_dir, "outro_post.png")
        make_hold_frame(
            post_panel,
            post_hold_png,
            None,
        )
        post_hold_n = max(1, int(round(POST_HOLD_SEC * FPS)))
        frame_paths.extend([post_hold_png] * post_hold_n)

    # ---------- write video ----------
    out_mp4 = os.path.join(FINAL_DIR, f"{region_name}_ON_with_PREPOST_{binary_tag}.mp4")
    clip = ImageSequenceClip(frame_paths, fps=FPS)
    clip.write_videofile(out_mp4, codec="libx264", audio=False, verbose=False, logger="bar")
    print(f"\n[OK] wrote {out_mp4}")


def main():
    for region_name, sessions in REGION_SESSIONS.items():
        if len(sessions) != 4:
            print(f"[SKIP] {region_name} needs exactly 4 sessions for 4-up, got {len(sessions)}")
            continue

        for binary_tag in BINARY_TAGS:
            render_region_video(region_name, sessions, binary_tag)


if __name__ == "__main__":
    main()