import csv
import os

import matplotlib.pyplot as plt
import numpy as np

SURFACE_DIR = "./out_binary_outer_surfaces_ON_only"
FINAL_DIR = "./final_prepost_frames_like02"
os.makedirs(FINAL_DIR, exist_ok=True)

REGION_SESSIONS = {
    "CIP": ["CIP_1", "CIP_2", "CIP_3", "CIP_4"],
    "V3A": ["V3A_1", "V3A_2", "V3A_3", "V3A_4"],
}

BINARY_TAGS = [
    "binary_0p500ms",
    "binary_1p000ms",
    "binary_2p000ms",
]

EPOCH_SPECS = [
    {
        "epoch_dir": "PRE_stimOnAnchor",
        "label": "PRE fixed",
    },
    {
        "epoch_dir": "POST_stimOffAnchor",
        "label": "POST fixed",
    },
]

# -------- mimic 02 exactly --------
DPI = 160
VIEW_ELEV = 30
VIEW_AZIM = -120
MAX_PAIRS_TO_PLOT = 6000
CMAP = "RdBu_r"
PERCENTILE = 98
GAMMA = 0.7


def load_surface(session, epoch_dir, binary_tag):
    base = os.path.join(
        SURFACE_DIR,
        session,
        epoch_dir,
        binary_tag,
        "fixed_window",
        "surface_out",
    )
    Z = np.load(os.path.join(base, "Z.npy"))
    tau = np.load(os.path.join(base, "tau_ms.npy"))
    return Z, tau


def load_fixed_window_meta(session, epoch_dir):
    path = os.path.join(SURFACE_DIR, session, epoch_dir, "fixed_window.csv")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    return {
        "anchor": row["anchor"],
        "t0_sec": float(row["t0_sec"]),
        "t1_sec": float(row["t1_sec"]),
        "window_sec": float(row["window_sec"]),
        "n_trials": int(float(row["n_trials"])),
    }


def fixed_available_for_all_sessions(sessions, epoch_dir, binary_tag):
    for session in sessions:
        base = os.path.join(
            SURFACE_DIR,
            session,
            epoch_dir,
            binary_tag,
            "fixed_window",
            "surface_out",
        )
        meta = os.path.join(SURFACE_DIR, session, epoch_dir, "fixed_window.csv")
        if not os.path.isdir(base) or not os.path.isfile(meta):
            return False
    return True


def compute_global_scale_fixed(sessions, epoch_dir, binary_tag):
    vals = []
    for session in sessions:
        Z, _ = load_surface(session, epoch_dir, binary_tag)
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


def plot_surface_to_png_like02(Z, tau_ms, out_png, title, abs_max_vis):
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

    # 不用 tight_layout，避免 3D+colorbar 错位
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

    # 这里也尽量别用 tight_layout，改手动
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.90, wspace=0.04, hspace=0.08)

    fig.savefig(grid_png, dpi=DPI)
    plt.close(fig)


def render_region_epoch(region_name, sessions, epoch_dir, label, binary_tag):
    if not fixed_available_for_all_sessions(sessions, epoch_dir, binary_tag):
        print(f"[SKIP] missing fixed-window outputs for {region_name} / {epoch_dir} / {binary_tag}")
        return

    abs_max_vis = compute_global_scale_fixed(sessions, epoch_dir, binary_tag)

    tmp_dir = os.path.join(FINAL_DIR, f"_tmp_{region_name}_{epoch_dir}_{binary_tag}")
    os.makedirs(tmp_dir, exist_ok=True)

    metas = []
    pngs = []

    for session_idx, session in enumerate(sessions):
        meta = load_fixed_window_meta(session, epoch_dir)
        Z, tau = load_surface(session, epoch_dir, binary_tag)

        metas.append(meta)

        out_png = os.path.join(tmp_dir, f"{session}_{epoch_dir}_{binary_tag}.png")
        title = (
            f"{session} | {region_name} | {label} | {binary_tag} | "
            f"[{meta['t0_sec']:.3f}, {meta['t1_sec']:.3f}) s"
        )
        plot_surface_to_png_like02(Z, tau, out_png, title, abs_max_vis)
        pngs.append(out_png)

    ref = metas[0]
    grid_png = os.path.join(FINAL_DIR, f"{region_name}_{epoch_dir}_{binary_tag}_4up.png")
    make_grid_frame_from_session_pngs(
        pngs,
        grid_png,
        (
            f"{region_name} | {label} | {binary_tag} | "
            f"{ref['anchor']} | window [{ref['t0_sec']:.3f}, {ref['t1_sec']:.3f}) s"
        ),
    )

    print(f"[OK] wrote {grid_png}")


def main():
    for region_name, sessions in REGION_SESSIONS.items():
        for spec in EPOCH_SPECS:
            epoch_dir = spec["epoch_dir"]
            label = spec["label"]

            for binary_tag in BINARY_TAGS:
                render_region_epoch(region_name, sessions, epoch_dir, label, binary_tag)


if __name__ == "__main__":
    main()