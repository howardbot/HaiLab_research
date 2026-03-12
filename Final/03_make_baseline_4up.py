import csv
import os

import matplotlib.pyplot as plt
import numpy as np

SURFACE_DIR = "./out_binary_outer_surfaces_ON_only"
FINAL_DIR = "./final_prepost_frames"
os.makedirs(FINAL_DIR, exist_ok=True)

SESSIONS = ["CIP_1", "CIP_2", "CIP_3", "CIP_4"]
BINARY_TAGS = [
    "binary_0p500ms",
    "binary_1p000ms",
    "binary_2p000ms",
]

EPOCH_SPECS = [
    {
        "epoch_dir": "PRE_stimOnAnchor",
        "label": "PRE fixed",
        "anchor": "StimOn",
    },
    {
        "epoch_dir": "POST_stimOffAnchor",
        "label": "POST fixed",
        "anchor": "StimOff",
    },
]

VMIN, VMAX = -0.3, 0.3
CMAP = "coolwarm"
DPI = 220
VIEW_ELEV = 30
VIEW_AZIM = -120
MAX_PAIRS_TO_PLOT = 6000


def load_surface(session, epoch_dir, binary_tag):
    base = os.path.join(SURFACE_DIR, session, epoch_dir, binary_tag, "fixed_window", "surface_out")
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


def plot_surface(ax, Z, tau_ms, title):
    pair_count, _ = Z.shape
    stride = 1
    if pair_count > MAX_PAIRS_TO_PLOT:
        stride = int(np.ceil(pair_count / MAX_PAIRS_TO_PLOT))

    Zp = Z[::stride, :]
    pair_axis = np.arange(Zp.shape[0])
    Xg, Yg = np.meshgrid(tau_ms, pair_axis)
    surf = ax.plot_surface(Xg, Yg, Zp, cmap=CMAP, vmin=VMIN, vmax=VMAX, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("Tau (ms)")
    ax.set_ylabel("Pair idx")
    ax.set_zlabel("Corr")
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    return surf


def available_for_all_sessions(epoch_dir, binary_tag):
    for session in SESSIONS:
        base = os.path.join(SURFACE_DIR, session, epoch_dir, binary_tag, "fixed_window", "surface_out")
        meta = os.path.join(SURFACE_DIR, session, epoch_dir, "fixed_window.csv")
        if not os.path.isdir(base) or not os.path.isfile(meta):
            return False
    return True


def main():
    for spec in EPOCH_SPECS:
        epoch_dir = spec["epoch_dir"]
        label = spec["label"]

        for binary_tag in BINARY_TAGS:
            if not available_for_all_sessions(epoch_dir, binary_tag):
                print(f"[SKIP] missing fixed-window outputs for {epoch_dir} / {binary_tag}")
                continue

            fig = plt.figure(figsize=(14, 9))
            surfs = []
            metas = []

            for idx, session in enumerate(SESSIONS):
                meta = load_fixed_window_meta(session, epoch_dir)
                Z, tau = load_surface(session, epoch_dir, binary_tag)
                ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
                surf = plot_surface(
                    ax,
                    Z,
                    tau,
                    (
                        f"{session} | {label} | {binary_tag}\n"
                        f"[{meta['t0_sec']:.3f}, {meta['t1_sec']:.3f}) s | trials={meta['n_trials']}"
                    ),
                )
                surfs.append(surf)
                metas.append(meta)

            fig.colorbar(surfs[0], ax=fig.axes, shrink=0.6, pad=0.02)
            ref = metas[0]
            fig.suptitle(
                (
                    f"{label} | {binary_tag} | {ref['anchor']} | "
                    f"window [{ref['t0_sec']:.3f}, {ref['t1_sec']:.3f}) s"
                ),
                fontsize=15,
            )
            plt.tight_layout()

            out = os.path.join(FINAL_DIR, f"{epoch_dir}_{binary_tag}_4up.png")
            fig.savefig(out, dpi=DPI)
            plt.close(fig)
            print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
