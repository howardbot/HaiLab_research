import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# =========================================================
OUT_DIR = "./demo_video_out"
OUT_MP4 = os.path.join(OUT_DIR, "nested_sliding_demo_full.mp4")
OUT_GIF = os.path.join(OUT_DIR, "nested_sliding_demo_full.gif")

SAVE_MP4 = True
SAVE_GIF = False

FPS = 20
DPI = 120

# ---- full timeline (for visualization only) ----
# make stim_on last longer
T_VIS_MIN = -0.2
T_VIS_MAX = 1.2
DT = 0.001   # 1 ms visual sampling

# ---- event ----
STIM_ON_TIME = 0.0

# ---- outer sliding (和你的代码一致) ----
ON_OUTER_RANGE_SEC = (0.200, 1.000)   # start t in [200ms,1000ms]
ON_OUTER_WIN_SEC   = 0.200            # 200 ms outer window
ON_OUTER_STEP_SEC  = 0.010            # 10 ms step

# ---- inner sliding ----
# take one windows size as a demo is fine
INNER_WIN_SEC  = 0.020                # 20 ms inner window demo
INNER_STEP_SEC = 0.001                # 1 ms step

# ---- tau grid ----
TAU_RANGE_SEC = (-0.150, 0.150)
TAU_STEP_SEC  = 0.001

# ---- colors ----
COLOR_OUTER = "#1f77b4"   # blue
COLOR_INNER = "#d62728"   # red
COLOR_A = "#1f77b4"
COLOR_B = "#ff7f0e"
COLOR_Z = "#9467bd"
COLOR_Z_PREV = "#c5b0d5"

# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def zscore(x):
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s

def corr_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return np.nan
    xx = x[m]
    yy = y[m]
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])

def gaussian(x, mu, sigma, amp=1.0):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def make_toy_signals(t):
    """
    Just a demo, making A and B these two signals different
    """
    rng = np.random.default_rng(7)

    # Signal A
    a = (
        gaussian(t, 0.27, 0.020, 1.2) +
        gaussian(t, 0.40, 0.030, 0.9) +
        gaussian(t, 0.58, 0.025, 1.1) +
        gaussian(t, 0.78, 0.030, 0.8)
    )

    # Signal B
    b = (
        gaussian(t, 0.31, 0.024, 1.0) +
        gaussian(t, 0.44, 0.028, 0.85) +
        gaussian(t, 0.62, 0.028, 1.15) +
        gaussian(t, 0.83, 0.032, 0.72)
    )

    # add small texture/noise
    a += 0.05 * np.sin(2*np.pi*8*t) + 0.03 * rng.normal(size=len(t))
    b += 0.05 * np.sin(2*np.pi*8*(t - 0.01)) + 0.03 * rng.normal(size=len(t))

    a = np.clip(a, 0, None)
    b = np.clip(b, 0, None)
    return a, b

def outer_starts_from_config(start_end, win_sec, step_sec):
    start, end = start_end
    last_start = end - win_sec + 1e-12
    return np.arange(start, last_start + 1e-12, step_sec)

def inner_starts_in_outer(outer_t0, outer_t1, inner_win, inner_step):
    if (outer_t1 - outer_t0) < inner_win:
        return np.array([])
    return np.arange(outer_t0, outer_t1 - inner_win + 1e-12, inner_step)

def tau_grid(tau_range, step):
    t0, t1 = tau_range
    if t0 > t1:
        t0, t1 = t1, t0
    k0 = int(np.ceil(t0 / step))
    k1 = int(np.floor(t1 / step))
    ks = np.arange(k0, k1 + 1, dtype=int)
    return ks * step

def compute_inner_series(signal, t, outer_t0, outer_t1, inner_win, inner_step):
    """
    Inner sliding simulation
    """
    starts = inner_starts_in_outer(outer_t0, outer_t1, inner_win, inner_step)
    vals = []
    for s in starts:
        e = s + inner_win
        m = (t >= s) & (t < e)
        if np.sum(m) == 0:
            vals.append(np.nan)
        else:
            vals.append(np.mean(signal[m]))
    return starts, np.asarray(vals, dtype=float)

def lagged_corr_curve(x, y, tau_vals, inner_step):
    """
    Refer to the code：
      for tau in taus:
          lag_steps = round(tau / step_size)
          corr_pair_at_lag(...)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(len(tau_vals), np.nan, dtype=float)

    n = len(x)
    if n < 3:
        return out

    for i, tau in enumerate(tau_vals):
        lag_steps = int(np.round(tau / inner_step))

        if abs(lag_steps) >= n:
            continue

        if lag_steps >= 0:
            a = x[:n - lag_steps]
            b = y[lag_steps:]
        else:
            L = -lag_steps
            a = x[L:]
            b = y[:n - L]

        out[i] = corr_safe(a, b)

    return out

# =========================================================
# PREPARE DATA
# =========================================================
ensure_dir(OUT_DIR)

t = np.arange(T_VIS_MIN, T_VIS_MAX + DT, DT)
sigA, sigB = make_toy_signals(t)

outer_starts = outer_starts_from_config(
    ON_OUTER_RANGE_SEC,
    ON_OUTER_WIN_SEC,
    ON_OUTER_STEP_SEC
)
tau_vals = tau_grid(TAU_RANGE_SEC, TAU_STEP_SEC)

# =========================================================
# FIGURE
# =========================================================
fig = plt.figure(figsize=(14, 8.2))
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1.0, 1.35, 1.35],
    hspace=0.42,
    wspace=0.28
)

ax_timeline = fig.add_subplot(gs[0, :])
ax_zoom = fig.add_subplot(gs[1, :])
ax_signals = fig.add_subplot(gs[2, 0])
ax_zcurve = fig.add_subplot(gs[2, 1])

fig.suptitle(
    "Nested Sliding Windows Demo (outer range/step and inner range/step matched to analysis)",
    fontsize=15
)

# ---------------- TOP: full timeline ----------------
ax_timeline.set_title("Full timeline with outer sliding window")
ax_timeline.set_xlim(T_VIS_MIN, T_VIS_MAX)
ax_timeline.set_ylim(0, 1)
ax_timeline.set_yticks([])
ax_timeline.set_xlabel("Time relative to StimOn (s)")

ax_timeline.axvline(STIM_ON_TIME, linestyle="--", linewidth=1.2)
ax_timeline.text(STIM_ON_TIME + 0.01, 0.92, "StimOn", fontsize=10)

ax_timeline.plot([T_VIS_MIN, T_VIS_MAX], [0.5, 0.5], linewidth=3)

outer_scan_bg = Rectangle(
    (ON_OUTER_RANGE_SEC[0], 0.18),
    ON_OUTER_RANGE_SEC[1] - ON_OUTER_RANGE_SEC[0],
    0.18,
    facecolor="#dddddd",
    edgecolor="none",
    alpha=0.6
)
ax_timeline.add_patch(outer_scan_bg)
ax_timeline.text(
    ON_OUTER_RANGE_SEC[0] + 0.01,
    0.20,
    "outer scan range",
    fontsize=9,
    verticalalignment="bottom"
)

outer_rect_top = Rectangle(
    (outer_starts[0], 0.34),
    ON_OUTER_WIN_SEC,
    0.32,
    fill=False,
    linewidth=2.5,
    edgecolor=COLOR_OUTER
)
ax_timeline.add_patch(outer_rect_top)

timeline_text = ax_timeline.text(
    0.01, 0.05, "",
    transform=ax_timeline.transAxes,
    fontsize=10.5
)

# ---------------- MIDDLE: zoom inside current outer ----------------
ax_zoom.set_title("Inside the current outer window: inner sliding window samples local activity")
ax_zoom.set_xlim(0, ON_OUTER_WIN_SEC)
ax_zoom.set_xlabel("Local time within outer window (s)")
ax_zoom.set_ylabel("Activity")

zoom_line_A, = ax_zoom.plot([], [], color=COLOR_A, linewidth=2, label="Signal A")
zoom_line_B, = ax_zoom.plot([], [], color=COLOR_B, linewidth=2, label="Signal B")

inner_rect = Rectangle(
    (0.0, -0.02),
    INNER_WIN_SEC,
    1.0,
    fill=False,
    linewidth=2.0,
    edgecolor=COLOR_INNER
)
ax_zoom.add_patch(inner_rect)

ax_zoom.legend(loc="upper right")
zoom_text = ax_zoom.text(
    0.01, 0.96, "",
    transform=ax_zoom.transAxes,
    fontsize=10,
    verticalalignment="top"
)

# ---------------- BOTTOM LEFT: full signals ----------------
ax_signals.set_title("Toy signals across the full response epoch")
ax_signals.set_xlim(T_VIS_MIN, T_VIS_MAX)
ax_signals.set_xlabel("Time relative to StimOn (s)")
ax_signals.set_ylabel("Activity")

ax_signals.plot(t, sigA, color=COLOR_A, linewidth=2, label="Signal A")
ax_signals.plot(t, sigB, color=COLOR_B, linewidth=2, label="Signal B")
ax_signals.legend(loc="upper right")

outer_rect_sig = Rectangle(
    (outer_starts[0], -0.02),
    ON_OUTER_WIN_SEC,
    2.0,
    fill=False,
    linewidth=2.5,
    edgecolor=COLOR_OUTER
)
ax_signals.add_patch(outer_rect_sig)

# ---------------- BOTTOM RIGHT: Z(tau) ----------------
ax_zcurve.set_title("Example local Z(τ) curve for the current outer frame")
ax_zcurve.set_xlim(TAU_RANGE_SEC[0], TAU_RANGE_SEC[1])
ax_zcurve.set_ylim(-1.05, 1.05)
ax_zcurve.set_xlabel("τ (s)")
ax_zcurve.set_ylabel("Correlation / example Z value")

ax_zcurve.axvline(0, linestyle="--", linewidth=1.0)
ax_zcurve.axhline(0, linewidth=1.0)

zcurve_prev_line, = ax_zcurve.plot([], [], color=COLOR_Z_PREV, linewidth=2.0, alpha=0.8, label="Previous outer frame")
zcurve_line, = ax_zcurve.plot([], [], color=COLOR_Z, linewidth=2.5, label="Current outer frame")
ax_zcurve.legend(loc="lower left")

z_text = ax_zcurve.text(
    0.02, 0.96, "",
    transform=ax_zcurve.transAxes,
    fontsize=10,
    verticalalignment="top"
)

# cache previous curve just for visualization
prev_curve_holder = {"curve": None}

# =========================================================
# ANIMATION
# =========================================================
def init():
    zoom_line_A.set_data([], [])
    zoom_line_B.set_data([], [])
    zcurve_prev_line.set_data([], [])
    zcurve_line.set_data([], [])
    timeline_text.set_text("")
    zoom_text.set_text("")
    z_text.set_text("")
    return (
        outer_rect_top, outer_rect_sig, inner_rect,
        zoom_line_A, zoom_line_B,
        zcurve_prev_line, zcurve_line,
        timeline_text, zoom_text, z_text
    )

def update(frame_idx):
    outer_t0 = outer_starts[frame_idx]
    outer_t1 = outer_t0 + ON_OUTER_WIN_SEC

    # move outer rectangles
    outer_rect_top.set_x(outer_t0)
    outer_rect_sig.set_x(outer_t0)

    # timeline text
    timeline_text.set_text(
        f"Outer start range = [{ON_OUTER_RANGE_SEC[0]:.3f}, {ON_OUTER_RANGE_SEC[1]:.3f}] s    "
        f"Outer window = {ON_OUTER_WIN_SEC*1000:.0f} ms    "
        f"Outer step = {ON_OUTER_STEP_SEC*1000:.0f} ms    "
        f"Current frame = [{outer_t0:.3f}, {outer_t1:.3f}] s"
    )

    # zoom current outer
    m = (t >= outer_t0) & (t <= outer_t1)
    tz = t[m] - outer_t0
    Az = sigA[m]
    Bz = sigB[m]

    if len(tz) > 0:
        zoom_line_A.set_data(tz, Az)
        zoom_line_B.set_data(tz, Bz)
        ymax = max(np.max(Az), np.max(Bz), 0.6) * 1.15
    else:
        zoom_line_A.set_data([], [])
        zoom_line_B.set_data([], [])
        ymax = 1.0

    ax_zoom.set_xlim(0, ON_OUTER_WIN_SEC)
    ax_zoom.set_ylim(-0.03, ymax)

    # inner starts inside outer
    starts = inner_starts_in_outer(
        outer_t0, outer_t1,
        INNER_WIN_SEC, INNER_STEP_SEC
    )

    if len(starts) > 0:
        # purely for display: let the red inner box cycle inside the current outer window
        inner_vis_idx = frame_idx % len(starts)
        inner_t0_global = starts[inner_vis_idx]
        inner_t0_local = inner_t0_global - outer_t0
        inner_rect.set_x(inner_t0_local)
        inner_rect.set_width(INNER_WIN_SEC)
    else:
        inner_rect.set_x(0.0)
        inner_rect.set_width(0.0)

    K = len(starts)

    zoom_text.set_text(
        f"Inner window = {INNER_WIN_SEC*1000:.0f} ms\n"
        f"Inner step = {INNER_STEP_SEC*1000:.0f} ms\n"
        f"Inner starts per outer frame = {K}\n"
        f"Formula: starts = arange(t0, t1 - inner_win, inner_step)"
    )

    # build local inner-windowed sequences
    _, seqA = compute_inner_series(
        sigA, t, outer_t0, outer_t1,
        INNER_WIN_SEC, INNER_STEP_SEC
    )
    _, seqB = compute_inner_series(
        sigB, t, outer_t0, outer_t1,
        INNER_WIN_SEC, INNER_STEP_SEC
    )

    # compute example local Z(tau)
    ztau = lagged_corr_curve(
        zscore(seqA), zscore(seqB),
        tau_vals, INNER_STEP_SEC
    )

    # previous frame curve for visual comparison
    prev_curve = prev_curve_holder["curve"]
    if prev_curve is None:
        zcurve_prev_line.set_data([], [])
    else:
        zcurve_prev_line.set_data(tau_vals, prev_curve)

    zcurve_line.set_data(tau_vals, ztau)
    prev_curve_holder["curve"] = ztau.copy()

    # difference from previous frame, just to show "frame-to-frame change"
    if prev_curve is None:
        dz = np.nan
    else:
        mm = np.isfinite(prev_curve) & np.isfinite(ztau)
        dz = np.sqrt(np.mean((ztau[mm] - prev_curve[mm])**2)) if np.any(mm) else np.nan

    z_text.set_text(
        f"τ range = [{TAU_RANGE_SEC[0]*1000:.0f}, {TAU_RANGE_SEC[1]*1000:.0f}] ms\n"
        f"τ step = {TAU_STEP_SEC*1000:.0f} ms\n"
        f"Current outer index = {frame_idx}\n"
        f"Example frame-to-frame curve change = {dz:.3f}" if np.isfinite(dz)
        else
        f"τ range = [{TAU_RANGE_SEC[0]*1000:.0f}, {TAU_RANGE_SEC[1]*1000:.0f}] ms\n"
        f"τ step = {TAU_STEP_SEC*1000:.0f} ms\n"
        f"Current outer index = {frame_idx}\n"
        f"Example frame-to-frame curve change = N/A"
    )

    return (
        outer_rect_top, outer_rect_sig, inner_rect,
        zoom_line_A, zoom_line_B,
        zcurve_prev_line, zcurve_line,
        timeline_text, zoom_text, z_text
    )

anim = FuncAnimation(
    fig,
    update,
    frames=len(outer_starts),
    init_func=init,
    interval=1000 / FPS,
    blit=False,
    repeat=False
)

# =========================================================
# SAVE
# =========================================================
if SAVE_MP4:
    writer = FFMpegWriter(
        fps=FPS,
        codec="mpeg4",
        bitrate=2500
    )
    anim.save(OUT_GIF, writer=writer, dpi=DPI)
    print(f"[OK] saved gif: {OUT_GIF}")


if SAVE_GIF and not os.path.exists(OUT_GIF):
    try:
        writer = PillowWriter(fps=FPS)
        anim.save(OUT_GIF, writer=writer, dpi=DPI)
        print(f"[OK] saved gif: {OUT_GIF}")
    except Exception as e:
        print(f"[WARN] gif save failed: {e}")

plt.show()