"""
用“实测数据特征”独立验证 V3A <-> CIP 的 session 配对。

注意：
  detect_pairs.py 使用的是 (slant, tilt) 刺激标签序列来找配对。
  本脚本刻意不使用刺激标签，而是从 EventT、行为变量和 spike response
  中提取特征，作为独立交叉验证。

对每一个 CIP x V3A 组合都会计算四类分数：

  A. Trial-duration fingerprint
     每个 trial 的持续时间：t(STIM_OFF) - t(STIM_ON)。
     如果两个 session 是同一次物理实验同步记录，它们的 trial 时长序列
     应当逐 trial 完全一致。

  B. FixDistance fingerprint
     每个 trial 的行为变量 FixDistance。
     同一只 monkey、同一个 trial 下，两个脑区文件记录的该行为变量应一致。

  C. Evoked-rate Spearman correlation
     每个 trial 的诱发放电：
       [STIM_ON, STIM_ON + 1s] 放电率
       减去 [STIM_ON - 0.5s, STIM_ON] 基线放电率。
     使用 Spearman 秩相关，能降低极端值和不同探针放电尺度差异的影响。

  D. Detrended-evoked Pearson correlation
     仍然使用 C 中的诱发放电，但先减去 25 个 trial 窗口的 running mean，
     去掉 session 内慢漂移，再做 Pearson 相关。

输出：
  1. 每种方法的评分矩阵，行是 CIP，列是 V3A。
  2. 每种方法下的最优一对一 assignment。
  3. 对每个 CIP 的四方法投票结果与最终 consensus pairing。
"""

import os
import sys
import argparse

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Loader import load_mat_session  # noqa: E402

# EventID.m 中使用的事件编号。
STIM_ON_ID = 118
STIM_OFF_ID = 120

# 当前数据中用于计算 population firing rate 的 tetrode 编号。
TETRODES = [2, 3, 4, 5, 6, 7, 8]

# 诱发放电窗口：刺激出现后 1 秒。
ON_WINDOW_SEC = 1.0

# 基线窗口：刺激出现前 0.5 秒。
PRE_WINDOW_SEC = 0.5

# detrend 使用的 running mean 窗口大小，单位是 trial。
DETREND_WIN = 25


def first_event_time(trial, eid):
    """返回一个 trial 中指定 EventID 第一次出现的时间。

    某些 trial 可能缺少指定事件；这时返回 np.nan，让后续统计函数通过
    finite mask 自动忽略。
    """
    e = np.atleast_1d(trial.EID)
    t = np.atleast_1d(trial.EventT)
    idx = np.where(e == eid)[0]
    return float(t[idx[0]]) if len(idx) else np.nan


def trial_durations(trials):
    """计算每个 trial 的刺激持续时间：STIM_OFF - STIM_ON。"""
    return np.array([first_event_time(t, STIM_OFF_ID) - first_event_time(t, STIM_ON_ID)
                     for t in trials])


def fix_distance(trials):
    """提取每个 trial 的 FixDistance 行为变量。

    np.atleast_1d(...).ravel()[0] 用来兼容 MATLAB 标量读入 Python 后
    可能变成 0-D/1-D 数组的情况。
    """
    return np.array([float(np.atleast_1d(t.FixDistance).ravel()[0]) for t in trials])


def _flatten_spike_times(obj):
    """把某个 tetrode 的 spike time 统一整理成一维 float 数组。

    UnitT_TTk 在不同 mat 文件里可能有两种形态：
      1. 普通一维数值数组；
      2. object array，里面每个元素又是一组 spike time，例如 multi-unit。

    后续窗口计数只需要所有 spike time 的并集，因此这里统一 flatten。
    """
    a = np.asarray(obj, dtype=object) if hasattr(obj, "dtype") and obj.dtype == object else np.asarray(obj)
    if a.dtype == object:
        parts = [np.asarray(x, dtype=float).ravel() for x in a.ravel() if np.size(x)]
        return np.concatenate(parts) if parts else np.empty(0)
    return a.astype(float).ravel()


def _count_in(spikes, lo, hi):
    """统计 spikes 中落在半开区间 [lo, hi) 的 spike 数。"""
    return int(np.sum((spikes >= lo) & (spikes < hi))) if spikes.size else 0


def population_rate_evoked(trials, on_win=ON_WINDOW_SEC, pre_win=PRE_WINDOW_SEC):
    """计算每个 trial 的 population evoked rate。

    定义：
      evoked = 刺激后窗口放电率 - 刺激前基线窗口放电率

    这样做可以抵消两个探针各自的慢性 baseline 漂移，同时保留由 stimulus
    sequence 驱动的 trial-by-trial response pattern。

    返回
    ----
    tuple[np.ndarray, np.ndarray]
        evoked: on_rate - pre_rate。
        on_rate: 仅刺激后窗口的放电率，用于打印 session 质量概览。
    """
    on_rate = np.full(len(trials), np.nan)
    pre_rate = np.full(len(trials), np.nan)
    for i, tr in enumerate(trials):
        t_on = first_event_time(tr, STIM_ON_ID)
        if not np.isfinite(t_on):
            continue
        n_on, n_pre = 0, 0
        for tt in TETRODES:
            attr = f"UnitT_TT{tt}"
            if not hasattr(tr, attr):
                continue
            spikes = _flatten_spike_times(getattr(tr, attr))
            # rate 的单位是 spikes/s；不同窗口长度分别除以 on_win/pre_win。
            n_on += _count_in(spikes, t_on, t_on + on_win)
            n_pre += _count_in(spikes, t_on - pre_win, t_on)
        on_rate[i] = n_on / on_win
        pre_rate[i] = n_pre / pre_win
    return on_rate - pre_rate, on_rate


def detrend_runmean(x, win=DETREND_WIN):
    """减去 running mean，去除随 trial 缓慢变化的趋势。

    使用 edge padding 可以避免序列两端因为卷积窗口不足而被系统性压低。
    """
    x = np.asarray(x, dtype=np.float64)
    if win <= 1:
        return x - np.nanmean(x)
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(win) / win
    smooth = np.convolve(xp, kernel, mode="same")[pad:pad + len(x)]
    return x - smooth


def safe_corr(a, b):
    """安全计算 Pearson 相关。

    只使用两个数组都为 finite 的位置；有效样本少于 5 或任一数组无方差时
    返回 np.nan，而不是抛出异常。
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    sa = np.sqrt((a * a).sum()); sb = np.sqrt((b * b).sum())
    if sa == 0 or sb == 0:
        return np.nan
    return float((a * b).sum() / (sa * sb))


def safe_spearman(a, b):
    """安全计算 Spearman 秩相关。

    这里用双重 argsort 得到 rank，再复用 safe_corr 计算 rank 的 Pearson
    correlation。该实现足够用于当前连续型评分；若大量 ties，scipy 的
    spearmanr 会更严格地处理并列秩。
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(np.float64)
    rb = np.argsort(np.argsort(b[m])).astype(np.float64)
    return safe_corr(ra, rb)


def seq_match_score(a, b, atol=1e-9):
    """计算两个 trial-by-trial 序列逐项相同的比例。

    用于 A/B 这类 fingerprint 特征。形状不同直接判为 0，因为 trial 数
    不一致时无法逐 trial 对齐。
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return 0.0
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return 0.0
    return float(np.mean(np.isclose(a[m], b[m], atol=atol)))


def print_matrix(name, cips, v3as, scores, fmt="{:7.4f}"):
    """按固定格式打印 score matrix，方便人工查看哪一列最高。"""
    print(f"\n{name}")
    print(" " * 8 + "  ".join(f"{v:>8s}" for v in v3as))
    for i, c in enumerate(cips):
        row = "  ".join(fmt.format(scores[i, j]) for j in range(len(v3as)))
        print(f"{c:<8s} {row}")


def hungarian_argmax(scores):
    """暴力搜索最大总分的一对一 assignment。

    这里矩阵通常只有 4x4，permutation 数量很小，因此不需要引入 scipy 的
    Hungarian algorithm 依赖。若未来 session 数明显增多，可以替换为
    scipy.optimize.linear_sum_assignment。
    """
    from itertools import permutations
    n = scores.shape[0]
    best, best_perm = -np.inf, None
    for perm in permutations(range(scores.shape[1]), n):
        s = sum(scores[i, perm[i]] for i in range(n))
        if s > best:
            best, best_perm = s, perm
    return list(best_perm), best


def main():
    ap = argparse.ArgumentParser()
    # 脚本位于 Final/session_pairing，因此默认数据目录回到仓库根目录 data。
    ap.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    args = ap.parse_args()

    # 文件名约定：CIP_*.mat 与 V3A_*.mat。排序保证每次运行矩阵行列顺序稳定。
    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".mat"))
    cips = [os.path.splitext(f)[0] for f in files if f.startswith("CIP_")]
    v3as = [os.path.splitext(f)[0] for f in files if f.startswith("V3A_")]
    assert len(cips) == len(v3as), "expected equal CIP/V3A counts"

    print(f"CIP sessions: {cips}")
    print(f"V3A sessions: {v3as}")

    feats = {}
    for name in cips + v3as:
        T = load_mat_session(os.path.join(args.data_dir, f"{name}.mat"))
        evoked, on_rate = population_rate_evoked(T)
        # 所有下游评分都从 feats 读取，避免重复解析 mat 和重复计算 spike rate。
        feats[name] = {
            "dur": trial_durations(T),
            "fix": fix_distance(T),
            "evoked": evoked,
            "evoked_detrend": detrend_runmean(evoked),
        }
        print(f"  loaded {name:<8s} n_trials={len(T)} "
              f"mean_on_rate={np.nanmean(on_rate):.1f}Hz "
              f"mean_evoked={np.nanmean(evoked):.1f}Hz "
              f"unique_FixDistance={len(np.unique(feats[name]['fix']))}")

    nC, nV = len(cips), len(v3as)
    sA = np.zeros((nC, nV))  # trial-duration match fraction，越接近 1 越像配对。
    sB = np.zeros((nC, nV))  # FixDistance match fraction，越接近 1 越像配对。
    sC = np.zeros((nC, nV))  # |Spearman r| of evoked rate，越大越像配对。
    sD = np.zeros((nC, nV))  # |Pearson r| of detrended evoked rate，越大越像配对。

    for i, c in enumerate(cips):
        for j, v in enumerate(v3as):
            # A/B 是“序列是否相同”的指纹分数；C/D 是 response pattern 相关性。
            sA[i, j] = seq_match_score(feats[c]["dur"], feats[v]["dur"], atol=1e-6)
            sB[i, j] = seq_match_score(feats[c]["fix"], feats[v]["fix"])
            sC[i, j] = abs(safe_spearman(feats[c]["evoked"], feats[v]["evoked"]))
            sD[i, j] = abs(safe_corr(feats[c]["evoked_detrend"], feats[v]["evoked_detrend"]))

    print_matrix("A) Trial-duration match fraction (1.0 = identical)", cips, v3as, sA)
    print_matrix("B) FixDistance match fraction", cips, v3as, sB)
    print_matrix("C) |Spearman r| of evoked rate (stim - pre)", cips, v3as, sC)
    print_matrix("D) |Pearson r| of detrended evoked rate", cips, v3as, sD)

    print("\n--- Optimal assignment per method ---")
    for name, S in [("A duration", sA), ("B FixDist", sB), ("C |evoked|", sC), ("D |residual|", sD)]:
        # 对每一种方法分别求全局最优的一对一匹配，而不是逐行贪心。
        perm, total = hungarian_argmax(S)
        pairs = [f"{cips[i]}<->{v3as[perm[i]]}" for i in range(nC)]
        print(f"  {name:<12s}: total={total:7.4f}  {' | '.join(pairs)}")

    # 保存每种方法的 assignment，用于后续对每个 CIP 做多数投票。
    method_perms = {}
    for name, S in [("A", sA), ("B", sB), ("C", sC), ("D", sD)]:
        perm, _ = hungarian_argmax(S)
        method_perms[name] = perm

    print("\n--- Per-CIP vote across methods ---")
    print(f"{'CIP':<8s} {'A':<8s} {'B':<8s} {'C':<8s} {'D':<8s}  consensus")
    consensus = []
    for i, c in enumerate(cips):
        votes = {m: v3as[method_perms[m][i]] for m in "ABCD"}
        # 多数投票：四种方法里被投得最多的 V3A 作为该 CIP 的 consensus。
        from collections import Counter
        win, n_win = Counter(votes.values()).most_common(1)[0]
        consensus.append(win)
        print(f"{c:<8s} " + " ".join(f"{votes[m]:<8s}" for m in "ABCD") + f"  {win}  ({n_win}/4)")

    print("\nFinal data-feature consensus pairing:")
    for c, v in zip(cips, consensus):
        print(f"  {c} <-> {v}")
    if len(set(consensus)) != len(consensus):
        # 如果同一个 V3A 被多个 CIP 投中，说明 consensus 不是合法的一对一配对，
        # 需要回头检查数据或改用整体 assignment。
        print("  WARNING: consensus is not a valid 1-to-1 assignment.")


if __name__ == "__main__":
    main()
