"""
根据刺激序列自动恢复 V3A 与 CIP 的同日同步记录 session 配对。

核心思路：
  1. 同一天同步记录的 V3A 和 CIP 文件来自同一次实验，因此每个 trial
     看到的刺激流完全相同，也就是 (slant, tilt) 序列完全一致。
  2. 对每个 session 的 (slant, tilt) 序列做 MD5 hash。
  3. hash 相同的一组 session 就判定为同一次记录，也就是候选配对。

可选验证：
  如果加上 --verify-eventt，会额外比较 STIM_ON 事件时间
  (EID == 118 对应的 EventT)。同步记录通常共享同一个实验时钟，
  所以真正配对的 session 事件时间应当几乎一致。

输出：
  默认写入本文件夹下的 pairs.csv，包含每个 session 及其 partner。
"""

import os
import sys
import csv
import hashlib
import argparse
from collections import defaultdict

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Loader import load_mat_session  # noqa: E402

# EventID.m 中 STIM_ON 的事件编号；用于可选的 EventT 交叉验证。
STIM_ON_ID = 118


def stim_seq_hash(trials):
    """返回一个 session 的刺激序列 hash，以及原始 (slant, tilt) 数组。

    参数
    ----
    trials : list-like
        load_mat_session 读出的 trial 对象列表。每个 trial 需要包含
        slant 和 tilt 字段。

    返回
    ----
    tuple[str, np.ndarray]
        第一个元素是 MD5 hash；第二个元素是 shape=(n_trials, 2)
        的刺激序列数组，第一列 slant，第二列 tilt。
    """
    seq = np.array([(float(t.slant), float(t.tilt)) for t in trials], dtype=np.float64)
    return hashlib.md5(seq.tobytes()).hexdigest(), seq


def stim_on_times(trials):
    """提取每个 trial 的第一个 STIM_ON 时间。

    如果某个 trial 找不到 STIM_ON，就填 np.nan，避免单个坏 trial
    让整个脚本中断。
    """
    out = []
    for t in trials:
        eid = np.atleast_1d(t.EID)
        et = np.atleast_1d(t.EventT)
        idx = np.where(eid == STIM_ON_ID)[0]
        out.append(float(et[idx[0]]) if len(idx) else np.nan)
    return np.array(out)


def discover_sessions(data_dir):
    """扫描数据目录中的 .mat 文件，并从文件名推断脑区。

    约定文件名类似 CIP_1.mat 或 V3A_3.mat，因此第一个下划线前的
    部分就是 area。
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
    sessions = []
    for f in files:
        name = os.path.splitext(f)[0]
        area = name.split("_", 1)[0]
        sessions.append((name, area, os.path.join(data_dir, f)))
    return sessions


def main():
    ap = argparse.ArgumentParser()
    # data 默认在仓库根目录的 data 文件夹；脚本位于 Final/session_pairing，
    # 因此 REPO_ROOT 需要回到 HaiLab_research 根目录。
    ap.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "pairs.csv"))
    ap.add_argument("--verify-eventt", action="store_true",
                    help="Also compare STIM_ON event times across candidate pairs.")
    args = ap.parse_args()

    sessions = discover_sessions(args.data_dir)
    if not sessions:
        print(f"No .mat files in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(sessions)} sessions from {args.data_dir}")
    info = {}
    for name, area, path in sessions:
        T = load_mat_session(path)
        h, _ = stim_seq_hash(T)
        # trials 暂时保留下来，只在 --verify-eventt 时需要；这样不用重复读 mat。
        info[name] = {"area": area, "path": path, "n": len(T), "hash": h, "trials": T}
        print(f"  {name:<10} area={area:<4} n={len(T)} hash={h[:12]}")

    # 将 hash 相同的 session 聚到同一组。正常情况下每组应为 1 个未配对
    # session，或 2 个同步记录 session；如果出现 3 个以上，也会完整输出。
    by_hash = defaultdict(list)
    for name, d in info.items():
        by_hash[d["hash"]].append(name)

    print("\nPairs (sessions sharing identical (slant,tilt) sequence):")
    rows = []
    for h, names in by_hash.items():
        areas = sorted({info[n]["area"] for n in names})
        if len(names) == 1:
            # 单独成组说明没有找到完全相同的刺激序列，partner 留空。
            print(f"  unpaired: {names[0]}")
            rows.append({"group_hash": h[:12], "session": names[0], "area": info[names[0]]["area"], "partner": ""})
            continue
        print(f"  {' <-> '.join(sorted(names))}  (areas: {','.join(areas)})")
        if args.verify_eventt and len(names) == 2:
            a, b = names
            ta = stim_on_times(info[a]["trials"])
            tb = stim_on_times(info[b]["trials"])
            d = ta - tb
            print(f"    STIM_ON time diff: max|.|={np.nanmax(np.abs(d)):.6f}s  std={np.nanstd(d):.6f}s")
        # 对同一组中的每个 session 都写一行；partner 用分号连接，
        # 兼容极少数一组超过两个 session 的情况。
        for n in sorted(names):
            partners = sorted(x for x in names if x != n)
            rows.append({"group_hash": h[:12], "session": n, "area": info[n]["area"],
                         "partner": ";".join(partners)})

    # 稳定排序后写 CSV，方便版本比较和人工检查。
    rows.sort(key=lambda r: (r["group_hash"], r["session"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_hash", "session", "area", "partner"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
