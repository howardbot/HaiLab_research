# src/features.py
import numpy as np

# 提取对齐后 firing rate 特征
def extract_firing_rate(T, label_type='slant', bin_size=0.001, window=(0, 0.2)):
    X = []  # 特征矩阵（trial × unit）
    y = []  # 标签（slant 或 tilt）

    for trial in T:
        # 获取 stimulus onset 时间
        eid = trial.EID
        etimes = trial.EventT
        stim_idx = np.where(eid == 118)[0]
        if len(stim_idx) == 0:
            continue  # 没有刺激的跳过
        stim_time = etimes[stim_idx[0]]

        features = []  # 当前 trial 的 firing rates

        for tt in range(1, 9):
            tt_field = f'UnitT_TT{tt}'
            if not hasattr(trial, tt_field):
                continue
            tt_units = getattr(trial, tt_field)

            if not isinstance(tt_units, (list, np.ndarray)):
                tt_units = [tt_units]  # 单个 unit

            for unit_spikes in tt_units:
                if unit_spikes is None:
                    features.append(0)
                    continue
                unit_spikes = np.atleast_1d(unit_spikes)
                if len (unit_spikes) == 0:
                    features.append(0)
                    continue
                aligned = np.array(unit_spikes) - stim_time
                spikes_in_window = ((aligned >= window[0]) & (aligned <= window[1])).sum()
                rate = spikes_in_window / (window[1] - window[0])
                features.append(rate)

        if label_type == 'slant':
            label = trial.slant
        elif label_type == 'tilt':
            label = trial.tilt
        else:
            raise ValueError("Unsupported label type")

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)
