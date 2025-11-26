import os
import numpy as np
from src.Loader import load_mat_session

STIM_ON_ID = 118
TAU_MS = 5.0   # fixed lag = 5 ms
BIN_SIZE = 0.005  # 5 ms bins
WINDOW = (0.0, 0.2)  # Only early window 0–200ms

def get_event_time(trial, eid):
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids = np.atleast_1d(trial.EID)
    times = np.atleast_1d(trial.EventT)
    idx = np.where(eids == eid)[0]
    return times[idx[0]] if len(idx) > 0 else None


# ======= Build neuron map & spike extraction (simple version) ========
def build_neuron_map(trial):
    nm = []
    for tt in range(1, 9):
        field = f"UnitT_TT{tt}"
        if hasattr(trial, field):
            units = getattr(trial, field)
            if isinstance(units, np.ndarray) and units.dtype != object:
                nm.append((tt, 0))
            else:
                for i in range(len(units)):
                    nm.append((tt, i))
    return nm

def extract_spikes(trial, nm):
    out = []
    for tt, unit in nm:
        units = getattr(trial, f"UnitT_TT{tt}", [])
        if isinstance(units, np.ndarray) and units.dtype != object:
            out.append(units)
        else:
            if unit < len(units):
                out.append(np.atleast_1d(units[unit]))
            else:
                out.append(np.array([]))
    return out


# ========== Compute binned spike counts ==========
def build_binned_matrix(all_spikes, stim_ons, window, binsize):
    T = len(all_spikes)
    N = len(all_spikes[0])
    t_edges = np.arange(window[0], window[1] + binsize, binsize)
    B = len(t_edges) - 1

    mat = np.zeros((N, T, B), dtype=float)

    for n in range(N):
        for t in range(T):
            sp = np.asarray(all_spikes[t][n]) - stim_ons[t]
            sp = sp[(sp >= window[0]) & (sp <= window[1])]
            cnt, _ = np.histogram(sp, bins=t_edges)
            mat[n, t, :] = cnt

    return mat, B


# ========== Compute correlation at fixed tau = 5 ms = 1 bin ==========
def corr_at_fixed_lag(Xi, Xj, lag_bins=1):
    # Xi, Xj: shape (T, B)
    T, B = Xi.shape
    if lag_bins >= B:
        return np.nan

    X1 = Xi[:, :B-lag_bins]
    X2 = Xj[:, lag_bins:]

    x = X1.reshape(-1)
    y = X2.reshape(-1)

    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


# ============================================================
# =========================== MAIN ============================
# ============================================================

def main():
    files = [f for f in os.listdir("./data") if f.endswith(".mat")]

    for fname in files:
        print(f"\n===== {fname} =====")
        T = load_mat_session(os.path.join("./data", fname))
        if len(T) == 0:
            print("  Empty session")
            continue

        neuron_map = build_neuron_map(T[0])
        if len(neuron_map) == 0:
            print("  No neurons")
            continue

        all_spikes = []
        stim_ons = []
        for trial in T:
            stim = get_event_time(trial, STIM_ON_ID)
            if stim is None:
                continue
            all_spikes.append(extract_spikes(trial, neuron_map))
            stim_ons.append(stim)

        all_spikes = np.array(all_spikes, dtype=object)
        stim_ons = np.array(stim_ons)
        Tn = len(all_spikes)
        N = len(neuron_map)
        print(f"  Trials: {Tn}, Neurons: {N}")

        # Build binned matrix at 5 ms resolution
        rates, B = build_binned_matrix(all_spikes, stim_ons, WINDOW, BIN_SIZE)
        print(f"  Binned: B={B} bins (5ms each)")

        # Compute correlation for each pair at fixed lag=5ms
        corr_mat = np.full((N, N), np.nan)

        lag_bins = int(round(TAU_MS / (BIN_SIZE*1000)))  # Should be exactly 1
        lag_bins = 1

        for i in range(N):
            for j in range(N):
                if i==j:
                    continue
                corr_mat[i,j] = corr_at_fixed_lag(rates[i], rates[j], lag_bins=lag_bins)

        # Print summary
        valid_corrs = corr_mat[~np.isnan(corr_mat)]
        if valid_corrs.size == 0:
            print("  All correlations are NaN.")
            continue

        print(f"  Fixed τ = +5ms : max corr = {np.max(valid_corrs):.3f}, mean = {np.mean(valid_corrs):.3f}")

        # Print the correlation matrix itself
        print("  Corr matrix (τ=5ms):")
        print(np.round(corr_mat, 3))


if __name__ == "__main__":
    main()
