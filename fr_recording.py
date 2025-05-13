# fr_recording.py（任务 1–4：补充 histogram debug 打印）
import os
import numpy as np
import matplotlib.pyplot as plt
from src.Loader import load_mat_session

# Constants
STIM_ON_ID = 118
STIM_OFF_ID = 120

def extract_spike_times(trial, neuron_map):
    spikes = []
    for tt, unit in neuron_map:
        field = f"UnitT_TT{tt}"
        if not hasattr(trial, field):
            spikes.append([])
            continue
        units = getattr(trial, field)
        if isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, '__len__'):
            if isinstance(units, np.ndarray) and units.dtype != object:
                spikes.append(units)
            elif unit < len(units):
                spikes.append(np.atleast_1d(units[unit]))
            else:
                spikes.append([])
        else:
            spikes.append(np.atleast_1d(units) if unit == 0 else [])
    return spikes

def get_event_time(trial, eid):
    if not hasattr(trial, 'EID') or not hasattr(trial, 'EventT'):
        return None
    eids = np.atleast_1d(trial.EID)
    times = np.atleast_1d(trial.EventT)
    idx = np.where(eids == eid)[0]
    return times[idx[0]] if len(idx) > 0 else None

def build_neuron_map(trial):
    neuron_map = []
    for tt in range(1, 9):
        field = f"UnitT_TT{tt}"
        if hasattr(trial, field):
            units = getattr(trial, field)
            if isinstance(units, (list, tuple, np.ndarray)) and hasattr(units, '__len__'):
                if isinstance(units, np.ndarray) and units.dtype != object:
                    neuron_map.append((tt, 0))
                else:
                    for i in range(len(units)):
                        neuron_map.append((tt, i))
            else:
                neuron_map.append((tt, 0))
    return neuron_map

def compute_population_rate(spike_trials, bin_size=0.01, duration=1.0):
    num_bins = int(duration / bin_size)
    rate = np.zeros(num_bins)
    count = 0
    for neuron_trials in spike_trials:
        for spikes in neuron_trials:
            hist, _ = np.histogram(spikes, bins=num_bins, range=(0, duration))
            rate += hist
            count += 1
    if count > 0:
        rate = rate / count
        rate /= bin_size
    return rate

def plot_raster_and_rate(all_spikes, align_times, neuron_map, fname, label, window, output_dir):
    import matplotlib.gridspec as gridspec

    num_trials = len(all_spikes)
    num_neurons = len(neuron_map)
    colors = plt.cm.tab20(np.linspace(0, 1, num_neurons))
    spacing = 1.2
    height = 0.8
    ytick_positions = [i * spacing + 1 for i in range(num_neurons)]

    fig = plt.figure(figsize=(36, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.2)
    ax_raster = fig.add_subplot(gs[0])
    ax_rate = fig.add_subplot(gs[1])

    bin_size = 0.01
    duration = window[1] - window[0]
    time_bins = np.arange(window[0], window[1], bin_size)
    pop_spikes = []
    all_aligned_latencies = []

    for neuron_idx in range(num_neurons):
        color = colors[neuron_idx % len(colors)]
        center = ytick_positions[neuron_idx]
        all_spike_times = []
        neuron_trial_spikes = []
        for trial_idx in range(num_trials):
            spikes = all_spikes[trial_idx][neuron_idx]
            aligned = np.array(spikes, dtype=float) - align_times[trial_idx]
            trial_spikes = aligned[(aligned >= window[0]) & (aligned <= window[1])]
            all_spike_times.extend(trial_spikes)
            neuron_trial_spikes.append(trial_spikes)
            all_aligned_latencies.extend(trial_spikes)
        pop_spikes.append(neuron_trial_spikes)
        if all_spike_times:
            ax_raster.vlines(all_spike_times, center - height / 2, center + height / 2, color=color, linewidth=0.6)

    ax_raster.set_title(f"{label} Raster - {fname}")
    ax_raster.set_ylabel("Neuron Index")
    ax_raster.set_yticks(ytick_positions)
    ax_raster.set_yticklabels([f"{i}" for i in range(num_neurons)])

    rate = compute_population_rate(pop_spikes, bin_size=bin_size, duration=duration)
    ax_rate.plot(time_bins[:len(rate)], rate, color="black")
    ax_rate.fill_between(time_bins[:len(rate)], 0, rate, alpha=0.3)
    ax_rate.set_ylabel("Rate (Hz)")
    ax_rate.set_xlabel("Time (s) from {}".format(label.lower()))

    for x in time_bins:
        ax_rate.axvline(x, color='gray', alpha=0.2, linestyle='--', linewidth=0.3)

    # Debug: latency histogram
    if label.lower() == "pre stim":
        debug_hist, debug_bins = np.histogram(all_aligned_latencies, bins=10, range=window)
        print(f"[DEBUG] Pre-stim latency histogram (bin counts): {debug_hist}")
        print(f"[DEBUG] Pre-stim histogram bin edges: {debug_bins}")

    fig.subplots_adjust(hspace=0.25)
    save_name = f"combined_raster_rate_{label.lower().replace(' ', '_')}_{os.path.splitext(fname)[0]}.png"
    plt.savefig(os.path.join(output_dir, save_name))
    plt.close()

def analyze_file(fname, window, align_to, label, output_dir):
    print(f"\n[Processing] {fname} [{label}]")
    T = load_mat_session(os.path.join("./data", fname))
    if len(T) == 0:
        print("[WARN] Empty session.")
        return

    neuron_map = build_neuron_map(T[0])
    all_spikes, align_times = [], []
    for trial in T:
        spikes = extract_spike_times(trial, neuron_map)
        align = get_event_time(trial, align_to)
        if align is None:
            continue
        all_spikes.append(spikes)
        align_times.append(align)

    if len(all_spikes) == 0:
        print("[WARN] No valid trials.")
        return

    all_spikes = np.array(all_spikes, dtype=object)
    align_times = np.array(align_times)

    plot_raster_and_rate(all_spikes, align_times, neuron_map, fname, label, window, output_dir)

def main():
    data_dir = "./data"
    output_dir = "raster_rate"
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

    for f in files:
        analyze_file(f, window=(0, 0.5), align_to=120, label="Post Stim", output_dir=output_dir)
        analyze_file(f, window=(-0.5, 0), align_to=118, label="Pre Stim", output_dir=output_dir)
        analyze_file(f, window=(0, 0.5), align_to=118, label="During Stim", output_dir=output_dir)

if __name__ == "__main__":
    main()
