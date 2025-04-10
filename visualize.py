import matplotlib.pyplot as plt
import seaborn as sns
from src.Loader import load_mat_session
from src.Features import extract_firing_rate
import os
import numpy as np
output_dir = "./Screenshots"
os.makedirs(output_dir,exist_ok=True)
data_dir = "./data"
mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

for fname in mat_files:
    print(f"Loading {fname}...")
    T = load_mat_session(os.path.join(data_dir, fname))
    X, y = extract_firing_rate(T, label_type='slant')

    if len(X) == 0:
        continue

    # take first 30 neurons
    if X.shape[1] > 30:
        X = X[:, :30]

    # heatmap (colors mean the responding frequency)
    plt.figure(figsize=(10, 6))
    sns.heatmap(X, cmap="viridis", cbar=True)
    plt.title(f"Firing Rate Heatmap - {fname}")
    plt.xlabel("Neuron (unit)")
    plt.ylabel("Trial")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"heatmap - {fname}.png"))
    #plt.show()

    # average firing rate
    avg_firing = np.mean(X, axis=0)

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(avg_firing)), avg_firing)
    plt.title(f"Average Firing Rate per Neuron - {fname}")
    plt.xlabel("Neuron index")
    plt.ylabel("Avg firing rate (Hz)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"average - {fname}.png"))
    #plt.show()
