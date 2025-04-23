# latency_connectivity.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from src.Loader import load_mat_session
from src.LatencyFeatures import extract_latency_matrix

data_dir = "./data"
output_dir = "./LatencyConnectivity"
os.makedirs(output_dir, exist_ok=True)

mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
threshold = 0.5  # correlation threshold for connectivity

for fname in mat_files:
    print(f"\nProcessing {fname}...")
    T = load_mat_session(os.path.join(data_dir, fname))
    L = extract_latency_matrix(T)

    if L.shape[0] < 5 or L.shape[1] < 2:
        print("Too few valid trials or neurons.")
        continue

    # Remove neurons with too many NaNs
    valid_neuron_mask = np.sum(~np.isnan(L), axis=0) > (0.5 * L.shape[0])
    L = L[:, valid_neuron_mask]
    L = np.where(np.isnan(L), np.nanmean(L, axis=0), L)  # fill remaining NaNs

    corr_matrix = np.corrcoef(L.T)

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title(f"Latency Correlation - {fname}")
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_latency_{fname}.png"))
    plt.close()

    # Graph
    adj = (np.abs(corr_matrix) > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, seed=42,k=0.8)


    plt.figure(figsize=(8, 8))
    nx.draw_networkx(G, pos,node_color='lightgreen', with_labels=True)
    plt.title(f"Latency Connectivity Graph (|r| > {threshold}) - {fname}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"graph_latency_{fname}.png"))
    plt.close()

    print(f"Done: {fname} latency heatmap + graph saved.")
