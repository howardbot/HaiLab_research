# neuron_connectivity.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from src.Loader import load_mat_session
from src.Features import extract_firing_rate

data_dir = "./data"
output_dir = "./Connectivity"
os.makedirs(output_dir, exist_ok=True)

mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

for fname in mat_files:
    print(f"\n Processing {fname} ...")
    T = load_mat_session(os.path.join(data_dir, fname))
    X, _ = extract_firing_rate(T, label_type='slant')

    if len(X) == 0 or X.shape[1] < 2:
        print("No valid trials or too few neurons.")
        continue

    # get first 30 neurons
    X = X[:, :30] if X.shape[1] > 30 else X

    # Matrix calculation
    corr_matrix = np.corrcoef(X.T)

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title(f"Neuron Correlation Matrix - {fname}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_corr_{fname}.png"))
    plt.close()

    # Construct the connectivity graph
    threshold = 0.5
    adj = (np.abs(corr_matrix) > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    G = nx.from_numpy_array(adj)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx(G, node_color='skyblue', with_labels=True)
    plt.title(f"Neuron Connectivity Graph (|r| > {threshold}) - {fname}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"graph_corr_{fname}.png"))
    plt.close()

    print(f"✅ Done: heatmap + graph saved for {fname}")
