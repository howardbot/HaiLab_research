import matplotlib.pyplot as plt
from src.Loader import load_mat_session
from src.Features import extract_firing_rate
import os
import numpy as np

# Helper function to determine brain area from filename
def infer_area(filename):
    if "CIP" in filename.upper():
        return "CIP"
    elif "V3A" in filename.upper():
        return "V3A"
    else:
        return "UNKNOWN"

# Load all .mat file names
data_dir = "./data"
mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

# First pass: find the minimum number of neurons (features) for each area
min_dim_cip = float('inf')
min_dim_v3a = float('inf')

for fname in mat_files:
    T = load_mat_session(os.path.join(data_dir, fname))
    X, _ = extract_firing_rate(T, label_type='slant')
    if len(X) == 0:
        continue

    area = infer_area(fname)
    if area == "CIP":
        min_dim_cip = min(min_dim_cip, X.shape[1])
    elif area == "V3A":
        min_dim_v3a = min(min_dim_v3a, X.shape[1])

# Second pass: extract and truncate data for CIP and V3A
cip_X = []
v3a_X = []

for fname in mat_files:
    T = load_mat_session(os.path.join(data_dir, fname))
    X, _ = extract_firing_rate(T, label_type='slant')
    if len(X) == 0:
        continue

    area = infer_area(fname)
    if area == "CIP" and X.shape[1] >= min_dim_cip:
        cip_X.append(X[:, :min_dim_cip])
    elif area == "V3A" and X.shape[1] >= min_dim_v3a:
        v3a_X.append(X[:, :min_dim_v3a])

# Combine all trials within each region
cip_X = np.vstack(cip_X)
v3a_X = np.vstack(v3a_X)

# Compute average firing rate per neuron for each region
avg_cip = np.mean(cip_X, axis=0)
avg_v3a = np.mean(v3a_X, axis=0)

# Plot CIP
plt.figure(figsize=(10, 4))
plt.bar(np.arange(len(avg_cip)), avg_cip)
plt.title("Average Firing Rate per Neuron - CIP (All Sessions Combined)")
plt.xlabel("Neuron Index")
plt.ylabel("Avg Firing Rate (Hz)")
plt.tight_layout()
plt.show()

# Plot V3A
plt.figure(figsize=(10, 4))
plt.bar(np.arange(len(avg_v3a)), avg_v3a)
plt.title("Average Firing Rate per Neuron - V3A (All Sessions Combined)")
plt.xlabel("Neuron Index")
plt.ylabel("Avg Firing Rate (Hz)")
plt.tight_layout()
plt.show()
