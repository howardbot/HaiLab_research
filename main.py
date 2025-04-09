# main.py
from src.Loader import load_mat_session
from src.Features import extract_firing_rate
from src.Models import train_classifier
import os
import numpy as np

# data path
data_dir = "./data"
mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

# name rule
def infer_area_label(filename):
    if "CIP" in filename.upper():
        return 0  # CIP
    elif "V3A" in filename.upper():
        return 1  # V3A
    else:
        raise ValueError(f"can't tell region from the name: {filename}")

# collect data
all_X = []
all_y = []
min_feat_dim = None

for fname in mat_files:
    print(f"Loading {fname}...")
    T = load_mat_session(os.path.join(data_dir, fname))
    X, _ = extract_firing_rate(T, label_type='slant')  # only feature, no need to have slant

    if len(X) == 0:
        print(f"[Warning] {fname} has no valid trials, skipping.")
        continue

    # record smallest feature dimension
    if min_feat_dim is None or X.shape[1] < min_feat_dim:
        min_feat_dim = X.shape[1]

    # put labels
    area_label = infer_area_label(fname)
    y = np.full(len(X), area_label)

    all_X.append(X)
    all_y.append(y)

# feature cut and combine
trimmed_X = [X[:, :min_feat_dim] for X in all_X]
X_all = np.vstack(trimmed_X)
y_all = np.concatenate(all_y)

print(f"\n🔎 Final dataset shape: X = {X_all.shape}, y = {y_all.shape}")
print("Training classifier to decode **brain area (CIP vs V3A)**...\n")

train_classifier(X_all, y_all)
