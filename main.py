from src.Loader import load_mat_session
from src.Features import extract_firing_rate
from src.Models import train_classifier
import os
import numpy as np

data_dir = "./data"
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
X_all = []
Y_all = []
for fname in mat_files:
    print(f'Loading{fname}...')
    T = load_mat_session(os.path.join(data_dir,fname))
    X,Y = extract_firing_rate(T,label_type='slant')
    X_all.append(X)
    Y_all.append(Y)

X_all = np.vstack(X_all)
Y_all = np.concatenate(Y_all)

train_classifier(X_all,Y_all)


