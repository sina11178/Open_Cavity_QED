import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd
from scipy.optimize import brentq
import os


# NOTE: We want to plot temp_fluctuations from the .npz files - It is called 'mean_fluctuations'

data = {4: [], 5: []}  # store results separately for L=4 and L=5

for fname in os.listdir("."):
    if not fname.endswith(".npz"):
        continue

    if "temp_stat_scaled_gamma" not in fname:
        continue

    # Detect L from filename
    if "_L_4_" in fname:
        L = 4
    elif "_L_5_" in fname:
        L = 5
    else:
        continue

    with np.load(fname) as f:
        gamma = f["scaled_gamma"]
        mean_fluc = f["mean_fluctuations"]

    data[L].append((float(gamma), float(mean_fluc)))


# PLOTTING
for L in data:
    data[L].sort(key=lambda x: x[0])

for L in [4, 5]:
    gammas = [x[0] for x in data[L]]
    fluctuations = [x[1] for x in data[L]]

    plt.plot(gammas, fluctuations, marker='o', label=f"L={L}")

plt.xlabel("scaled_gamma")
plt.ylabel("mean_fluctuations")
plt.yscale("log")  # Use logarithmic scale for better visibility
plt.legend()
plt.show()