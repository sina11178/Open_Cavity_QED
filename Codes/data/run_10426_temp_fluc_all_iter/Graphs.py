import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd
from scipy.optimize import brentq
import sys
from joblib import Parallel, delayed


l_array = [2, 3, 4, 5, 6]
Nb = 10
Nd = 1000
J = -1.07
mu = 1.3
Omd = 4
omeg = 3.927

l = l_array[4]
gamma_array = np.load(f"./b0/unscaled_gamma.npy")

gamma = gamma_array[0]

file_name_temp = f"./b0/Tj_unscaled_gam_{np.round(gamma,3)}_L{l}_J_{J}_mu_{mu}_Omd_{Omd}_omeg_{omeg}_Nb_{Nb}_Nd_{Nd}.npy"

temp_data = np.load(file_name_temp)

def histogram_over_temp(data, bins=50):
    data = data[:, 0]
    Gamma_scaled = gamma * np.sqrt(l)
    plt.hist(data, bins=bins)
    plt.xlabel("Temperature T_0")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Temperature over all disorder at site 0  for Γ√L = {np.round(Gamma_scaled, 3)}, L = {l}- (J, μ, Ω_d, ω, N_b, N_d) = ({J}, {mu}, {Omd}, {omeg}, {Nb}, {Nd})")
    plt.show()

def histogram_over_temp_disorder_avg(data, bins=50):
    data = np.mean(data, axis=0)
    Gamma_scaled = gamma * np.sqrt(l)
    plt.hist(data, bins=bins)
    plt.xlabel("Temperature T_0")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Temperature over all disorder at site 0  for Γ√L = {np.round(Gamma_scaled, 3)}, L = {l}- (J, μ, Ω_d, ω, N_b, N_d) = ({J}, {mu}, {Omd}, {omeg}, {Nb}, {Nd})")
    plt.show()

def histogram_over_std(data, bins=50):
    data = np.std(data, axis=1)
    Gamma_scaled = gamma * np.sqrt(l)
    plt.hist(data, bins=bins)
    plt.xlabel("Temperature T_0")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Temperature over all disorder at site 0  for Γ√L = {np.round(Gamma_scaled, 3)}, L = {l}- (J, μ, Ω_d, ω, N_b, N_d) = ({J}, {mu}, {Omd}, {omeg}, {Nb}, {Nd})")
    plt.show()

histogram_over_temp(temp_data)
#histogram_over_temp_disorder_avg(temp_data)
#histogram_over_std(temp_data)