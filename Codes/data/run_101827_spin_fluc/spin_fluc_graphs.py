import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd
from scipy.optimize import brentq
import sys
from joblib import Parallel, delayed


l_array = [4, 5, 6, 7, 8]
Nb = 10
Nd = 1000
J = -1.07
mu = 1.3
Omd = 4
omeg = 3.927


gamma_array = np.load(f"./b0/unscaled_gamma.npy")



plt.figure()

power = 0

for l in l_array:
    spatial_var_array = []
    for gamma in gamma_array:
        spatial_var_disorder = []
        file_name_spin = f"./b0/Spin_j_unscaled_gam_{np.round(gamma,3)}_L{l}_J_{J}_mu_{mu}_Omd_{Omd}_omeg_{omeg}_Nb_{Nb}_Nd_{Nd}.npy"
        spin_data = np.load(file_name_spin)
        for j in range(Nd):
            spatial_var_disorder.append(np.var(spin_data[j]))
        spatial_var_array.append(np.mean(spatial_var_disorder))

    gamma_root_l = gamma_array * np.power(l, 1/2)
    extra_scaling = np.power(l, power)
    plt.plot(gamma_root_l * extra_scaling, spatial_var_array, label=f"L = {l}")

plt.xlabel(f"(Γ*√L)*L^{np.round(power, 3)}")
plt.ylabel("Spatial Variance of Spin <δS(Γ*√L)>")
plt.yscale("log")
plt.title(f"Spatial Variance of Spin <δS> vs ΓL^{np.round(power, 3)} for (J, μ, Ω_d, ω, N_b, N_d) = ({J}, {mu}, {Omd}, {omeg}, {Nb}, {Nd})")
plt.legend()

plt.show()
