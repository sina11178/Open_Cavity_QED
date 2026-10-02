import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd
from scipy.optimize import brentq
import sys
from joblib import Parallel, delayed


l_array = [2, 3, 4, 5, 6, 7]
Nb = 10
Nd = 1000
J = -0.1
mu = 1.3
Omd = 4
omeg = 3.927

gamma_array = np.load(f"./b0/unscaled_gamma.npy")

plt.figure()
for l in l_array:
    file_name_fluc = f"./b0/fluc_half_scale_L{l}_J_{J}_mu_{mu}_Omd_{Omd}_omeg_{omeg}_Nb_{Nb}_Nd_{Nd}.npy"
    file_name_std = f"./b0/fluc_std_half_scale_L{l}_J_{J}_mu_{mu}_Omd_{Omd}_omeg_{omeg}_Nb_{Nb}_Nd_{Nd}.npy"

    scaled_gamma_array = np.power(l, 1/2) * gamma_array
    fluc = np.load(file_name_fluc)
    fluc_std = np.load(file_name_std)

    line, = plt.plot(scaled_gamma_array, fluc, label = l, marker="o")
    plt.errorbar(scaled_gamma_array, fluc, yerr=fluc_std/np.sqrt(Nd-1), fmt='o', capsize=5, color=line.get_color())

#plt.ylim(-5, 5)
#plt.xlim(0, 0.5)
#plt.yscale('log')
plt.xlabel("Gamma * √L")
plt.ylabel("<δT/T>")
plt.title("Temperature Fluctuations - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , (J, μ, Ωd, ω, Nb) = " + str((J, mu, Omd, omeg, Nb)))
plt.legend()
plt.show()
