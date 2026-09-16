import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd
from scipy.optimize import brentq
from joblib import Parallel, delayed

# For our model, we will consider (bosons) ⊗ (spin)

# Hamiltonian parts

'''
4/6/26

NOTE to self:

- For this code, I have used Kokis sign convention for sig_Z:
    - |0> --> -1
    - |1> --> +1
- This sign convention was used for my temp fluctuation code, as well as this code
    - Going on, I keep using this sign convention unless specified
    - |0> --> -1
    - |1> --> +1
- I got magnetization at each site using -sigma_zj (rather than using koki's sign convention); the results are the same, as I only plot std here
    - If I plotted magnetization at each site, I would see a sign difference at most; however, the physics would remain the same


'''

def Hamiltonian_numpy(L, hz, J):
    rows, cols, data = [], [], []
    dim = 2**L
    H = np.zeros((dim, dim), dtype=np.complex128)
    for s in range(dim):
        # Diagonal terms
        for i in range(L):
            j = (i+1)%L
            H[s, s] += J * (1 if ((s >> i) & 1) == ((s >> j) & 1) else -1)
            H[s, s] += hz * (1 if ((s >> i) & 1) == 0 else -1) # >> is a right ward shift (cutting off bits on the right)
            if (s >> i) & 1 != (s >> j) & 1:
                s_prime = s ^ (1 << i) ^ (1 << j)
                H[s_prime, s] += 2*J
    return H

def Hamiltonian_numpy_disorder(L, hz, J, μ, Nd):
    dim = 2**L
    seed = Nd
    rng = np.random.Generator(np.random.MT19937(seed))
    h_i = 2*μ * rng.random(L) - μ  #  Disorder List for each site

    H = np.zeros((dim, dim), dtype=np.complex128)
    for s in range(dim):
        # Diagonal terms
        for i in range(L):
            j = (i+1)%L
            H[s, s] += (J/4) * (1 if ((s >> i) & 1) == ((s >> j) & 1) else -1)
            H[s, s] += (hz + h_i[i])/2 * (1 if ((s >> i) & 1) == 0 else -1) # >> is a right ward shift (cutting off bits on the right)
            if (s >> i) & 1 != (s >> j) & 1:  # NOTE: Particle hopping in fermion language; Create truth table to see where this is coming from
                s_prime = s ^ (1 << i) ^ (1 << j)
                H[s_prime, s] += 2*(J/4)
    return H

# NOTE: HEISENBERG CHAIN
def Hamiltonian_sparse(L, hz, J):
    rows, cols, data = [], [], []

    dim = 2**L
    #H = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for s in range(dim):
        diag = 0
        # Diagonal terms
        for i in range(L):
            j = (i+1)%L
            diag += J * (1 if ((s >> i) & 1) == ((s >> j) & 1) else -1)
            diag += hz * (1 if ((s >> i) & 1) == 0 else -1) # >> is a right ward shift (cutting off bits on the right)
        cols.append(s)
        rows.append(s)
        data.append(diag)
    H = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    return H

def find_sz(length, sz_sector):
    list_number = np.array(range(2**length))
    bit_count = np.array([i.bit_count() for i in list_number])
    sector_of_each_state = (bit_count - length) + bit_count
    indices = np.where(sector_of_each_state == sz_sector)[0]
    return indices


def single_disorder(k, base_seed, J, μ, l, hz, Sz = 0):
    
    H = Hamiltonian_numpy_disorder(l, hz, J, μ, k+base_seed)
    # NOTE: I THINK THIS IS THE MOST MEMORY EFFICIENT
    finding_sz = find_sz(l, Sz)
    H_sub = H[np.ix_(finding_sz, finding_sz)]
    eigvals, U = np.linalg.eigh(H_sub)
    U = np.abs(U)**2
    KL = []
    #avg_eigvals = np.empty(U.shape[1] - 1)
    for n in range(U.shape[1] - 1):
        p = U[:, n]
        q = U[:, n + 1]
        KL.append(np.sum(p * (np.log(p) - np.log(q))))
    return KL, eigvals

'''
def main_no_disorder():
    J= 1
    hz = 5
    L = [4, 5, 6]
    plt.figure()
    for idx, l in enumerate(L):
        H = Hamiltonian_numpy(l, hz, J)
        KL, eigvals = KL_no_disorder(H)
        
        plt.plot(eigvals[:-1], KL, label=f'L={l}', marker='o')
        print("Fluctuations for L = " + str(l) + " Complete")
    plt.legend()
    plt.title(f" Longitudinal field ising chain: (J, hz) = {str((J, hz))}")
    plt.xlabel('E')
    plt.ylabel('KL(n, n+1)')
    #plt.ylim(0, 10)
    
    plt.tight_layout()
    plt.show()
    #print(f"Fluctuation: {spin_fluctuation}")
'''
def main_parallelize():
    base_seed = 900
    J= 1
    μ = 5
    hz= 0
    L = [12]
    Nd = 10
    plt.figure()
    for idx, l in enumerate(L):
        result= Parallel(n_jobs=-1)(
                delayed(single_disorder)(k, base_seed, J, μ, l, hz)
                for k in range(Nd)
            )
        results, results_eigvals = zip(*result)
        KL_data = np.mean(results, axis=0)
        average_eigvals = np.mean(results_eigvals, axis=0)
        plt.plot(average_eigvals[:-1], KL_data, label=f'L={l}', marker='o')
        print("Fluctuations for L = " + str(l) + " Complete")
    plt.legend()
    plt.title(f" Level Statistics of Ising model w/ longitudinal Ising model: Nd = {str(Nd)}, base_seed = {str(base_seed)}, (J, μ, hz) = {str((J, μ, hz))}, Sz Sector = {str(0)}")
    plt.xlabel('E')
    plt.ylabel('KL(n, n+1)')
    
    plt.tight_layout()
    plt.show()


main_parallelize()