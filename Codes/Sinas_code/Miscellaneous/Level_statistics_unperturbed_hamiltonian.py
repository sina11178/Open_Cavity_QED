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



def H_0(J, mu, L, Nb, Nd): # Nb --> Number of bosons
    dim = (2**L)
    H = np.zeros((dim, dim), dtype=np.complex128)
    s_dim = 2**L
    #h_i = 2*mu * np.random.rand(L) - mu

    seed = Nd
    #seed = np.random.randint(0, Nd+1)
    rng = np.random.Generator(np.random.MT19937(seed))
    h_i = 2*mu * rng.random(L) - mu
    #print(h_i)
    #h_i = [-0.01802542253143996, 0.07315167189045962, -0.032767256062385564, -0.11333270691216582] # NOTE: This was used to match Koki's code
    #h_i = [-0.01802542253143996, 0.07315167189045962, -0.032767256062385564, -0.11333270691216582, 0.04765435346883812, 0.026099231375528387] # NOTE: This was used to match Koki's code
    for s in range(dim):
        # Diagonal terms
        for i in range(L):
            j = (i + 1) % L
            H[s, s] += J * (1 if ((s >> i) & 1) == ((s >> j) & 1) else -1)
#        for i in range(L):
            H[s, s] += h_i[i] * (-1 if ((s >> i) & 1) == 0 else 1) # >> is a right ward shift (cutting off bits on the right)

    H = np.kron(np.eye(Nb), H)  # Extend to bosonic space
    return H

def H_1(L, Nb):
    dim = 2**L
    H = np.zeros((dim, dim), dtype=np.complex128)

    for s in range(dim):
        # Off-diagonal terms
        for i in range(L):
            s2 = (s) ^ (1 << i)  # << Flip the i-th spin; It is a left ward shift (adding zeros on the right)
            H[s2, s] += 1.0  # NOTE: To understand code, think of H[s2, s] as <s2|H|s> --> i.e. What you would expect this to give you
    H = np.kron(np.eye(Nb), H)  #
    return H

def create_b(Nb, alpha, L):
    """
    Construct truncated bosonic annihilation operator
    in Nb-dimensional Fock basis.
    """
    data = np.sqrt(np.arange(1, Nb))   # √1, √2, ..., √(Nb-1)
    b = np.diag(data, k=1)
    #b.setdiag(-1*alpha)  # Add the displacement term to the diagonal if code doesn't work
    b = np.kron(b, np.eye(2**L))  # Extend to spin space
    return b

def b_dagger_b(omega, b, b_dagger, L):
    H = omega * b_dagger @ b
    return H

# Transition matrix --> Rho SS


def transition_A(U, b, H_number, kappa, omega, L, Nb):
    dim_total = (2**L) * Nb
    A = np.zeros((dim_total, dim_total), dtype=np.complex128)
    # for m in range(dim_total):
    #     U_m = U[:, m].conj().T  # m-th eigenvector of H
    #     for n in range(dim_total):
    #         U_n = U[:, n]  # n-th eigenvector of H
    #         if n == m:
    #             first_term = np.abs(U_m @ b @ U_n)**2
    #             second_term = U_m @ H_number @ U_n 
    #             A[m, n] = first_term - second_term
    #         else:
    #             first_term = np.abs(U_m @ b @ U_n)**2
    #             A[m, n] = first_term
    b_rot = U.conj().T @ b @ U
    A = np.abs(b_rot)**2
    A -= np.diag(np.diag(U.conj().T @ H_number @ U).real)
    A = A.real
    kappa = 1
    A = kappa * A
    #eigenvalues, eigenvectors = spla.eigs(A, k=1, sigma=1e-20, which='LM')  # May need to change sigma...


    #NOTE: THE SPLA METHOD IS FASTER HERE

    #eigvals_A, eigvecs_A = np.linalg.eig(A) # NOTE: You can also use spla.eigs for different accuracy --> This would be best case scenario for sparse matrices
    eigvals_A, eigvecs_A = spla.eigs(A, k=1, sigma=1e-30, which='LM') #NOTE: GAMMA=0 DIFFERS BETWEEN THIS AND the numpy method --> Probably due to a degenracy in spectrum of A and they pick 2 different degenerate eigenvectors

    return eigvals_A, eigvecs_A
    #return eigenvalues, eigenvectors


# Gives c_m vector --> rho_ss = sum(c_m|m><m|)
def rho_ss(U, b, H_number, kappa, omega, L, Nb):
    eigvals_A, eigvecs_A = transition_A(U, b, H_number, kappa, omega, L, Nb)

    idx = np.argmin(np.abs(eigvals_A))
    rho_ss = eigvecs_A[:, idx]
    #rho_ss = eigvecs_A[:,0]

    #rho_ss = ei[:, 0]  # Steady state density matrix (unnormalized)
    if np.sum(rho_ss) < 0:
        rho_ss = -rho_ss   # enforce consistent sign before normalising
    rho_ss = rho_ss / np.sum(rho_ss)  # Normalize
    #return np.real(rho_ss)
    #print("rho_ss min:", rho_ss.min(), "max:", rho_ss.max(), "sum:", rho_ss.sum())
    #print(rho_ss)

    return np.real(rho_ss)

# sigma_y operator for j-th spin (only in spin space)
def sigma_zj(j, L, Nb):
    dim = 2**L
    sigma_z = np.zeros((dim, dim), dtype=np.complex128)
    for s in range(dim):
        sigma_z[s, s] += (1 if ((s >> j) & 1) == 0 else -1) # >> is a right ward shift (cutting off bits on the right)
    sigma_z = np.kron(np.eye(Nb), sigma_z)  # Extend to bosonic space
    #sigma_y = sp.kron(sp.eye(Nb), sigma_y, format='csr')
    return sigma_z

def cal_local_spin(j, rhoss, U, Nb, L):
    spin_j = sigma_zj(j, L, Nb)
    rho_ss_mat = np.diag(rhoss)
    mat = U.conj().T @ spin_j @ U
    mat = mat @ rho_ss_mat
    local_spin_ss = np.trace(mat)

    return local_spin_ss


'''
# LEVEL SPACING STATISTICS
def level_spacing_statistics(eigenvalues):
    """
    Compute level spacing statistics for a set of eigenvalues.
    Returns the r-statistic (ratio of consecutive level spacings).
    
    Oganesyan & Huse (2007): <r> ~ 0.386 for Poisson (integrable)
                              <r> ~ 0.530 for GOE (chaotic/ergodic)
    """
    E = np.sort(eigenvalues)
    
    # Consecutive spacings
    spacings = np.diff(E)
    
    # Remove near-zero spacings (degeneracies)
    spacings = spacings[spacings > 1e-10]
    
    # r-statistic: ratio of consecutive spacings
    r = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    
    mean_r = np.mean(r)
    
    return {
        "r_values": r,
        "mean_r": mean_r,
        "n_levels": len(E),
        "n_spacings": len(spacings),
    }

def plot_level_spacing(eigenvalues, title="Level Spacing Statistics"):
    stats = level_spacing_statistics(eigenvalues)
    r = stats["r_values"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # --- r-statistic distribution ---
    axes[0].hist(r, bins=50, density=True, alpha=0.7, color="steelblue", label="Data")
    x = np.linspace(0, 1, 200)
    # Poisson
    axes[0].plot(x, 2 / (1 + x)**2, 'r--', label=f"Poisson (⟨r⟩≈0.386)")
    # GOE (Wigner-like approximation)
    axes[0].plot(x, (27/4) * (x + x**2) / (1 + x + x**2)**(5/2), 'g--', label=f"GOE (⟨r⟩≈0.530)")
    axes[0].axvline(stats["mean_r"], color='k', linestyle=':', label=f"⟨r⟩ = {stats['mean_r']:.3f}")
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("P(r)")
    axes[0].set_title("r-statistic distribution")
    axes[0].legend()

    # --- Raw level spacings ---
    E = np.sort(eigenvalues)
    spacings = np.diff(E)
    spacings = spacings[spacings > 1e-10]
    mean_s = np.mean(spacings)
    s = spacings / mean_s  # Unfold by mean spacing
    axes[1].hist(s, bins=300, density=True, alpha=0.7, color="steelblue", label="Data")
    x2 = np.linspace(0, 4, 200)
    axes[1].plot(x2, np.exp(-x2), 'r--', label="Poisson: e^{-s}")
    axes[1].plot(x2, (np.pi/2) * x2 * np.exp(-np.pi * x2**2 / 4), 'g--', label="GOE (Wigner surmise)")
    axes[1].set_xlabel("s / ⟨s⟩")
    axes[1].set_ylabel("P(s)")
    axes[1].set_title("Level spacing distribution")
    axes[1].legend()

    fig.suptitle(f"{title} | ⟨r⟩ = {stats['mean_r']:.3f} | N = {stats['n_levels']} levels")
    plt.tight_layout()
    plt.show()
    
    return stats

# KL DIVERGENCE CALCULATION FOR r-STATISTIC DISTRIBUTION
def goe_r_distribution(r):
    return (27/4) * (r + r**2) / (1 + r + r**2)**(5/2)

def poisson_r_distribution(r):
    return 2 / (1 + r)**2


def kl_divergence_r_goe(r_values, bins=50, eps=1e-12):

    # Histogram of numerical data
    hist, edges = np.histogram(
        r_values,
        bins=bins,
        range=(0,1),
        density=True
    )

    # Bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]

    # GOE prediction sampled at bin centers
    p_goe = goe_r_distribution(centers)

    # Normalize discretely
    p_num = hist / np.sum(hist * dr)
    p_goe = p_goe / np.sum(p_goe * dr)

    # Avoid log(0)
    p_num = np.clip(p_num, eps, None)
    p_goe = np.clip(p_goe, eps, None)

    # KL divergence
    dkl = np.sum(p_num * np.log(p_num / p_goe)) * dr

    return centers, p_num, p_goe, dkl


def kl_divergence_r_poisson(r_values, bins=50, eps=1e-12):

    # Histogram of numerical data
    hist, edges = np.histogram(
        r_values,
        bins=bins,
        range=(0,1),
        density=True
    )

    # Bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]

    # GOE prediction sampled at bin centers
    p_goe = poisson_r_distribution(centers)

    # Normalize discretely
    p_num = hist / np.sum(hist * dr)
    p_goe = p_goe / np.sum(p_goe * dr)

    # Avoid log(0)
    p_num = np.clip(p_num, eps, None)
    p_goe = np.clip(p_goe, eps, None)

    # KL divergence
    dkl = np.sum(p_num * np.log(p_num / p_goe)) * dr

    return centers, p_num, p_goe, dkl


def single_disorder(k, base_seed, J, μ, l, Nb, H1, C_H1, H2_scaled, H_number, H_number_norm, b, kappa, ω):
    #kl_values_GOE = []
    #kl_values_Poisson = []
    H0 = H_0(J, μ, l, Nb, k + base_seed)
    H = H0 + (H1 * C_H1) + H2_scaled + H_number

    eigvals, U = np.linalg.eigh(H)

    stats = level_spacing_statistics(eigvals)

    r_values = stats["r_values"]
    centers, p_num, p_goe, KL_GOE = kl_divergence_r_goe(r_values)
    centers, p_num, p_poisson, KL_Poisson = kl_divergence_r_poisson(r_values)
    #kl_values_GOE.append(KL_GOE)
    #kl_values_Poisson.append(KL_Poisson)

    return KL_GOE, KL_Poisson

    #ss = rho_ss(U, b, H_number_norm, kappa, ω, l, Nb)
    #for j in range(l):
    #    spin_j.append(cal_local_spin(j, ss, U, Nb, l))
    #delta_spin = np.var(spin_j)

    #return delta_spin
'''

def single_disorder(k, base_seed, J, μ, l, Nb, H1, C_H1, H2_scaled, H_number, H_number_norm, b, kappa, ω):
    
    H0 = H_0(J, μ, l, Nb, k + base_seed)
    H = H0 + (H1 * C_H1) + H2_scaled + H_number

    # NOTE: I THINK THIS IS THE MOST MEMORY EFFICIENT
    eigvals, U = np.linalg.eigh(H)
    U = np.abs(U)**2
    KL = np.empty(U.shape[1] - 1)
    #avg_eigvals = np.empty(U.shape[1] - 1)
    for n in range(U.shape[1]-1):
        p = U[:, n]
        q = U[:, n+1]
        KL[n] = np.sum(p * (np.log(p) - np.log(q)))
        #avg_eigvals[n] = (eigvals[n]+eigvals[n+1])/2
    #return KL, avg_eigvals
    return KL, eigvals


def main():
    base_seed = 0
    GAMMA = np.linspace(0.001, .85, 50)  # NOTE: IF USING SPLA, DONT USE GAMMA TOO CLOSE TO 0
    #GAMMA = [0]
    J= -1.07
    μ = 1.2 
    Ωd = 4.0
    ω = np.pi / 0.8
    L = [2, 3, 4]
    Nb = 10
    Nd = 10
    debye_omega = 4.0
    kappa = 0
    alpha = 1
    #All_H = []

    for l in L:
        spin_fluctuation = []
        H1 = H_1(l, Nb)
        b = create_b(Nb, alpha=alpha, L=l)
        b_dagger = b.conj().T
        H_number = b_dagger_b(ω, b, b_dagger, l)
        H2 = H1 @ (b + b_dagger)
        for G in GAMMA:
            #spin_j = []
            G = G * np.power(l, 1/2)
            fluctuations = []
            C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
            H2_scaled = H2 * G
            for k in range(Nd):
                spin_j = []
                H0 = H_0(J, μ, l, Nb, k + base_seed)
                H = H0 + (H1 * C_H1) + H2_scaled + H_number

                eigvals, U = np.linalg.eigh(H)
                ss = rho_ss(U, b, H_number/ω, kappa, ω, l, Nb)
                for j in range(l):
                    spin_j.append(cal_local_spin(j, ss, U, Nb, l))
                delta_spin = np.var(spin_j)
                # delta_spin = np.std(spin_j)
                #mean = np.mean(spin_j)
                #fluctuations.append(delta_spin/mean)
                fluctuations.append(delta_spin)
                print("Iteration "+ str(k) + " for L "+ str(l))
            spin_fluctuation.append(np.mean(fluctuations))
        plt.plot(GAMMA * np.power(l, 1/2), spin_fluctuation, label = l)
        #plt.plot(GAMMA, spin_fluctuation, label = l)
        plt.xlabel("Gamma * √L")
        plt.yscale("log")
        plt.ylabel("<δS>")
        plt.title("Spin Fluctuations - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , base_seed = " + str(base_seed)+ " (J, μ, Ωd, ω, Nb) = " + str((J, μ, Ωd, ω, Nb)))
        plt.legend()

    plt.show()
    #print(f"Fluctuation: {spin_fluctuation}")

def main_parallelize():
    base_seed = 0
    #GAMMA = np.linspace(0.001, .85, 25)  # NOTE: IF USING SPLA, DONT USE GAMMA TOO CLOSE TO 0
    GAMMA = [1.05]
    J= 0
    #J = -0.5
    μ = 0.5
    #μ = 1
    Ωd = 4
    #Ωd = 0
    #ω = np.pi / 0.8
    ω = np.pi/0.8
    L = [2, 3, 4, 5]
    Nb = 10
    Nd = 10
    debye_omega = 4.0
    #debye_omega = 0
    kappa = 0
    alpha = 1
    #All_H = []
    #fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    power_scaling = 1
    plt.figure()
    for idx, l in enumerate(L):
        #kl_values = []
        #kl_values_GOE = []
        #kl_values_Poisson = []
        H1 = H_1(l, Nb)
        b = create_b(Nb, alpha=alpha, L=l)
        b_dagger = b.conj().T
        H_number = b_dagger_b(ω, b, b_dagger, l)
        H2 = H1 @ (b + b_dagger)
        for G in GAMMA:
            #spin_j = []
            # G = G/ np.power(l, 1/2)
            G = G * np.power(l, power_scaling)
            fluctuations = []
            C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
            H2_scaled = H2 * G
            H_number_norm = H_number / ω
            result= Parallel(n_jobs=-1)(
                    delayed(single_disorder)(k, base_seed, J, μ, l, Nb, H1, C_H1, 
                                            H2_scaled, H_number, H_number_norm, b, kappa, ω)
                    for k in range(Nd)
                )
            results, results_eigvals = zip(*result)
            KL_data = np.mean(results, axis=0)
            average_eigvals = np.mean(results_eigvals, axis=0)
            #kl_val_goe =     [r[0] for r in results]
            #kl_val_poisson = [r[1] for r in results]
            #kl_values_GOE.append(np.mean(kl_val_goe))
            #kl_values_Poisson.append(np.mean(kl_val_poisson))

        '''
        axes[int(l/2)-1][l % 2].plot(GAMMA*np.sqrt(l), kl_values_GOE, 'g-o', label="KL vs GOE")
        axes[int(l/2)-1][l%2].plot(GAMMA*np.sqrt(l), kl_values_Poisson, 'r-o', label="KL vs Poisson")
        axes[int(l/2)-1][l % 2].set_xlabel(f"Γ√L")
        axes[int(l/2)-1][l % 2].set_ylabel("KL Divergence")
        axes[int(l/2)-1][l%2].legend()
        axes[int(l/2)-1][l%2].set_title(f"L = {l}")
        '''
        plt.plot(average_eigvals[:-1], KL_data, label=f'L={l}') #, marker='o')
        print("Fluctuations for L = " + str(l) + " Complete")
        #plt.plot(GAMMA * np.power(l, 1/2), spin_fluctuation, label = l)
        #plt.plot(GAMMA, spin_fluctuation, label = l)
        #plt.xlabel("Gamma * √L")
        #plt.yscale("log")
        #plt.ylabel("<δS>")
        #plt.title("Spin Fluctuations - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , base_seed = " + str(base_seed)+ " (J, μ, Ωd, ω, Nb) = " + str((J, μ, Ωd, ω, Nb)))
        #plt.legend()
    plt.legend()
    plt.title(f" Level Statistics: Nd = {str(Nd)}, Nb = {str(Nb)}, base_seed = {str(base_seed)}, (J, μ, Ωd, ω, Γ√L) = {str((J, μ, Ωd, ω, GAMMA[0]))}")
    plt.xlabel('E')
    plt.ylabel('KL(n, n+1)')
    #plt.ylim(0, 10)
    
    plt.tight_layout()
    plt.show()
    #print(f"Fluctuation: {spin_fluctuation}")


def trim_20_percent(arr):
    n = len(arr)
    trim_count = int(n * 0.2)
    trimmed_arr = arr[trim_count:n - trim_count]
    return trimmed_arr



def main_parallelize_average():
    base_seed = 0
    #GAMMA = np.linspace(0.001, .85, 25)  # NOTE: IF USING SPLA, DONT USE GAMMA TOO CLOSE TO 0
    GAMMA = np.linspace(0.001, .85, 25)
    J= -0.5
    #J = -0.5
    μ = 0.6
    #μ = 1
    Ωd = 4
    #Ωd = 0
    #ω = np.pi / 0.8
    ω = np.pi/0.8
    L = [2, 3, 4, 5]
    Nb = 10
    Nd = 10
    debye_omega = 4.0
    #debye_omega = 0
    kappa = 0
    alpha = 1
    #All_H = []
    #fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    power_scaling = 1/2
    plt.figure()
    for idx, l in enumerate(L):
        kl_values = []
        H1 = H_1(l, Nb)
        b = create_b(Nb, alpha=alpha, L=l)
        b_dagger = b.conj().T
        H_number = b_dagger_b(ω, b, b_dagger, l)
        H2 = H1 @ (b + b_dagger)
        for G in GAMMA:
            #spin_j = []
            # G = G/ np.power(l, 1/2)
            G = G * np.power(l, power_scaling)
            fluctuations = []
            C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
            H2_scaled = H2 * G
            H_number_norm = H_number / ω
            result= Parallel(n_jobs=-1)(
                    delayed(single_disorder)(k, base_seed, J, μ, l, Nb, H1, C_H1, 
                                            H2_scaled, H_number, H_number_norm, b, kappa, ω)
                    for k in range(Nd)
                )
            results, results_eigvals = zip(*result)
            KL_data = np.mean(results, axis=0)
            #average_eigvals = np.mean(results_eigvals, axis=0)
            trimmed_kl = trim_20_percent(KL_data)
            kl_values.append(np.mean(trimmed_kl))

        plt.plot(GAMMA * np.power(l, power_scaling), kl_values, label=f'L={l}') #, marker='o')
        print("Fluctuations for L = " + str(l) + " Complete")
        #plt.plot(GAMMA * np.power(l, 1/2), spin_fluctuation, label = l)
        #plt.plot(GAMMA, spin_fluctuation, label = l)
        #plt.xlabel("Gamma * √L")
        #plt.yscale("log")
        #plt.ylabel("<δS>")
        #plt.title("Spin Fluctuations - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , base_seed = " + str(base_seed)+ " (J, μ, Ωd, ω, Nb) = " + str((J, μ, Ωd, ω, Nb)))
        #plt.legend()
    plt.legend()
    plt.title(f" Level Statistics: Nd = {str(Nd)}, Nb = {str(Nb)}, base_seed = {str(base_seed)}, (J, μ, Ωd, ω) = {str((J, μ, Ωd, ω))}")
    plt.xlabel('Γ√L')
    plt.ylabel('<KL>')
    #plt.ylim(0, 10)
    
    plt.tight_layout()
    plt.show()
    #print(f"Fluctuation: {spin_fluctuation}")




main_parallelize_average()