import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd
from scipy.optimize import brentq
import sys
# For our model, we will consider (bosons) ⊗ (spin)

# Hamiltonian parts

def H_0(J, mu, L, Nb, Nd): # Nb --> Number of bosons
    dim = (2**L)
    H = np.zeros((dim, dim), dtype=np.complex128)
    s_dim = 2**L
    #h_i = 2*mu * np.random.rand(L) - mu

    #seed = 10 # NOTE: Currently hardcoded -> Must fix this
    seed = Nd
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

    eigvals_A, eigvecs_A = np.linalg.eig(A) # NOTE: You can also use spla.eigs for different accuracy --> This would be best case scenario for sparse matrices

    idx = np.argmin(np.abs(eigvals_A))
    #print(eigvecs_A[:, idx])

    return eigvals_A, eigvecs_A
    #return eigenvalues, eigenvectors


'''
def transition_A(U, b, H_number, kappa, omega, L, Nb):
    dim_total = (2**L) * Nb
    first_term = U.conj().T @ b @ U
    first_term = np.abs(first_term)**2
    b_dag_b = H_number / omega
    second_term = np.diag(np.diagonal(U.conj().T @ b_dag_b @ U))
    kappa = 1
    A = first_term - second_term
    A = kappa * A
    eigenvalues, eigenvectors = spla.eigsh(A, k=1, sigma=1e-15, which='LM')  # May need to change sigma...
    return eigenvalues, eigenvectors
'''


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

    return np.real(rho_ss)

# sigma_y operator for j-th spin (only in spin space)
def sigma_yj(j, L, Nb):
    dim = 2**L
    sigma_y = np.zeros((dim, dim), dtype=np.complex128)
    for s in range(dim):
        s2 = s ^ (1 << j)  # Flip the j-th spin
        sigma_y[s2, s] += 1j * (1 if ((s >> j) & 1) == 0 else -1) # Double check this
    sigma_y = np.kron(np.eye(Nb), sigma_y)  # Extend to bosonic space
    #sigma_y = sp.kron(sp.eye(Nb), sigma_y, format='csr')
    return sigma_y

'''
def sigma_jy_2(j, L, U, Nb):
    sigma_jy = sigma_yj(j, L, Nb)
    dim = (2**L) *Nb
    sig_y_squared = np.zeros((dim, dim), dtype=np.complex128)
    for m in range(dim):
        bra_m = U[:, m].conj().T  # m-th eigenvector of H
        for n in range(dim):
            ket_n = U[:, n]  # n-th eigenvector of H
            sig_y_squared[m, n] = np.abs(bra_m @ sigma_jy @ ket_n)**2
    return sig_y_squared
'''
def sigma_jy_2(j, L, U, Nb):
    sigma_jy = sigma_yj(j, L, Nb)
    # Rotate operator to energy basis
    A = U.conj().T @ sigma_jy @ U

    # Square matrix elements
    return np.abs(A)**2


'''
# Calculate energy current NOTE: ERRONEOUS HERE....
def cal_ecur(sig_jy_squared, rho_ss, eigvals, Tem, debye, small_gamma = 1):
    ecur = 0.0
    for m in range(len(eigvals)):
        for n in range(len(eigvals)):
            dE = eigvals[m] - eigvals[n]  # NOTE: We get a 0 value here... (problematic...)
            if abs(dE) > 1e-12:
                #if abs(dE/Tem) < 1e-15:
                #    continue
                if abs(dE/Tem) <1.0e-6:
                    n_B = Tem / dE  # Use limit of Bose-Einstein distribution for small dE/Tem
                    #print("Use approx")
                else:
                    n_B = 1.0 / (np.exp(dE / Tem) - 1)  # Bose-Einstein distribution
                ecur += (dE**2)* sig_jy_squared[m, n] * n_B *  rho_ss[n] / (dE**4 + debye**4)
    constant_factor = 4 * np.pi * small_gamma # Adjust this factor as needed --> I will absorb, as it's just a factor
    ecur = ecur 
    return ecur
'''

def cal_ecur(sig_jy_squared, rho_ss, eigvals, Tem, debye, small_gamma = 1):
    
    dE = eigvals[:, None] - eigvals[None, :]

    #mask = np.abs(dE) > 1e-12
    mask = np.abs(dE) != 0

    bose = np.zeros_like(dE)

    small = mask & (np.abs(dE/Tem) < 1e-6)
    bose[small] = Tem / dE[small]

    #large = mask & (~small)
    #bose[large] = 1/(np.exp(dE[large]/Tem) - 1)
    x = dE/Tem
    large = x < 700
    bose[large & ~small & mask] = 1/(np.exp(x[large & ~small & mask]) - 1)

    ecur = np.sum(
        (dE**2) *
        sig_jy_squared *
        bose *
        rho_ss[None,:] /
        (dE**4 + debye**4)
    )

    return ecur


#NOTE: MAY BE ERRONEOUS
def cal_slope(sig_jy_squared, rho_ss, eigvals, debye, small_gamma = 1):
    slope = 0.0
    for m in range(len(eigvals)):
        for n in range(len(eigvals)):
            dE = eigvals[m] - eigvals[n]
#                dE = eigvals[m] - eigvals[n]
            slope += dE* sig_jy_squared[m, n] * rho_ss[n] / (dE**4 + debye**4)
    return slope    


# NOTE: You get imaginary and negative temperatures at a certain point
def cal_localT(j, rhoss, eigvals, U, Nb, debye, L, small_gamma = 1):
    sigma_jy_squared = sigma_jy_2(j, L, U, Nb)
    #print("sigma_jy_squared symmetric:", np.allclose(sigma_jy_squared, sigma_jy_squared.T))
    #print("slope positive check - sum of dE*sjy2*rho terms, first 5 eigvals:", eigvals[:5])



    ite = 15
    Tem = np.zeros(ite, dtype=np.float64)
    ecur = np.zeros(ite, dtype=np.float64) # Double check dtype
    slope = cal_slope(sigma_jy_squared, rhoss, eigvals, debye, small_gamma)

    Tem[0] = 500
    ecur[0] = cal_ecur(sigma_jy_squared, rhoss, eigvals, Tem[0], debye, small_gamma)
    Tem[1] = Tem[0] - ecur[0]/slope
    #print(f"j={j}, slope={slope:.6e}, ecur[0]={ecur[0]:.6e}, Tem[1]={Tem[1]:.6f}")

    '''
    dE_mat = eigvals[:, None] - eigvals[None, :]
    DB = dE_mat / (dE_mat**4 + debye**4)
    terms = sigma_jy_squared * rhoss[None, :] * DB  # note: rho indexed on n (columns)
    #print(f"j={j}, slope={slope:.6e}, sum of positive terms={terms[terms>0].sum():.6e}, sum of negative terms={terms[terms<0].sum():.6e}")
    #print(f"Rho_ss: {rhoss.shape}")
    print(f"j={j}, sig_jy_squared min={sigma_jy_squared.min():.3e}, max={sigma_jy_squared.max():.3e}, any negative={np.any(sigma_jy_squared < 0)}")
    '''

    for i in range(1, ite-1):
        ecur[i] = cal_ecur(sigma_jy_squared, rhoss, eigvals, Tem[i], debye, small_gamma)
        if np.abs(ecur[i] - ecur[i-1]) < 1e-12 and np.abs(ecur[i]) < 1e-12:  # Check for convergence
            #print(f"Converged at iteration {i} with local temperature {Tem[i]:.6f}")
            Tem[ite-1] = Tem[i]
            break
        Tem[i+1] = (ecur[i-1]*Tem[i] - Tem[i-1]*ecur[i])/(ecur[i-1]-ecur[i])

    #print(f"  --> final T={Tem[ite-1]:.6f}")

    return Tem[ite-1]


def f_T(T, sig_jy_squared, rho_ss, eigvals, debye):
    target_ecur = 0
    return cal_ecur(sig_jy_squared, rho_ss, eigvals, T, debye) - target_ecur

'''
def cal_localT(j, rhoss, eigvals, U, Nb, debye, L, small_gamma = 1):
    sigma_jy_squared = sigma_jy_2(j, L, U, Nb)
    T_local = brentq(
        f_T,
        1e-6,        # Tmin
        500,         # Tmax (adjust)
        args=(sigma_jy_squared, rhoss, eigvals, debye)
    )
    return T_local
'''

# THIS IS FOR QUICK TESTING
def main_TEST():
    L = 4
    GAMMA = [0.1]
    J= -1.07
    μ = 1.3 
    Ωd = 4.0
    ω = np.pi / 0.8
    Nb = 10
    Nd = 1
    debye_omega = 4.0 # NOTE: This is equal to Ωd based on what they overleaf says (Previously --> 10.0)
    kappa = 0
    alpha = 1

    base_seed = 0  # NOTE: Sets base seed

    temp_fluctuation = []
    H1 = H_1(L, Nb)
    b = create_b(Nb, alpha=alpha, L=L)
    b_dagger = b.conj().T
    H_number = b_dagger_b(ω, b, b_dagger, L)
    H2 = H1 @ (b + b_dagger)

    for G in GAMMA:
        fluctuations = []
        C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
        H2_scaled = H2 * G
        for k in range(Nd):
            Temp_j = []
            H0 = H_0(J, μ, L, Nb, (base_seed + k))
            H = H0 + (H1 * C_H1) + H2_scaled + H_number

            eigvals, U = np.linalg.eigh(H)
            ss = rho_ss(U, b, H_number/ω, kappa, ω, L, Nb)
            for j in range(L):
                Temp_j.append(cal_localT(j, ss, eigvals, U, Nb, debye_omega, L))
            delta_T = np.std(Temp_j)
            mean_T = np.mean(Temp_j)
            fluctuations.append(delta_T/mean_T)
        temp_fluctuation.append(np.mean(fluctuations))

    print(f"Fluctuation: {temp_fluctuation}")


# THIS IS FOR THE CLUSTER
def main():

    GAMMA_ARRAY = np.linspace(0, 0.5, 20)
    L_ARRAY = [4, 5]
    L = L_ARRAY[int(int(sys.argv[1])/20)]
    GAMMA = [GAMMA_ARRAY[int(sys.argv[1]) % 20] * np.power(L, 1/6)] # NOTE: WE PUT SCALING HERE AS 1/6

    J= -1.07
    μ = 1.3 
    Ωd = 4.0
    ω = np.pi / 0.8
    Nb = 10
    Nd = 50
    debye_omega = 4.0 # NOTE: This is equal to Ωd based on what they overleaf says (Previously --> 10.0)
    kappa = 0
    alpha = 1

    base_seed = 0  # NOTE: Sets base seed

    temp_fluctuation = []
    mean_delta_T = []
    mean_Ts = []

    temp_fluc_std = []
    deltaT_std = []
    Ts_std = []

    H1 = H_1(L, Nb)
    b = create_b(Nb, alpha=alpha, L=L)
    b_dagger = b.conj().T
    H_number = b_dagger_b(ω, b, b_dagger, L)
    H2 = H1 @ (b + b_dagger)

    # NOTE: G here will be the SCALED gamma (NOT the unscaled version)
    for G in GAMMA:
        fluctuations = []
        deltaTs = []
        Ts = []

        C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
        H2_scaled = H2 * G
        for k in range(Nd):
            Temp_j = []
            H0 = H_0(J, μ, L, Nb, (base_seed + k))
            H = H0 + (H1 * C_H1) + H2_scaled + H_number

            eigvals, U = np.linalg.eigh(H)
            ss = rho_ss(U, b, H_number/ω, kappa, ω, L, Nb)
            for j in range(L):
                Temp_j.append(cal_localT(j, ss, eigvals, U, Nb, debye_omega, L))
            delta_T = np.std(Temp_j)
            mean_T = np.mean(Temp_j)
            fluctuations.append(delta_T/mean_T)
            deltaTs.append(delta_T)
            Ts.append(mean_T)
    
        temp_fluctuation.append(np.mean(fluctuations))
        mean_delta_T.append(np.mean(deltaTs))
        mean_Ts.append(np.mean(Ts))

        Ts_std.append(np.std(Ts))
        deltaT_std.append(np.std(deltaTs))
        temp_fluc_std.append(np.std(fluctuations))


        np.savez(
    f"temp_stat_scaled_gamma_{G:.3f}_Nd_{Nd}_Nb_{Nb}_base_seed_{base_seed}_L_{L}_Original_params.npz",
    
    # Parameters
    scaling = 1/6,
    J = J,
    mu = μ,
    Omega_d = Ωd,
    omega = ω,
    Nb = Nb,
    Nd = Nd,
    debye_omega = debye_omega,
    base_seed = base_seed,
    L = L,
    scaled_gamma = G,
    seed_base = base_seed,

    # Computed quantities (convert lists to arrays)
    mean_fluctuations = temp_fluctuation[0],
    mean_deltaT = mean_delta_T[0],
    mean_T = mean_Ts[0],
    std_fluctuations = temp_fluc_std[0],
    std_deltaT = deltaT_std[0],
    std_T = Ts_std[0]
)

main()


'''
Things to keep note of:

- I compared with Kokis code with
    - μ = 0.2
    - debye = 10
    - Hard Coded \mu_i values
    - Everything matched (Check Excel sheet)

ALSO, THE ORIGINAL PARAMS ARE:

    J= -1.07
    μ = 1.3 
    Ωd = 4.0
    ω = np.pi / 0.8
    Nb = 10
    Nd = 1
    debye_omega = 4.0
    
'''
