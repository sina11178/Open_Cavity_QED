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
def H_0(J, mu, L, Nb, Nd): # Nb --> Number of bosons
    dim = (2**L)
    H = np.zeros((dim, dim), dtype=np.complex128)
    s_dim = 2**L
    #h_i = 2*mu * np.random.rand(L) - mu

    seed = Nd
    #seed = np.random.randint(0, Nd+1)
    rng = np.random.Generator(np.random.MT19937(seed))
    h_i = 2*mu * rng.random(L) - mu
    for s in range(dim):
        # Diagonal terms
        for i in range(L):
            j = (i + 1) % L
            H[s, s] += J * (1 if ((s >> i) & 1) == ((s >> j) & 1) else -1)
#        for i in range(L):
            H[s, s] += h_i[i] * (-1 if ((s >> i) & 1) == 0 else 1) # >> is a right ward shift (cutting off bits on the right)

    H = np.kron(np.eye(Nb), H)  # Extend to bosonic space
    return H
'''

def H_0(J, mu, L, Nb, Nd): # Nb --> Number of bosons
    dim = 2**L
    s = np.arange(dim)
    H_diag = np.zeros(dim)

    rng = np.random.Generator(np.random.MT19937(Nd))
    h_i = 2 * mu * rng.random(L) - mu

    for i in range(L):
        j = (i + 1) % L
        same = ((s >> i) & 1) == ((s >> j) & 1)
        H_diag += np.where(same, J, -J)
        spin_sign = np.where((s >> i) & 1, 1, -1)
        H_diag += h_i[i] * spin_sign

    H = np.diag(H_diag.astype(np.complex128))
    return np.kron(np.eye(Nb), H)
'''
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
'''

def H_1(L, Nb):
    dim = 2**L
    H = np.zeros((dim, dim), dtype=np.complex128)
    s = np.arange(dim)
    for i in range(L):
        s2 = s ^ (1 << i)
        H[s2, s] += 1.0
    return np.kron(np.eye(Nb), H)


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

def magnetization_operator(L, Nb):
    dim = 2**L
    sig_z_mag = np.zeros((dim, dim), dtype=np.complex128)
    for s in range(dim):
        for j in range(L):
            sig_z_mag[s, s] += (-1 if ((s >> j) & 1) == 0 else 1) # >> is a right ward shift (cutting off bits on the right)
    sig_z_mag = np.kron(np.eye(Nb), sig_z_mag)  # Extend to bosonic space

    return sig_z_mag


def magnetization_fourier_mode(L, Nb):
    dim = 2**L
    sig_z_mag = np.zeros((dim, dim), dtype=np.complex128)
    for s in range(dim):
        for j in range(L):
            sig_z_mag[s, s] += (-1 if ((s >> j) & 1) == 0 else 1) * np.exp(2j * np.pi * j / L) # >> is a right ward shift (cutting off bits on the right)
    sig_z_mag = np.kron(np.eye(Nb), sig_z_mag)  # Extend to bosonic space

    return sig_z_mag

'''
def two_point_correlation(L, Nb):
    dim = 2**L
    sig_z_corr = np.zeros((dim, dim), dtype=np.complex128)
    for s in range(dim):
        for j in range(L):
            for k in range(L):
                sig_z_corr[s, s] += (-1 if ((s >> j) & 1) == 0 else 1) * (-1 if ((s >> k) & 1) == 0 else 1) # >> is a right ward shift (cutting off bits on the right)
    sig_z_corr = np.kron(np.eye(Nb), sig_z_corr)  # Extend to bosonic space

    return sig_z_corr
'''
def two_point_correlation(L, Nb): # This is equivalent to sum_jk <sigma_zj sigma_zk> = <(sum_j sigma_zj)^2>
    s = np.arange(2**L)
    # Get all spin signs: shape (2**L, L)
    spin_signs = np.where(((s[:, None] >> np.arange(L)) & 1), 1, -1)
    # Sum over all j,k pairs: equivalent to (sum_j sign_j)^2
    diag = (spin_signs.sum(axis=1)) ** 2
    sig_z_corr = np.diag(diag.astype(np.complex128))
    return np.kron(np.eye(Nb), sig_z_corr)


def cal_magnetization(rhoss, U, magnetization_op):
    rho_ss_mat = np.diag(rhoss)
    mat = U.conj().T @ magnetization_op @ U
    mat = mat @ rho_ss_mat
    magnetization_ss = np.trace(mat)

    return magnetization_ss


# RETURNS EQ 6 of HUSE PAPER
def cal_magnetization_fourier_mode(rhoss, U, magnetization_op):
    rho_ss_mat = np.diag(rhoss)
    mat = U.conj().T @ magnetization_op @ U
    mat = mat @ rho_ss_mat
    mat = (mat * mat.conj()).real # Take the magnitude squared of the Fourier mode magnetization
    magnetization_ss = np.trace(mat)

    mat_norm = magnetization_op.conj().T @ magnetization_op
    mat_norm = U.conj().T @ mat_norm @ U
    mat_norm = mat_norm @ rho_ss_mat
    magnetization_norm = np.trace(mat_norm)

    f  = 1 - magnetization_ss / magnetization_norm

    return f

def cal_two_point_correlation(rhoss, U, corr_op):
    rho_ss_mat = np.diag(rhoss)
    mat = U.conj().T @ corr_op @ U
    mat = mat @ rho_ss_mat
    corr_ss = np.trace(mat)

    return corr_ss


def single_disorder(k, base_seed, J, μ, l, Nb, H1, C_H1, H2_scaled, H_number, H_number_norm, b, kappa, ω, magn_operator, type):
    H0 = H_0(J, μ, l, Nb, k + base_seed)
    H = H0 + (H1 * C_H1) + H2_scaled + H_number
    _, U = np.linalg.eigh(H)
    ss = rho_ss(U, b, H_number_norm, kappa, ω, l, Nb)
    if type == "N":
        return cal_magnetization(ss, U, magn_operator).real
    elif type == "FM":
        return cal_magnetization_fourier_mode(ss, U, magn_operator).real
    elif type == "2p":
        return cal_two_point_correlation(ss, U, magn_operator).real
    else:
        raise ValueError("Invalid type specified. Choose 'N', 'FM', or '2p'.")



def main():
    base_seed = 0
    GAMMA = np.linspace(0.001, .85, 35)  # NOTE: IF USING SPLA, DONT USE GAMMA TOO CLOSE TO 0
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
    type = "2p" # "N" for normal magnetization graphing, "FM" for Fourier mode magnetization graphing, "2p" for two point correlation graphing

    if type == "N":  # Normal magnetization graphing
        for l in L:
            magn_operator = magnetization_operator(l, Nb)
            spin_magnetization = []
            H1 = H_1(l, Nb)
            b = create_b(Nb, alpha=alpha, L=l)
            b_dagger = b.conj().T
            H_number = b_dagger_b(ω, b, b_dagger, l)
            H2 = H1 @ (b + b_dagger)
            for G in GAMMA:
                #spin_j = []
                G = G * np.power(l, 1/2)
                magnetizations = []
                C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
                H2_scaled = H2 * G
                for k in range(Nd):
                    H0 = H_0(J, μ, l, Nb, k + base_seed)
                    H = H0 + (H1 * C_H1) + H2_scaled + H_number
                    eigvals, U = np.linalg.eigh(H)
                    ss = rho_ss(U, b, H_number/ω, kappa, ω, l, Nb)
                    magnetization = cal_magnetization(ss, U, magn_operator)

                    magnetizations.append(magnetization)
                    print("Iteration "+ str(k) + " for L "+ str(l))
                spin_magnetization.append(np.mean(magnetizations))
            plt.plot(GAMMA * np.power(l, 1/2), np.abs(spin_magnetization), label = l)
            #plt.plot(GAMMA, spin_fluctuation, label = l)
            plt.xlabel("Gamma * √L")
            #plt.xscale("log")
            #plt.yscale("log")
            plt.ylabel("|<M>|")
            plt.title("Spin Magnetization - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , base_seed = " + str(base_seed) + " (J, μ, Ωd, ω, Nb) = " + str((J, μ, Ωd, ω, Nb)))
            plt.legend()

        plt.show()
        #print(f"Fluctuation: {spin_fluctuation}")
    
    elif type == "FM":  # Fourier mode magnetization graphing
        for l in L:
            magn_operator = magnetization_fourier_mode(l, Nb)
            spin_magnetization = []
            H1 = H_1(l, Nb)
            b = create_b(Nb, alpha=alpha, L=l)
            b_dagger = b.conj().T
            H_number = b_dagger_b(ω, b, b_dagger, l)
            H2 = H1 @ (b + b_dagger)
            for G in GAMMA:
                #spin_j = []
                G = G * np.power(l, 1/2)
                magnetizations = []
                C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
                H2_scaled = H2 * G
                for k in range(Nd):
                    H0 = H_0(J, μ, l, Nb, k + base_seed)
                    H = H0 + (H1 * C_H1) + H2_scaled + H_number
                    eigvals, U = np.linalg.eigh(H)
                    ss = rho_ss(U, b, H_number/ω, kappa, ω, l, Nb)
                    magnetization = cal_magnetization_fourier_mode(ss, U, magn_operator)

                    magnetizations.append(magnetization)
                    print("Iteration "+ str(k) + " for L "+ str(l))
                spin_magnetization.append(np.mean(magnetizations))
            plt.plot(GAMMA * np.power(l, 1/2), np.abs(spin_magnetization), label = l)
            #plt.plot(GAMMA, spin_fluctuation, label = l)
            plt.xlabel("Gamma * √L")
            #plt.xscale("log")
            #plt.yscale("log")
            plt.ylabel("|<f>|")
            plt.title("Spin Magnetization - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , base_seed = " + str(base_seed) + " (J, μ, Ωd, ω, Nb) = " + str((J, μ, Ωd, ω, Nb)))
            plt.legend()

        plt.show()

    elif type == "2p":  # Two point correlation graphing
        for l in L:
            magn_operator = two_point_correlation(l, Nb)
            spin_magnetization = []
            H1 = H_1(l, Nb)
            b = create_b(Nb, alpha=alpha, L=l)
            b_dagger = b.conj().T
            H_number = b_dagger_b(ω, b, b_dagger, l)
            H2 = H1 @ (b + b_dagger)
            for G in GAMMA:
                #spin_j = []
                G = G * np.power(l, 1/2)
                magnetizations = []
                C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
                H2_scaled = H2 * G
                for k in range(Nd):
                    H0 = H_0(J, μ, l, Nb, k + base_seed)
                    H = H0 + (H1 * C_H1) + H2_scaled + H_number
                    eigvals, U = np.linalg.eigh(H)
                    ss = rho_ss(U, b, H_number/ω, kappa, ω, l, Nb)
                    magnetization = cal_two_point_correlation(ss, U, magn_operator)

                    magnetizations.append(magnetization)
                    print("Iteration "+ str(k) + " for L "+ str(l))
                spin_magnetization.append(np.mean(magnetizations))
            plt.plot(GAMMA * np.power(l, 1/2), np.abs(spin_magnetization), label = l)
            #plt.plot(GAMMA, spin_fluctuation, label = l)
            plt.xlabel("Gamma * √L")
            #plt.xscale("log")
            #plt.yscale("log")
            plt.ylabel("|<M_2p>|")
            plt.title("2-point correlation - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , base_seed = " + str(base_seed)+ " (J, μ, Ωd, ω, Nb) = " + str((J, μ, Ωd, ω, Nb)))
            plt.legend()

        plt.show()
    else:
        raise ValueError("Invalid type specified. Choose 'N', 'FM', or '2p'.")


# NOTE: Faster but uses more memory --> Parallelize over disorder realizations (Nd) for each gamma and L; use for laptop,but not for cluster
def main_parallelize():
    base_seed = 0
    GAMMA = np.linspace(0.001, 0.5, 35)  # NOTE: IF USING SPLA, DONT USE GAMMA TOO CLOSE TO 0
    #GAMMA = [0]
    J= -1.07
    μ = 1.2 
    Ωd = 4.0
    ω = np.pi / 0.8
    L = [2, 3, 4, 5, 6]
    Nb = 10
    Nd = 10
    debye_omega = 4.0
    kappa = 0
    alpha = 1
    type = "FM" # "N" for normal magnetization graphing, "FM" for Fourier mode magnetization graphing, "2p" for two point correlation graphing

    if type == "N":  # Normal magnetization graphing
        for l in L:
            magn_operator = magnetization_operator(l, Nb)
            spin_magnetization = []
            H1 = H_1(l, Nb)
            b = create_b(Nb, alpha=alpha, L=l)
            b_dagger = b.conj().T
            H_number = b_dagger_b(ω, b, b_dagger, l)
            H2 = H1 @ (b + b_dagger)
            for G in GAMMA:
                #spin_j = []
                G = G * np.power(l, 1/2)
                magnetizations = []
                C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
                H2_scaled = H2 * G
                H_number_norm = H_number / ω
                magnetizations = Parallel(n_jobs=-1)(
                    delayed(single_disorder)(k, base_seed, J, μ, l, Nb, H1, C_H1, 
                                            H2_scaled, H_number, H_number_norm, b, kappa, ω, magn_operator, type)
                    for k in range(Nd)
                )
                spin_magnetization.append(np.mean(magnetizations))
            plt.plot(GAMMA * np.power(l, 1/2), np.abs(spin_magnetization), label = l)
            #plt.plot(GAMMA, spin_fluctuation, label = l)
            plt.xlabel("Gamma * √L")
            #plt.xscale("log")
            #plt.yscale("log")
            plt.ylabel("|<M>|")
            plt.title("Spin Magnetization - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , base_seed = " + str(base_seed)+ " (J, μ, Ωd, ω, Nb) = " + str((J, μ, Ωd, ω, Nb)))
            plt.legend()

        plt.show()
        #print(f"Fluctuation: {spin_fluctuation}")
    
    elif type == "FM":  # Fourier mode magnetization graphing
        for l in L:
            magn_operator = magnetization_fourier_mode(l, Nb)
            spin_magnetization = []
            H1 = H_1(l, Nb)
            b = create_b(Nb, alpha=alpha, L=l)
            b_dagger = b.conj().T
            H_number = b_dagger_b(ω, b, b_dagger, l)
            H2 = H1 @ (b + b_dagger)
            for G in GAMMA:
                #spin_j = []
                G = G * np.power(l, 1/2)
                magnetizations = []
                C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
                H2_scaled = H2 * G
                H_number_norm = H_number / ω
                magnetizations = Parallel(n_jobs=-1)(
                    delayed(single_disorder)(k, base_seed, J, μ, l, Nb, H1, C_H1, 
                                            H2_scaled, H_number, H_number_norm, b, kappa, ω, magn_operator, type)
                    for k in range(Nd)
                )
                spin_magnetization.append(np.mean(magnetizations))
            
            print("Finished all disorder realizations for L = " + str(l))
            plt.plot(GAMMA * np.power(l, 1/2), np.abs(spin_magnetization), label = l)
            #plt.plot(GAMMA, spin_fluctuation, label = l)
            plt.xlabel("Gamma * √L")
            #plt.xscale("log")
            #plt.yscale("log")
            plt.ylabel("|<f>|")
            plt.title("Spin Magnetization - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , base_seed = " + str(base_seed)+ " (J, μ, Ωd, ω, Nb) = " + str((J, μ, Ωd, ω, Nb)))
            plt.legend()

        plt.show()

    elif type == "2p":  # Two point correlation graphing
        for l in L:
            magn_operator = two_point_correlation(l, Nb)
            spin_magnetization = []
            H1 = H_1(l, Nb)
            b = create_b(Nb, alpha=alpha, L=l)
            b_dagger = b.conj().T
            H_number = b_dagger_b(ω, b, b_dagger, l)
            H2 = H1 @ (b + b_dagger)
            for G in GAMMA:
                #spin_j = []
                G = G * np.power(l, 1/2)
                magnetizations = []
                C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
                H2_scaled = H2 * G
                H_number_norm = H_number / ω
                magnetizations = Parallel(n_jobs=-1)(
                    delayed(single_disorder)(k, base_seed, J, μ, l, Nb, H1, C_H1, 
                                            H2_scaled, H_number, H_number_norm, b, kappa, ω, magn_operator, type)
                    for k in range(Nd)
                )
                spin_magnetization.append(np.mean(magnetizations))
            print("Finished all disorder realizations for L = " + str(l))
            plt.plot(GAMMA * np.power(l, 1/2), np.abs(spin_magnetization)/np.sum(spin_magnetization), label = l)
            #plt.plot(GAMMA, spin_fluctuation, label = l)
            plt.xlabel("Gamma * √L")
            #plt.xscale("log")
            #plt.yscale("log")
            plt.ylabel("|<M_2p>|")
            plt.title("2-point correlation - Nd = " + str(Nd) + ", Nb = " + str(Nb) + " , base_seed = " + str(base_seed)+ " (J, μ, Ωd, ω, Nb) = " + str((J, μ, Ωd, ω, Nb)))
            plt.legend()

        plt.show()
    else:
        raise ValueError("Invalid type specified. Choose 'N', 'FM', or '2p'.")





main_parallelize()