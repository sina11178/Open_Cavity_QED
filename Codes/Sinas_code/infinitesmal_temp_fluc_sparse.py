import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd

# For our model, we will consider (bosons) ⊗ (spin)

# Hamiltonian parts

def H_0(J, mu, L, Nb): # Nb --> Number of bosons
    dim = (2**L)
    H = sp.lil_matrix((dim, dim), dtype=np.complex128)
    s_dim = 2**L
    h_i = np.random.choice([-mu, mu], size=L)
    for s in range(dim):
        # Diagonal terms
        for i in range(L):
            j = (i + 1) % L
            H[s, s] += J * (1 if ((s >> i) & 1) == ((s >> j) & 1) else -1)
        for i in range(L):
            H[s, s] += h_i[i] * (1 if ((s >> i) & 1) == 0 else -1) # >> is a right ward shift (cutting off bits on the right)

    H = H.tocsr() # CSR --> More efficient for arithmetic
    H = sp.kron(sp.eye(Nb), H, format='csr')  # Extend to bosonic space
    return H

def H_1(L, Nb):
    dim = 2**L
    H = sp.lil_matrix((dim, dim), dtype=np.complex128)
    for s in range(dim):
        # Off-diagonal terms
        for i in range(L):
            s2 = (s) ^ (1 << i)  # << Flip the i-th spin; It is a left ward shift (adding zeros on the right)
            H[s2, s] += 1.0  # NOTE: To understand code, think of H[s2, s] as <s2|H|s> --> i.e. What you would expect this to give you
    H = H.tocsr()
    H = sp.kron(sp.eye(Nb), H, format='csr')  #
    return H

def create_b(Nb, alpha, L):
    """
    Construct truncated bosonic annihilation operator
    in Nb-dimensional Fock basis.
    """
    data = np.sqrt(np.arange(1, Nb))   # √1, √2, ..., √(Nb-1)
    b = sp.diags(data, offsets=1, format="csr")
    #b.setdiag(-1*alpha)  # Add the displacement term to the diagonal if code doesn't work
    b = sp.kron(b, sp.eye(2**L), format='csr')  # Extend to spin space
    return b

def b_dagger_b(omega, b, b_dagger, L):
    H = omega * b_dagger @ b
    H = H.tocsr()
    return H

# Transition matrix --> Rho SS


def transition_A(U, b, H_number, kappa, omega, L, Nb):
    dim_total = (2**L) * Nb
    A = sp.lil_matrix((dim_total, dim_total), dtype=np.complex128)
    for m in range(dim_total):
        U_m = U[:, m].conj().T  # m-th eigenvector of H
        for n in range(dim_total):
            U_n = U[:, n]  # n-th eigenvector of H
            if n == m:
                first_term = np.abs(U_m @ b @ U_n)**2
                second_term = U_m @ H_number @ U_n 
                A[m, n] = first_term - second_term
            else:
                first_term = np.abs(U_m @ b @ U_n)**2
                A[m, n] = first_term

    A = kappa * A
    A = A.tocsr()
    #eigenvalues, eigenvectors = spla.eigsh(A, k=1, sigma=1e-15, which='LM')  # May need to change sigma...
    eigenvalues, eigenvectors = spla.eigs(A, k=1, sigma=1e-15, which='LM')  # May need to change sigma...

    return eigenvalues, eigenvectors


# Gives c_m vector --> rho_ss = sum(c_m|m><m|)
def rho_ss(U, b, H_number, kappa, omega, L, Nb):
    eigenvalues, eigenvectors = transition_A(U, b, H_number, kappa, omega, L, Nb)
    if eigenvalues[0] > 1e-14:  # Check if the smallest eigenvalue is close to zero
        print("Warning: No zero eigenvalue found. Steady state may not be valid.")

    rho_ss = eigenvectors[:, 0]  # Steady state density matrix (unnormalized)
    rho_ss = rho_ss / np.sum(rho_ss)  # Normalize
    return np.real(rho_ss)


# sigma_y operator for j-th spin (only in spin space)
def sigma_yj(j, L, Nb):
    dim = 2**L
    sigma_y = sp.lil_matrix((dim, dim), dtype=np.complex128)
    for s in range(dim):
        s2 = s ^ (1 << j)  # Flip the j-th spin
        sigma_y[s2, s] += 1j * (1 if ((s >> j) & 1) == 0 else -1) # Double check this
    sigma_y = sp.kron(sp.eye(Nb), sigma_y, format='csr')  # Extend to bosonic space
    sigma_y = sigma_y.tocsr()
    #sigma_y = sp.kron(sp.eye(Nb), sigma_y, format='csr')
    return sigma_y

def sigma_jy_2(j, L, U, Nb):
    sigma_jy = sigma_yj(j, L, Nb)
    dim = (2**L) *Nb
    sig_y_squared = sp.lil_matrix((dim, dim), dtype=np.complex128)
    for m in range(dim):
        bra_m = U[:, m].conj().T  # m-th eigenvector of H
        for n in range(dim):
            ket_n = U[:, n]  # n-th eigenvector of H
            sig_y_squared[m, n] = np.abs(bra_m @ sigma_jy @ ket_n)**2
    sig_y_squared = sig_y_squared.tocsr()
    return sig_y_squared


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
                    print("Use approx")
                else:
                    n_B = 1.0 / (np.exp(dE / Tem) - 1)  # Bose-Einstein distribution
                ecur += (dE**2)* sig_jy_squared[m, n] * n_B *  rho_ss[n] / (dE**4 + debye**4)
    constant_factor = 4 * np.pi * small_gamma # Adjust this factor as needed --> I will absorb, as it's just a factor
    ecur = ecur 
    return ecur



#NOTE: MAY BE ERRONEOUS
def cal_slope(sig_jy_squared, rho_ss, eigvals, debye, small_gamma = 1):
    slope = 0.0
    for m in range(len(eigvals)):
        for n in range(len(eigvals)):
            dE = eigvals[m] - eigvals[n]
            if np.abs(dE)>1e-12:
#                dE = eigvals[m] - eigvals[n]
                slope += dE* sig_jy_squared[m, n] * rho_ss[n] / (dE**4 + debye**4)
    constant_factor = 4 * np.pi * small_gamma # Adjust this factor as needed --> I will absorb, as it's just a factor
    slope = slope
    return slope    


# NOTE: You get imaginary and negative temperatures at a certain point
def cal_localT(j, rhoss, eigvals, U, Nb, debye, L, small_gamma = 1):
    sigma_jy_squared = sigma_jy_2(j, L, U, Nb)

    ite = 15
    Tem = np.zeros(ite, dtype=np.complex128)
    ecur = np.zeros(ite, dtype=np.complex128) # Double check dtype
    slope = cal_slope(sigma_jy_squared, rhoss, eigvals, debye, small_gamma)

    Tem[0] = 500
    ecur[0] = cal_ecur(sigma_jy_squared, rhoss, eigvals, Tem[0], debye, small_gamma)
    Tem[1] = Tem[0] - ecur[0]/slope

    for i in range(1, ite-1):
        ecur[i] = cal_ecur(sigma_jy_squared, rhoss, eigvals, Tem[i], debye, small_gamma)
        if np.abs(ecur[i] - ecur[i-1]) < 1e-12 and np.abs(ecur[i]) < 1e-12:  # Check for convergence
            print(f"Converged at iteration {i} with local temperature {Tem[i]:.6f}")
            Tem[ite-1] = Tem[i]
            break
        Tem[i+1] = (ecur[i-1]*Tem[i] - Tem[i-1]*ecur[i])/(ecur[i-1]-ecur[i])

    return Tem[ite-1]




def main():
    GAMMA = np.linspace(0, 1, 5)
    J= -1.07
    μ = 1.3 
    Ωd = 4.0
    ω = np.pi / 0.8
    kappa = 0.5
    L = 2
    Nb = 10
    Nd = 100
    debye_omega = 10.0
    kappa = 0.00000001
    alpha = 1
    #All_H = []
    temp_fluctuation = []

    for G in GAMMA:
        Temp_j = []
        fluctuations = []
        C_H1 = (-8*Ωd * ω * G)/ (kappa**2 + 4*ω**2) # prefactor for H1
        H1 = H_1(L, Nb)
        b = create_b(Nb, alpha=alpha, L=L)
        b_dagger = b.getH()
        H_number = b_dagger_b(ω, b, b_dagger, L)
        H2 = H1 @ (b + b_dagger) * G
        for k in range(Nd):
            H0 = H_0(J, μ, L, Nb)
            H = H0 + (H1 * C_H1) + H2 + H_number

            eigvals, U = np.linalg.eigh(H.toarray())
            ss = rho_ss(U, b, H_number/ω, kappa, ω, L, Nb)
            for j in range(L):
                Temp_j.append(cal_localT(j, ss, eigvals, U, Nb, debye_omega, L, small_gamma = 1))
            delta_T = np.std(Temp_j)
            mean_T = np.mean(Temp_j)
            fluctuations.append(delta_T/mean_T)
        temp_fluctuation.append(np.mean(fluctuations))

    plt.plot(GAMMA*np.sqrt(L), temp_fluctuation)
    plt.show()

main()