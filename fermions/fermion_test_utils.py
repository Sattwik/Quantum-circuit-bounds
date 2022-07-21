import abc
from typing import List, Tuple, Callable, Dict
from functools import partial
import time

import numpy as np
import scipy
import networkx as nx
import qutip
import tensornetwork as tn
tn.set_default_backend("jax")
import jax.numpy as jnp
import jax.scipy.linalg
from jax import jit, grad, vmap, value_and_grad
from jax.example_libraries import optimizers

from fermions import gaussian

# anticommutator
def anticommutator(a, b):
    return a * b + b * a

def commutator(a, b):
    return a * b - b * a

# lists of operators
def Is(i): return [qutip.qeye(2) for j in range(0, i)]
def I(N): return qutip.tensor(Is(N))
def Sx(N, i): return qutip.tensor(Is(i) + [qutip.sigmax()] + Is(N - i - 1))
def Sy(N, i): return qutip.tensor(Is(i) + [qutip.sigmay()] + Is(N - i - 1))
def Sz(N, i): return qutip.tensor(Is(i) + [qutip.sigmaz()] + Is(N - i - 1))

# sum, product, and power of lists of operators
def osum(lst): return np.sum(np.array(lst, dtype=object))

def oprd(lst, d=None):
    if len(lst) == 0:
        return d
    p = lst[0]
    for U in lst[1:]:
        p = p*U
    return p

def opow(op, N): return oprd([op for i in range(N)])

def a(N, n, Opers=None):
    Sa, Sb, Sc = Sz, Sx, Sy
    if Opers is not None:
        Sa, Sb, Sc = Opers
    return oprd([Sa(N, j) for j in range(n)], d = I(N)) * (Sb(N, n) + 1j * Sc(N, n))/2.0

def ad(N, n, Opers=None):
    Sa, Sb, Sc = Sz, Sx, Sy
    if Opers is not None:
        Sa, Sb, Sc = Opers
    return oprd([Sa(N, j) for j in range(n)], d = I(N)) * (Sb(N, n) - 1j * Sc(N, n))/2.0

def x(N, n):
    return np.sqrt(1/2.0) * (a(N,n) + ad(N,n))

def p(N, n):
    return 1j * np.sqrt(1/2.0) * (ad(N,n) - a(N,n))

def list_as(N):
    return [a(N, n) for n in range(0, N)]

def list_ads(N):
    return [ad(N, n) for n in range(0, N)]

def Omega(N):
    return np.sqrt(1/2) * np.block([[np.eye(N), np.eye(N)],[1j * np.eye(N), -1j * np.eye(N)]])

def diagonal_2x2_dm(p):
    P0 = qutip.basis(2,0) * qutip.basis(2,0).dag()
    P1 = qutip.basis(2,1) * qutip.basis(2,1).dag()

    return (1-p) * P0 + (p) * P1

def rotate_cr_ann_ops(U, N):
    alpha = list_ads(N) + list_as(N)

    beta = [0] * 2 * N
    for i in range(2 * N):
        for j in range(2 * N):
            beta[i] += U[i, j] * alpha[j]

    return beta

def rotate_majorana_ops(O, N):
    x_list = [x(N, n) for n in range(0, N)]
    p_list = [p(N, n) for n in range(0, N)]
    r_list = x_list + p_list

    s_list = [0.0 + 0.0j] * 2 * N

    for i in range(2 * N):
        for j in range(2 * N):
            s_list[i] += O[i, j] * r_list[j]

    return s_list

#--------------- Gates and operations ---------------#
def fswap_gate(idx: int, N: int) -> qutip.Qobj:
    """
    Parameters:
    ----------
    idx: index of mode to be swapped with idx + 1.
    """
    template_qobj = qutip.tensor([qutip.qeye(2)] * 2)
    swap_gate_array = np.array([[1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, -1]], dtype = complex)

    swap_gate_qobj = qutip.Qobj(swap_gate_array,
                        dims = template_qobj.dims, shape = template_qobj.shape)

    swap_gate = qutip.tensor([qutip.qeye(2)] * (idx) + [swap_gate_qobj] +
                             [qutip.qeye(2)] * (N - idx - 2))

    return swap_gate

def ptrace_n_replace(rho: qutip.Qobj, idx: int, N: int) -> qutip.Qobj:
    """
    Traces out the mode at index |idx| and replaces with identity/2.
    """
    # move mode to the very end (right)
    for j in range(idx, N - 1):
        # print('here')
        fswap = fswap_gate(j, N)
        rho = fswap * rho * fswap.dag()

    # trace out and replace
    a_end = list_as(N)[N - 1]
    ad_end = list_ads(N)[N - 1]
    zero_end = qutip.tensor([qutip.qeye(2)] * (N - 1) + [qutip.basis(2,0)])

    tr_idx_rho = zero_end.dag() * rho * zero_end + \
                 zero_end.dag() * a_end * rho * ad_end * zero_end

    if N == 1:
        tr_idx_rho = tr_idx_rho * qutip.qeye(2)/2.0
    else:
        tr_idx_rho = qutip.tensor([tr_idx_rho, qutip.qeye(2)/2.0])

    # swap back
    for j in range(N - 2, idx - 1, -1):
        fswap = fswap_gate(j, N)
        tr_idx_rho = fswap * tr_idx_rho * fswap.dag()

    if np.isclose(tr_idx_rho.tr(), 0.0, atol = 1e-9) or np.isclose(rho.tr(), 0.0, atol = 1e-9):
        return tr_idx_rho
    else:
        return (tr_idx_rho/tr_idx_rho.tr()) * rho.tr()

def noise_on_full_op(rho: qutip.Qobj, prob: float, N: int) -> qutip.Qobj:
    rho_noisy = rho

    for i in range(N):
        rho_noisy = prob * ptrace_n_replace(rho_noisy, i, N) + (1 - prob) * rho_noisy

    return rho_noisy

def full_op_from_majorana(h: np.array, rotate = None):
    if rotate is None:
        N = h.shape[0]//2

        x_list = [x(N, n) for n in range(0, N)]
        p_list = [p(N, n) for n in range(0, N)]
        r_list = x_list + p_list

        full_op = 0
        for i in range(0, 2 * N):
            for j in range(0, 2 * N):
                full_op += 1j * h[i,j] * r_list[i] * r_list[j]

        return full_op

    else:
        N = h.shape[0]//2
        V = rotate
        alpha = list_ads(N) + list_as(N)
        H_beta = 1j * V.conj().T @ h @ V

        full_op = 0
        for i in range(0, 2 * N):
            for j in range(0, 2 * N):
                full_op += H_beta[i,j] * alpha[i].dag() * alpha[j]
                # NB using alpha here as H_beta is already rotated

        return full_op

#--------------- Tests ---------------#
def test_noise_on_hamiltonian(s: np.array, p: float, N: int):
    # full
    s_full = full_op_from_majorana(s)
    s_noisy_full = noise_on_full_op(s_full, p, N)

    # rep
    s_prime = gaussian.noise_on_fghamiltonian(jnp.array(s), p)
    s_prime_full = full_op_from_majorana(np.array(s_prime))

    return np.linalg.norm(s_noisy_full.full() - s_prime_full.full())

def test_unitary_on_hamiltonian(s: np.array, h: np.array):
    s_full = full_op_from_majorana(s)
    h_full = full_op_from_majorana(h)

    U_full = (-1j * h_full).expm()

    s_prime_full_other = U_full * s_full * U_full.dag()

    s_prime = np.array(gaussian.unitary_on_fghamiltonian(jnp.array(s), jnp.array(h)))
    s_prime_full = full_op_from_majorana(s_prime)

    return np.linalg.norm(s_prime_full.full() - s_prime_full_other.full())

def test_energy(Gamma_mjr: np.array, f: np.array, V: np.array, h: np.array):
    e_majorana = gaussian.energy(jnp.array(Gamma_mjr), jnp.array(h))

    N = len(f)
    rho_beta = qutip.tensor([diagonal_2x2_dm(f[n]) for n in range(N)])
    h_full = full_op_from_majorana(h, rotate = V)

    e_full = (rho_beta * h_full).tr()

    return e_full - e_majorana, e_full, e_majorana

def test_trace_fgs(parent_h: np.array, N: int):
    s_full = full_op_from_majorana(parent_h)

    trace_full = s_full.expm().tr()

    trace_majorana = gaussian.trace_fgstate(jnp.array(parent_h))

    return trace_full - trace_majorana, trace_full, trace_majorana

def test_covariance_def(Gamma_mjr: np.array, f:np.array, O: np.array, N: int):
    gamma = gaussian.covariance_from_corr_major(jnp.array(Gamma_mjr))
    gamma = O @ np.array(gamma) @ O.T

    gamma_prime = np.zeros((2*N, 2*N), dtype = complex)
    rho_beta = qutip.tensor([diagonal_2x2_dm(f[n]) for n in range(N)])
    x_list = [x(N, n) for n in range(0, N)]
    p_list = [p(N, n) for n in range(0, N)]
    r_list = x_list + p_list

    for i in range(2*N):
        for j in range(2*N):
            gamma_prime[i,j] = 1j * (rho_beta * commutator(r_list[i], r_list[j])).tr()

    return np.linalg.norm(np.array(gamma) - gamma_prime)

def test_unitary_on_fgstate(Gamma_mjr: np.array, f: np.array, O: np.array, V: np.array, h: np.array):
    N = len(f)
    rho_beta = qutip.tensor([diagonal_2x2_dm(f[n]) for n in range(N)])
    h_full = full_op_from_majorana(h, rotate = V)
    Ut_full = (-1j * h_full).expm()
    rhot = Ut_full * rho_beta * Ut_full.dag()

    Gamma_mjr_prime_full = np.zeros((2*N, 2*N), dtype = complex)
    x_list = [x(N, n) for n in range(0, N)]
    p_list = [p(N, n) for n in range(0, N)]
    r_list = x_list + p_list

    for i in range(2*N):
        for j in range(2*N):
            Gamma_mjr_prime_full[i,j] = (rhot * r_list[i] * r_list[j]).tr()

    Gamma_mjr_prime = gaussian.unitary_on_fgstate(jnp.array(Gamma_mjr), jnp.array(h))
    Gamma_mjr_prime = O @ np.array(Gamma_mjr_prime) @ O.T

    return np.linalg.norm(Gamma_mjr_prime_full - Gamma_mjr_prime)

def test_corr_major_from_parenth(parent_h: np.array, N: int):
    Gamma_mjr = np.array(gaussian.corr_major_from_parenth(jnp.array(parent_h)))

    Gamma_mjr_prime = np.zeros((2*N, 2*N), dtype = complex)
    x_list = [x(N, n) for n in range(0, N)]
    p_list = [p(N, n) for n in range(0, N)]
    r_list = x_list + p_list

    rho = (-full_op_from_majorana(parent_h)).expm()
    rho = rho/rho.tr()

    for i in range(2*N):
        for j in range(2*N):
            Gamma_mjr_prime[i,j] = (rho * r_list[i] * r_list[j]).tr()

    return np.linalg.norm(Gamma_mjr - Gamma_mjr_prime)

def test_noise_on_fgstate(parent_h: np.array, test_h: np.array, N: int, p: float, key: jnp.array):

    rho = (-full_op_from_majorana(parent_h)).expm()
    rho = rho/rho.tr()

    rho_noisy = noise_on_full_op(rho, p, N)
    test_exp = (rho_noisy * full_op_from_majorana(test_h)).tr()

    Gamma_mjr = gaussian.corr_major_from_parenth(jnp.array(parent_h))

    num_mc_samples = 100
    test_exp_gaussian = 0
    for mc_num in range(int(num_mc_samples)):
        Gamma_mjr_noisy, key = \
                    gaussian.noise_on_fgstate_mc_sample(Gamma_mjr, p, N, key)

        test_exp_gaussian += gaussian.energy(Gamma_mjr_noisy, test_h)/num_mc_samples

    return np.abs(test_exp - test_exp_gaussian), test_exp, test_exp_gaussian

def primal_clean_circuit_full(circ_params: gaussian.PrimalParams):

    N = circ_params.N
    d = circ_params.d

    psi_init = qutip.tensor([qutip.basis(2,0)] * N)

    psi = psi_init
    for i in range(d):
        H_layer_full = full_op_from_majorana(np.array(circ_params.layer_hamiltonians[i]))
        U_layer_full = (-1j * H_layer_full).expm()
        psi = U_layer_full * psi

    H_parent_full = full_op_from_majorana(np.array(circ_params.h_parent))

    return psi, H_parent_full * psi, qutip.expect(H_parent_full, psi)

def primal_noisy_circuit_full(dual_params: gaussian.DualParams):

    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    p = dual_params.p

    rho_init = qutip.tensor([diagonal_2x2_dm(0) for n in range(N)])

    rho = rho_init
    for i in range(d):
        H_layer_full = full_op_from_majorana(np.array(dual_params.circ_params.layer_hamiltonians[i]))
        U_layer_full = (-1j * H_layer_full).expm()
        rho = U_layer_full * rho * U_layer_full.dag()

        rho = noise_on_full_op(rho, p, N)

    H_parent_full = full_op_from_majorana(np.array(dual_params.circ_params.h_parent))

    return (H_parent_full * rho).tr()

def full_noisy_dual_layer(h_layer: np.array, sigma_layer: np.array, p: float):
    N = h_layer.shape[0]//2
    sigma_full = full_op_from_majorana(sigma_layer)
    sigma_full = noise_on_full_op(sigma_full, p, N)

    U_layer = (-1j * full_op_from_majorana(-h_layer)).expm()
    sigma_full = U_layer * sigma_full * U_layer.dag()

    return sigma_full

def dual_full(dual_vars: jnp.array, dual_params: gaussian.DualParams):
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_parent = dual_params.circ_params.h_parent
    layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
    p = dual_params.p
    rho_init = qutip.tensor([diagonal_2x2_dm(0) for n in range(N)])

    lambdas, sigmas = gaussian.unvec_and_process_dual_vars(dual_vars, dual_params)
    lambdas = np.array(lambdas)
    sigmas = np.array(sigmas)

    cost = 0
    # log Tr exp terms
    for i in range(d):
        if i == d-1:
            hi = np.array(h_parent) + sigmas[:,:,i]

            # print('Parent Hamiltonian = ', h_parent)
            # print('Sigma[' + str(i) + ']= ', sigmas[:,:,i])

            hi = full_op_from_majorana(hi)

            # print('Full hd = ', hi)
        else:
            # print('Sigma[' + str(i) + ']= ', sigmas[:,:,i])
            # print('Sigma[' + str(i + 1) + ']= ', sigmas[:,:,i + 1])
            # print('Layer H[' + str(i + 1) + ']= ', np.array(layer_hamiltonians[i+1]))

            hi = full_op_from_majorana(sigmas[:,:,i]) - \
                 full_noisy_dual_layer(np.array(layer_hamiltonians[i+1]),
                                  sigmas[:,:,i+1], p)

            # print('Epsilondagsigma[' + str(i) + '] = ', full_noisy_dual_layer(np.array(layer_hamiltonians[i+1]), sigmas[:,:,i+1], p))
            # print('Full h[' + str(i) + '] = ', hi)

        cost += -lambdas[i] * np.log((-hi/lambdas[i]).expm().tr())

    # init. state term
    epsilon_1_dag_sigma1 = \
    full_noisy_dual_layer(np.array(layer_hamiltonians[0]), sigmas[:,:,0], p)

    cost += -(rho_init * epsilon_1_dag_sigma1).tr()

    # entropy term
    q = 1 - p
    q_powers = np.array([q**i for i in range(d)])

    entropy_bounds = N * p * np.log(2) * \
             np.array([np.sum(q_powers[:i+1]) for i in range(d)])

    cost += np.dot(lambdas, entropy_bounds)

    return -np.real(cost)

def test_circuit_parent_hamiltonian(N: int, d: int, local_d: int, key: jnp.array):

    circ_params = gaussian.PrimalParams(N, d, local_d, key)
    key, subkey = jax.random.split(circ_params.key_after_ham_gen)

    e_circ = gaussian.energy_after_circuit(circ_params)

    psi, h_times_psi, e_psi = primal_clean_circuit_full(circ_params)

    return key, np.abs(e_circ - e_psi), (e_psi * psi - h_times_psi).norm()

# def test_circuit_parent_hamiltonian_local(N: int, d: int, local_d: int, key: jnp.array):
#
#     circ_params = gaussian.PrimalParams(N, d, local_d, key)
#     key, subkey = jax.random.split(circ_params.key_after_ham_gen)
#
#     e_circ = gaussian.energy_after_circuit(circ_params)
#     e_circ_local = gaussian.energy_after_circuit_local(circ_params)
#
#     return key, np.abs(e_circ - e_circ_local), jnp.linalg.norm(circ_params.h_parent - circ_params.h_parent_local)


# def full_state_from_corr(Gamma_mjr: np.array):
#
#     Gamma = Omega(N).conj().T @ Gamma_mjr @ Omega(N)
#     w, v = np.linalg.eigh(Gamma)
