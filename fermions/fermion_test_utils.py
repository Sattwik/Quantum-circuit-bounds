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

#--------------- Testing methods ---------------#
def full_state_from_corr(Gamma_mjr: np.array):

    Gamma = Omega(N).conj().T @ Gamma_mjr @ Omega(N)
    w, v = np.linalg.eigh(Gamma)

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

def test_unitary_on_hamiltonian(s: np.array, h: np.array):
    s_full = full_op_from_majorana(s)
    h_full = full_op_from_majorana(h)

    U_full = (-1j * h_full).expm()

    s_prime_full_other = U_full * s_full * U_full.dag()

    s_prime = np.array(gaussian.unitary_on_hamiltonian(jnp.array(s), jnp.array(h)))
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

    trace_majorana = gaussian.trace_fgs(jnp.array(parent_h), N)

    return trace_full - trace_majorana, trace_full, trace_majorana
