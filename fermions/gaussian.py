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

#--------------- Tools ---------------#
def random_normal_hamiltonian_majorana(N: int, key: jnp.array):

    """
    Parameters
    ----------
    N: number of fermionic modes

    Returns
    -------
    A random f.g.h. (2N x 2N) in Majorana representation
    """

    key, subkey = jax.random.split(key)
    h = jax.random.normal(subkey, (2*N, 2*N))

    return h - h.T, key

def random_normal_corr_majorana(N: int, Ome: jnp.array, key: jnp.array):

    """
    Parameters
    ----------
    N: number of fermionic modes

    Returns
    -------
    Correlation matrix (2N x 2N) in Majorana representation of a random f.g.s.
    """

    # generate occupation probabilities
    key, subkey = jax.random.split(key)
    f = jax.random.uniform(subkey, (N,))

    F = jnp.diag(jnp.concatenate((f, 1.0-f)))

    key, subkey = jax.random.split(key)
    random_symm_mat = jax.random.normal(subkey, (2*N, 2*N))

    random_symm_mat = random_symm_mat + random_symm_mat.conj().T

    _, O = jnp.linalg.eigh(random_symm_mat)

    V = jnp.matmul(O.T, Ome)

    return jnp.matmul(V, jnp.matmul(F, V.conj().T)), f, V, O, key

def Omega(N: int) -> jnp.array:
    return jp.sqrt(1/2) * jnp.block(
                [[jnp.eye(N), jnp.eye(N)],
                 [1j * jnp.eye(N), -1j * jnp.eye(N)]])

def covariance_from_corr_major(Gamma_mjr: jnp.array, N: int):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix of f.g.s. (majorana rep.)
    N: number of fermionic modes
    """
    gamma = 1j * (2 * Gamma_mjr - jnp.identity(2 * N, dtype = complex))
    return gamma

def corr_major_from_covariance(gamma: jnp.array, N: int):
    """
    Parameters
    ----------
    gamma: Covariance matrix of f.g.s. (majorana rep.)
    N: number of fermionic modes
    """
    Gamma_mjr = (jnp.identity(2 * N, dtype = complex) - 1j * gamma)/2.0
    return Gamma_mjr

def corr_from_corr_major(gamma: jnp.array, N: int):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix of f.g.s. (majorana rep.)
    N: number of fermionic modes
    """
    Ome = Omega(N)
    Gamma = jnp.matmul(jnp.matmul(Ome.conj().T, Gamma_mjr), Ome)
    return Gamma

def corr_major_from_parenth(h: jnp.array, N: int):

    w, v = jnp.linalg.eig(-2j * h)
    w_Gamma_mjr = jnp.diag((jnp.ones(2*N) + jnp.exp(w)) ** (-1))

    Gamma_mjr = jnp.matmul(jnp.matmul(v, w_Gamma_mjr), v.conj().T)
    return Gamma_mjr

#--------------- Primal methods ---------------#
def energy(Gamma_mjr: jnp.array, h: jnp.array):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix of f.g.s. (majorana rep.)
    h: Hamiltonian of which to calculate expectation (majorana rep.)

    Returns
    -------
    Expected energy
    """

    return -1j * jnp.trace(jnp.matmul(h, Gamma_mjr))

def unitary_on_fgstate(Gamma_mjr: jnp.array, h: jnp.array):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix of f.g.s. (majorana rep.)
    h: Generator of Gaussian unitary in majorana rep..
       Gaussian unitary = e^{-iH} where H = i r^{\dagger} h r.

    Returns
    -------
    Gamma_mjr_prime: Correlation matrix of f.g.s. after unitary.
    """
    w, v = jnp.linalg.eig(2 * h)

    exp_p2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(w))), jnp.conj(jnp.transpose(v)))
    exp_m2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(-w))), jnp.conj(jnp.transpose(v)))

    return jnp.matmul(jnp.matmul(exp_p2h, Gamma_mjr), exp_m2h)

#--------------- Dual methods ---------------#
def trace_fgstate(parent_h: jnp.array, N: int):

    """
    Parameters
    ----------
    parent_h: Parent Hamiltonian of the f.g.s (Majorana rep.)
    N: number of fermionic modes

    Returns
    -------
    Trace of f.g.s.
    """

    w, v = jnp.linalg.eigh(1j * parent_h)

    positive_eigs = w[N:]

    return jnp.prod(jnp.exp(positive_eigs) + jnp.exp(-positive_eigs))

def unitary_on_fghamiltonian(s: jnp.array, h: jnp.array):
    """
    Parameters
    ----------
    s: Hamiltonian (from dual variable) to act on (Majorana representation)
    h: Generator of unitary (Majorana representation)

    Returns
    -------
    Majorana rep after unitary
    """

    w, v = jnp.linalg.eig(2 * h)

    exp_p2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(w))), jnp.conj(jnp.transpose(v)))
    exp_m2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(-w))), jnp.conj(jnp.transpose(v)))

    return jnp.matmul(jnp.matmul(exp_p2h, s), exp_m2h)

def noise_on_fghamiltonian(s: jnp.array, p: float, N: int):
    """
    Parameters
    ----------
    s: Hamiltonian (from dual variable) to act on (Majorana representation)
    p: noise probability
    N: number of fermionic modes

    Returns
    -------
    Majorana rep after depol. noise on individual fermions
    """

    s_prime = s

    for k in range(N):
        s_prime_zeroed_out = s_prime.at[k, :].set(jnp.zeros(2 * N))
        s_prime_zeroed_out = s_prime_zeroed_out.at[:, k].set(jnp.zeros(2 * N))
        s_prime_zeroed_out = s_prime_zeroed_out.at[k + N, :].set(jnp.zeros(2 * N))
        s_prime_zeroed_out = s_prime_zeroed_out.at[:, k + N].set(jnp.zeros(2 * N))

        s_prime = (1 - p) * s_prime + p * (s_prime_zeroed_out)

    return s_prime

def noise_on_fgstate_mc_sample(Gamma_mjr: jnp.array, p: float, N: int,
                               key: jnp.array):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix (Majorana rep.)
    p: noise probability
    N: number of fermionic modes
    key: to generate random mc sample

    Returns
    -------
    Correlation matrix (Majorana rep.) of f.g.s. after one MC sampling of noise
    (on every mode).
    """
    gamma = covariance_from_corr_major(Gamma_mjr, N)

    # print('gamma = ', gamma)

    key, subkey = jax.random.split(key)
    mc_probs = jax.random.uniform(key, shape = (N,))

    # print(mc_probs)

    gamma_noisy = gamma

    for k in range(N):
        if mc_probs[k] <= p:
            # print('k = ', k)
            gamma_noisy = gamma_noisy.at[k, :].set(jnp.zeros(2 * N))
            gamma_noisy = gamma_noisy.at[:, k].set(jnp.zeros(2 * N))
            gamma_noisy = gamma_noisy.at[k + N, :].set(jnp.zeros(2 * N))
            gamma_noisy = gamma_noisy.at[:, k + N].set(jnp.zeros(2 * N))

    # print('gamma_noisy = ', gamma_noisy)

    Gamma_mjr_noisy = corr_major_from_covariance(gamma_noisy, N)

    return Gamma_mjr_noisy, key

# def noise_on_fgstate_mc_realisation(Gamma_mjr: jnp.array, key: jnp.array,
#                                     p: float, N: int):
#     """
#     Parameters
#     ----------
#     Gamma_mjr: Correlation matrix (Majorana rep.)
#     key: to generate random mc samples
#     p: noise probability
#     N: number of fermionic modes
#
#     Returns
#     -------
#     Correlation matrix (Majorana rep.) of f.g.s. after MC sim. of noise
#     """
#
#     s_prime = s
#
#     for k in range(N):
#         s_prime_zeroed_out = s_prime.at[k, :].set(jnp.zeros(2 * N))
#         s_prime_zeroed_out = s_prime_zeroed_out.at[:, k].set(jnp.zeros(2 * N))
#         s_prime_zeroed_out = s_prime_zeroed_out.at[k + N, :].set(jnp.zeros(2 * N))
#         s_prime_zeroed_out = s_prime_zeroed_out.at[:, k + N].set(jnp.zeros(2 * N))
#
#         s_prime = (1 - p) * s_prime + p * (s_prime_zeroed_out)
#
#     return s_prime
