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

    V = jnp.matmul(O, Ome)

    return jnp.matmul(V, jnp.matmul(F, V.conj().T)), f, V, key

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

#--------------- Primal methods ---------------#
def trace_fgs(parent_h: jnp.array, N: int):

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

#--------------- Dual methods ---------------#
def unitary_on_hamiltonian(s: jnp.array, h: jnp.array):

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

def noise_on_hamiltonian(s: jnp.array, p: float, N: int):

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

        s_prime_zeroed_out = s_prime.at[k, :].set(jnp.zeros(N))
        s_prime_zeroed_out = s_prime_zeroed_out.at[:, k].set(jnp.zeros(N))

        s_prime = (1 - p) * s_prime + p * (s_prime_zeroed_out)

    return s_prime
