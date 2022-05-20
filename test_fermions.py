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

from fermions import gaussian, fermion_test_utils

#------------------------------------------------------------------------------#
#----------------------------------- TESTS ------------------------------------#
#------------------------------------------------------------------------------#

N = 4
seed = 69

key = jax.random.PRNGKey(seed)

s, key = gaussian.random_normal_hamiltonian_majorana(N, key)
h, key = gaussian.random_normal_hamiltonian_majorana(N, key)
parent_h, key = gaussian.random_normal_hamiltonian_majorana(N, key)
parent_h = parent_h/jnp.sqrt(N)

Ome = fermion_test_utils.Omega(N)
Gamma_mjr, f, V, O, key = gaussian.random_normal_corr_majorana(N, jnp.array(Ome), key)

print("Unitary on Hamiltonian test = ", fermion_test_utils.test_unitary_on_hamiltonian(np.array(s), np.array(h)))
print("Energy test = ", fermion_test_utils.test_energy(np.array(Gamma_mjr), np.array(f), np.array(V), np.array(h)))
print("FGS trace test = ", fermion_test_utils.test_trace_fgs(np.array(parent_h), N))
print("Noise on hamiltonian test = ", fermion_test_utils.test_noise_on_hamiltonian(np.array(s), 0.5, N))
print("Covariance def. test = ", fermion_test_utils.test_covariance_def(np.array(Gamma_mjr), np.array(f), np.array(O), N))
print("Unitary on fgs test = ", fermion_test_utils.test_unitary_on_fgstate(np.array(Gamma_mjr), np.array(f), np.array(O), np.array(V), np.array(h)))
print("Corr major from parenth test = ", fermion_test_utils.test_corr_major_from_parenth(np.array(h), N))
print("Noise on fgstate test = ", fermion_test_utils.test_noise_on_fgstate(np.array(s), np.array(h), N, 1.0, key))
