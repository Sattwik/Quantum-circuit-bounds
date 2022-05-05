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
#----------------------------------- TEST 1 -----------------------------------#
#------------------------------------------------------------------------------#

# TEST1: unitary on hamiltonian

N = 4
seed = 69

key = jax.random.PRNGKey(seed)

s, key = gaussian.random_normal_hamiltonian_majorana(N, key)
h, key = gaussian.random_normal_hamiltonian_majorana(N, key)
parent_h, key = gaussian.random_normal_hamiltonian_majorana(N, key)
parent_h = parent_h/jnp.sqrt(N)

Ome = fermion_test_utils.Omega(N)
Gamma_mjr, f, V, key = gaussian.random_normal_corr_majorana(N, jnp.array(Ome), key)

print(fermion_test_utils.test_unitary_on_hamiltonian(np.array(s), np.array(h)))
print(fermion_test_utils.test_energy(np.array(Gamma_mjr), np.array(f), np.array(V), np.array(h)))
print(fermion_test_utils.test_trace_fgs(np.array(parent_h), N))
