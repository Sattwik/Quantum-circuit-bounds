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
import matplotlib.pyplot as plt

from fermions import gaussian, fermion_test_utils

#------------------------------------------------------------------------------#
#----------------------------------- TESTS ------------------------------------#
#------------------------------------------------------------------------------#

rng = np.random.default_rng()

N = 100
d = 5
seed = rng.integers(low=0, high=100, size=1)[0]
key = jax.random.PRNGKey(seed)

params = gaussian.PrimalParams(N, d, key)

key, subkey = jax.random.split(params.key_after_ham_gen)
theta_init = jax.random.uniform(key, shape = (d,), minval = 0.0, maxval = 2 * np.pi)

# objective = gaussian.circ_obj(theta_init, params)

obj_over_opti, circ_opt_result = gaussian.optimize_circuit(theta_init, params)

w_target, v_target = jnp.linalg.eig(1j * params.h_target)
w_target = np.real(w_target)
gs_energy_target = sum(w_target[w_target < 0])

print("gs energy = ", gs_energy_target)
print("circuit val = ", obj_over_opti[-1])

# plt.plot(obj_over_opti)
# plt.show()
