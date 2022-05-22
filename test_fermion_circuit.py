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

N = 80
d = 3
seed = rng.integers(low=0, high=100, size=1)[0]
key = jax.random.PRNGKey(seed)

circ_params = gaussian.PrimalParams(N, d, key)

key, subkey = jax.random.split(circ_params.key_after_ham_gen)
theta_init = jax.random.uniform(key, shape = (d,), minval = 0.0, maxval = 2 * np.pi)

# objective = gaussian.circ_obj(theta_init, params)

circ_obj_over_opti, circ_opt_result = gaussian.optimize_circuit(theta_init, circ_params)

w_target, v_target = jnp.linalg.eig(1j * circ_params.h_target)
w_target = np.real(w_target)
gs_energy_target = sum(w_target[w_target < 0])

actual_sol = gs_energy_target
print("gs energy = ", actual_sol)

clean_sol = circ_obj_over_opti[-1]
print("clean sol = ", clean_sol)

p = 0.001
dual_params = gaussian.DualParams(circ_params, jnp.array(circ_opt_result.x), p)

key, subkey = jax.random.split(key)
dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))

dual_obj_over_opti, dual_opt_result = gaussian.optimize_dual(dual_vars_init, dual_params)

noisy_bound = -dual_obj_over_opti[-1]
print("noisy bound = ", noisy_bound)

print("bound <= sol? ", noisy_bound < clean_sol)






# plt.plot(obj_over_opti)
# plt.show()
