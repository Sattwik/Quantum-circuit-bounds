import pickle
import os
from datetime import datetime
from datetime import date
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import qutip
from matplotlib import rc
import matplotlib
import tensornetwork as tn
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.test_util import check_grads
import scipy

from vqa_bounds import maxcut1D, maxcut1Dlocal, graphs, circuit_utils, dual_utils

m = 6
lattice = graphs.define_lattice((m,))

graph = graphs.create_random_connectivity(lattice)

d = 4
p = 0.1

sys_obj = maxcut1D.MaxCut1D(graph, lattice, d, p)
sys_obj_local = maxcut1Dlocal.MaxCut1DLocal(graph, lattice, d, p)

gamma_init = np.ones(d)
beta_init = np.ones(d//2)

start = time.time()

circuit_params_init = np.concatenate((gamma_init, beta_init))

ub_array = np.concatenate((np.ones(d) * 2 * np.pi, np.ones(d//2) * np.pi))
bnds = scipy.optimize.Bounds(lb = 0, ub = ub_array)

circ_obj_over_opti, circ_opt_result = circuit_utils.optimize_circuit(circuit_params_init, bnds, sys_obj)

sys_obj_local.update_opt_circ_params(circ_opt_result.x)

dual_vars_init = jnp.zeros(sys_obj.total_num_vars)
dual_vars_init_local = jnp.ones(sys_obj_local.total_num_vars)

a_bound = -5
sigma_bound = 100

dual_obj_over_opti, dual_opt_result = dual_utils.optimize_dual(dual_vars_init, sys_obj, a_bound, sigma_bound, use_bounds = True)
dual_obj_over_opti_local, dual_opt_result_local = dual_utils.optimize_dual(dual_vars_init_local, sys_obj_local, a_bound, sigma_bound, use_bounds = True)

end = time.time()

print("Runtime (s) = ", end - start)

actual_sol = np.min(sys_obj.H.full())
clean_sol = circ_obj_over_opti[-1]
noisy_sol = sys_obj.primal_noisy()
noisy_bound = -dual_obj_over_opti[-1]
noisy_bound_local = -dual_obj_over_opti_local[-1]

print("actual_sol = ", actual_sol)
print("clean_sol = ", clean_sol)
print("noisy_sol = ", noisy_sol)
print("noisy_bound = ", noisy_bound)
print("noisy_bound_local = ", noisy_bound_local)

# file = "../vqa_data/maxcut1D_dual_debug.pkl"
#
# with open(file, "rb") as f:
#     sys_obj, error_circ_params, error_dual_vars = pickle.load(f)
#
# sys_obj.update_opt_circ_params(error_circ_params)
#
# new_dual = -1 * sys_obj.dual_obj(jnp.array(error_dual_vars))
#
# print(new_dual)
#
# dual_vars_init = jnp.ones(sys_obj.total_num_vars)
#
# dual_obj_over_opti, dual_opt_result = dual_utils.optimize_dual(dual_vars_init, sys_obj)



# sys_obj.update_opt_circ_params(np.concatenate((gamma, beta)))
#
# state = sys_obj.circuit_layer(layer_num = 0, var_tensor = sys_obj.rho_init_tensor)
# state = sys_obj.circuit_layer(layer_num = 1, var_tensor = state)
#
# print(np.linalg.norm(state.tensor - sys_obj.rho_init_tensor.tensor))

# dual fails for
# p = 0, d = 4, m = 6
# gamma = array([0.80018109, 0.80018109, 1.43863243, 1.43863243])
# beta = array([0.99118603, 1.25685429])
