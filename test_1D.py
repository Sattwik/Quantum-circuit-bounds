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
from jax.config import config
import jax
config.update("jax_enable_x64", True)

from vqa_bounds import graphs, circuit_utils, dual_utils
from vqa_bounds import maxcut1D, maxcut1Dlocal, maxcut1Dlocalpurity, maxcut1Dlocalpuritylambda, maxcut1Dpurity

m = 4
lattice = graphs.define_lattice((m,))

graph = graphs.create_random_connectivity(lattice)

d_purity = 9
d_vne = 1
d = d_purity + d_vne
p = 0.1

sys_obj = maxcut1D.MaxCut1D(graph, lattice, d, p)
sys_obj_nc = maxcut1D.MaxCut1DNoChannel(graph, lattice, d, p)
sys_obj_local = maxcut1Dlocal.MaxCut1DLocal(graph, lattice, d, p)
sys_obj_local_pur = maxcut1Dlocalpurity.MaxCut1DLocalPurity(graph, lattice, d_purity, d_vne, p)
sys_obj_pur = maxcut1Dpurity.MaxCut1DPurity(graph, lattice, d_purity, d_vne, p)
sys_obj_local_pur_lambda = maxcut1Dlocalpuritylambda.MaxCut1DLocalPurityLambda(graph, lattice, d_purity, d_vne, p)

gamma_init = np.ones(d)
beta_init = np.ones(d//2)

start = time.time()

circuit_params_init = np.concatenate((gamma_init, beta_init))

ub_array = np.concatenate((np.ones(d) * 2 * np.pi, np.ones(d//2) * np.pi))
bnds = scipy.optimize.Bounds(lb = 0, ub = ub_array)

circ_obj_over_opti, circ_opt_result = circuit_utils.optimize_circuit(circuit_params_init, bnds, sys_obj)

sys_obj_local.update_opt_circ_params(circ_opt_result.x)
sys_obj_local_pur.update_opt_circ_params(circ_opt_result.x)
sys_obj_pur.update_opt_circ_params(circ_opt_result.x)
sys_obj_local_pur_lambda.update_opt_circ_params(circ_opt_result.x)

# dual_vars_init = jnp.zeros(sys_obj.total_num_vars)
# key = jax.random.PRNGKey(69)
# dual_vars_init_local = jax.random.normal(key, shape = (sys_obj_local.total_num_vars,))
# dual_vars_init_local_pur = dual_vars_init_local.at[sys_obj_local_pur.d_purity:].get()

a_bound = -10.0
sigma_bound = 100.0

dual_vars_init_nc = jnp.zeros((1,))
num_iters = 300
dual_obj_over_opti_nc, dual_opt_result_nc = \
dual_utils.optimize_dual(dual_vars_init_nc, sys_obj_nc, num_iters,
                         a_bound, sigma_bound, use_bounds = False)

a_nc = dual_opt_result_nc.x[0]

# dual_vars_init = jnp.zeros(sys_obj.total_num_vars)
# dual_vars_init = dual_vars_init.at[d-1].set(a_nc)
# dual_vars_init = dual_vars_init.at[:d-1].set(a_nc - 2)
# dual_vars_init_local = jnp.zeros(sys_obj_local.total_num_vars)
# dual_vars_init_local = dual_vars_init_local.at[d-1].set(a_nc)
# dual_vars_init_local = dual_vars_init_local.at[:d-1].set(a_nc - 2)

dual_vars_init_pur = 1e-9 * jnp.ones(sys_obj_pur.total_num_vars)
dual_vars_init_pur = dual_vars_init_pur.at[0].set(a_nc)

sys_obj_local_pur.a_vars = jnp.array(dual_opt_result_nc.x).at[0].set(a_nc-3)
dual_vars_init_local_pur = 1e-9 * jnp.ones(sys_obj_local_pur.total_num_vars)
# dual_vars_init_local_pur = dual_vars_init_local_pur.at[0].set(a_nc)

# dual_vars_init_local_pur_lambda = jnp.zeros(sys_obj_local_pur_lambda.total_num_vars)
# dual_vars_init_local_pur_lambda = dual_vars_init_local_pur_lambda.at[d-1].set(a_nc)
# dual_vars_init_local_pur_lambda = dual_vars_init_local_pur_lambda.at[:d-1].set(a_nc - 2)


# key = jax.random.PRNGKey(69)
# dual_vars_init_local = jax.random.normal(key, shape = (sys_obj_local.total_num_vars,))
# dual_vars_init_local_pur_lambda = dual_vars_init_local

# num_iters = 300
# dual_obj_over_opti, dual_opt_result = dual_utils.optimize_dual(dual_vars_init, sys_obj, num_iters, a_bound, sigma_bound, use_bounds = False)
#
# num_iters = 300
# dual_obj_over_opti_local, dual_opt_result_local = dual_utils.optimize_dual(dual_vars_init_local, sys_obj_local, num_iters, a_bound, sigma_bound, use_bounds = False)

# num_iters = 300
# dual_obj_over_opti_local_pur_lambda, dual_opt_result_local_pur_lambda = dual_utils.optimize_dual(dual_vars_init_local_pur_lambda, sys_obj_local_pur_lambda, num_iters, a_bound, sigma_bound, use_bounds = False)

# num_iters = 300
# dual_obj_over_opti_local_pur, dual_opt_result_local_pur = dual_utils.optimize_dual(dual_vars_init_local_pur, sys_obj_local_pur, num_iters, a_bound, sigma_bound, use_bounds = False, opt_method = 'Newton-CG')

# num_iters = 5000
# dual_obj_over_opti_pur, dual_opt_result_pur = dual_utils.optimize_dual(dual_vars_init_pur, sys_obj_pur, num_iters, a_bound, sigma_bound, opt_method = 'L-BFGS-B', use_bounds = False)

num_iters = 5000
dual_obj_over_opti_local_pur, dual_opt_result_local_pur = dual_utils.optimize_dual(dual_vars_init_local_pur, sys_obj_local_pur, num_iters, a_bound, sigma_bound, use_bounds = False, opt_method = 'L-BFGS-B')

# alpha = 0.1
# # eta = 1.0
# num_iters = int(1e4)
# method = "adam"
# dual_obj_over_opti_local_pur, dual_opt_result_local_pur = dual_utils.optax_optimize(sys_obj_local_pur, dual_vars_init_local_pur, alpha, num_iters, method)

# eta = 1e-2
# # # eta = 1.0
# num_iters = int(1e4)
# dual_obj_over_opti_pur, dual_opt_result_pur = dual_utils.gd_optimize(sys_obj_pur, dual_vars_init_pur, eta, num_iters)

# eta = 1e-6
# # # eta = 1.0
# num_iters = int(32e4)
# dual_obj_over_opti_local_pur, dual_opt_result_local_pur = dual_utils.gd_optimize(sys_obj_local_pur, dual_vars_init_local_pur, eta, num_iters)

end = time.time()

print("Runtime (s) = ", end - start)

actual_sol = np.min(sys_obj.H.full())
clean_sol = circ_obj_over_opti[-1]
noisy_sol = sys_obj.primal_noisy()
# noisy_bound = -sys_obj.dual_obj(jnp.array(dual_opt_result.x))
noisy_bound_nc = -sys_obj_nc.dual_obj(jnp.array(dual_opt_result_nc.x))
# noisy_bound_local = -sys_obj_local.dual_obj(jnp.array(dual_opt_result_local.x))
# noisy_bound_local_pur_lambda = -sys_obj_local_pur_lambda.dual_obj(jnp.array(dual_opt_result_local_pur_lambda.x))
noisy_bound_local_pur = -sys_obj_local_pur.dual_obj(jnp.array(dual_opt_result_local_pur.x))
# noisy_bound_pur = -sys_obj_pur.dual_obj(jnp.array(dual_opt_result_pur.x))

print("actual_sol = ", actual_sol)
print("clean_sol = ", clean_sol)
print("noisy_sol = ", noisy_sol)
# print("noisy_bound = ", noisy_bound)
print("noisy_bound_nc = ", noisy_bound_nc)
# print("noisy_bound_local = ", noisy_bound_local)
# print("noisy_bound_local_pur_lambda = ", noisy_bound_local_pur_lambda)
# print("noisy_bound_pur = ", noisy_bound_pur)
print("noisy_bound_local_pur = ", noisy_bound_local_pur)

plt.plot(dual_obj_over_opti_local_pur)
plt.show()

# plt.plot(dual_obj_over_opti_pur)
# plt.show()




# init_grad = sys_obj_local_pur.dual_grad_last_layer(dual_vars_init_local_pur)
# init_grad = init_grad.at[0].set(0)
#
# func_list = []
# step_size_list = np.linspace(1e-9, 1e-5, 100)
#
# for i in range(len(step_size_list)):
#     func_list.append(sys_obj_local_pur.dual_obj(dual_vars_init_local_pur-step_size_list[i] * init_grad))
#
# plt.plot(step_size_list, func_list)
# plt.show()

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
