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
from vqa_bounds import sumsigma1DNN

m = 6
lattice = graphs.define_lattice((m,))

d_purity = 3
d_vne = 1
d = 4
p = 0.1


key = jax.random.PRNGKey(69)
# sys_obj_nc = maxcut1D.MaxCut1DNoChannel(graph, lattice, d, p)
sys_obj_local_pur = sumsigma1DNN.SumSigma1DNN(key, lattice, d_purity, d_vne, p, mode = 'local')
sys_obj_nc = sumsigma1DNN.SumSigma1DNN(key, lattice, d_purity, d_vne, p, mode = 'nc')

a_bound = -10.0
sigma_bound = 100.0

dual_vars_init_nc = jnp.zeros((1,))
num_iters = 300
dual_obj_over_opti_nc, dual_opt_result_nc = \
dual_utils.optimize_dual(dual_vars_init_nc, sys_obj_nc, num_iters,
                         a_bound, sigma_bound, use_bounds = False)

a_nc = dual_opt_result_nc.x[0]
#
# # dual_vars_init = jnp.zeros(sys_obj.total_num_vars)
# # dual_vars_init = dual_vars_init.at[d-1].set(a_nc)
# # dual_vars_init = dual_vars_init.at[:d-1].set(a_nc - 2)
# # dual_vars_init_local = jnp.zeros(sys_obj_local.total_num_vars)
# # dual_vars_init_local = dual_vars_init_local.at[d-1].set(a_nc)
# # dual_vars_init_local = dual_vars_init_local.at[:d-1].set(a_nc - 2)
#
# dual_vars_init_pur = 1e-9 * jnp.ones(sys_obj_pur.total_num_vars)
# dual_vars_init_pur = dual_vars_init_pur.at[0].set(a_nc)
#
# sys_obj_local_pur.a_vars = dual_opt_result_nc.x
sys_obj_local_pur.a_vars = jnp.array(dual_opt_result_nc.x).at[0].set(a_nc)
sys_obj_local_pur.Lambdas = jnp.exp(sys_obj_local_pur.a_vars)
dual_vars_init_local_pur = 1e-9 * jnp.ones(sys_obj_local_pur.total_num_vars)
# dual_vars_init_local_pur = dual_vars_init_local_pur.at[0].set(a_nc)

# key = jax.random.PRNGKey(87)
# dual_vars_init_local_pur = jax.random.normal(key, shape = (sys_obj_local_pur.total_num_vars,))
# dual_vars_init_local_pur = dual_vars_init_local_pur.at[0].set(a_nc)
#
# # dual_vars_init_local_pur_lambda = jnp.zeros(sys_obj_local_pur_lambda.total_num_vars)
# # dual_vars_init_local_pur_lambda = dual_vars_init_local_pur_lambda.at[d-1].set(a_nc)
# # dual_vars_init_local_pur_lambda = dual_vars_init_local_pur_lambda.at[:d-1].set(a_nc - 2)
#
#
# # key = jax.random.PRNGKey(69)
# dual_vars_init_local = jax.random.normal(key, shape = (sys_obj_local.total_num_vars,))
# dual_vars_init_local_pur_lambda = dual_vars_init_local
#
# # num_iters = 300
# # dual_obj_over_opti, dual_opt_result = dual_utils.optimize_dual(dual_vars_init, sys_obj, num_iters, a_bound, sigma_bound, use_bounds = False)
# #
# # num_iters = 300
# # dual_obj_over_opti_local, dual_opt_result_local = dual_utils.optimize_dual(dual_vars_init_local, sys_obj_local, num_iters, a_bound, sigma_bound, use_bounds = False)
#
# # num_iters = 300
# # dual_obj_over_opti_local_pur_lambda, dual_opt_result_local_pur_lambda = dual_utils.optimize_dual(dual_vars_init_local_pur_lambda, sys_obj_local_pur_lambda, num_iters, a_bound, sigma_bound, use_bounds = False)
#
# num_iters = 300
# dual_obj_over_opti_local_pur, dual_opt_result_local_pur = dual_utils.optimize_dual(dual_vars_init_local_pur, sys_obj_local_pur, num_iters, a_bound, sigma_bound, use_bounds = False, opt_method = 'L-BFGS-B')
#
# num_iters = 5000
# dual_obj_over_opti_pur, dual_opt_result_pur = dual_utils.optimize_dual(dual_vars_init_pur, sys_obj_pur, num_iters, a_bound, sigma_bound, opt_method = 'L-BFGS-B', use_bounds = False)
#
num_iters = 5000
dual_obj_over_opti_local_pur, dual_opt_result_local_pur = dual_utils.optimize_dual(dual_vars_init_local_pur, sys_obj_local_pur, num_iters, a_bound, sigma_bound, use_bounds = False, opt_method = 'L-BFGS-B')
#
# alpha = 0.1
# # eta = 1.0
# num_iters = int(1e4)
# method = "adam"
# dual_obj_over_opti_local_pur, dual_opt_result_local_pur = dual_utils.optax_optimize(sys_obj_local_pur, dual_vars_init_local_pur, alpha, num_iters, method)
#
# eta = 1e-2
# # # eta = 1.0
# num_iters = int(1e4)
# dual_obj_over_opti_pur, dual_opt_result_pur = dual_utils.gd_optimize(sys_obj_pur, dual_vars_init_pur, eta, num_iters)
#
# eta = 1e-6
# # # eta = 1.0
# num_iters = int(16e4)
# dual_obj_over_opti_local_pur, dual_opt_result_local_pur = dual_utils.gd_optimize(sys_obj_local_pur, dual_vars_init_local_pur, eta, num_iters)
#
# end = time.time()
#
# print("Runtime (s) = ", end - start)
#
actual_sol = np.min(sys_obj_local_pur.H.full())
# clean_sol = circ_obj_over_opti[-1]
noisy_sol = sys_obj_local_pur.primal_noisy()
# # noisy_bound = -sys_obj.dual_obj(jnp.array(dual_opt_result.x))
noisy_bound_nc = -sys_obj_nc.dual_obj(jnp.array(dual_opt_result_nc.x))
# # noisy_bound_local = -sys_obj_local.dual_obj(jnp.array(dual_opt_result_local.x))
# # noisy_bound_local_pur_lambda = -sys_obj_local_pur_lambda.dual_obj(jnp.array(dual_opt_result_local_pur_lambda.x))
noisy_bound_local_pur = -sys_obj_local_pur.dual_obj(jnp.array(dual_opt_result_local_pur.x))
# noisy_bound_pur = -sys_obj_pur.dual_obj(jnp.array(dual_opt_result_pur.x))
#
print("actual_sol = ", actual_sol)
# print("clean_sol = ", clean_sol)
print("noisy_sol = ", noisy_sol)
# # print("noisy_bound = ", noisy_bound)
print("noisy_bound_nc = ", noisy_bound_nc)
# # print("noisy_bound_local = ", noisy_bound_local)
# # print("noisy_bound_local_pur_lambda = ", noisy_bound_local_pur_lambda)
# print("noisy_bound_pur = ", noisy_bound_pur)
print("noisy_bound_local_pur = ", noisy_bound_local_pur)
#
plt.plot(dual_obj_over_opti_local_pur)
plt.show()
#
# plt.plot(dual_obj_over_opti_pur)
# plt.show()




# init_grad = sys_obj_local_pur.dual_grad(dual_vars_init_local_pur)
# # init_grad = init_grad.at[0].set(0)
#
# func_list = []
# step_size_list = np.linspace(1e-12, 1e-9, 100)
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
