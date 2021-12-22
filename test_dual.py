import pickle
import os
from datetime import datetime
from datetime import date

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

from vqa import graphs, problems, algorithms, dual, dual_jax

#------------------------------------------------------------------------------#
#----------------------------------- TEST 1 -----------------------------------#
#------------------------------------------------------------------------------#

# TEST1: gamma = 0, beta = 0 in the circuit layer should not change the init state
# PASSED: 12-12-21

# m = 2
# n = 3
# lattice = graphs.define_lattice(m = m, n = n)
#
# graph = graphs.create_random_connectivity(lattice)
#
# maxcut_obj = problems.MaxCut(graph, lattice)
#
# d = 2

# gamma = np.zeros(d)
# beta = np.zeros(d)
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, d = d, gamma = gamma, beta = beta, p = 0)
#
# state = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)
# state = dual_obj.circuit_layer(layer_num = 1, var_tensor = state)
#
# print(np.linalg.norm(state.tensor - dual_obj.rho_init_tensor.tensor))

#------------------------------------------------------------------------------#
#----------------------------------- TEST 2 -----------------------------------#
#------------------------------------------------------------------------------#

# TEST2: gamma != 0, beta != 0 in the circuit layer should match with
# direct computation
# PASSED: 12-12-21

# m = 2
# n = 3
# lattice = graphs.define_lattice(m = m, n = n)
#
# graph = graphs.create_random_connectivity(lattice)
#
# maxcut_obj = problems.MaxCut(graph, lattice)
#
# d = 2
#
# gamma = np.random.uniform(low = 0, high = 2 * np.pi, size = d)
# beta = np.random.uniform(low = 0, high = np.pi, size = d)
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, d = d, gamma = gamma, beta = beta, p = 0)
# state = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)
# state = dual_obj.circuit_layer(layer_num = 1, var_tensor = state)
#
# state_dm = dual_obj.tensor_2_mat(state.tensor)
#
# psi_after_step = maxcut_obj.init_state()
# psi_after_step = maxcut_obj.Up(gamma[0]) * psi_after_step
# psi_after_step = maxcut_obj.Um(beta[0]) * psi_after_step
#
# psi_after_step = maxcut_obj.Up(gamma[1]) * psi_after_step
# psi_after_step = maxcut_obj.Um(beta[1]) * psi_after_step
#
# rho_after_step = psi_after_step * psi_after_step.dag()
# rho = rho_after_step.full()
#
# print(np.linalg.norm(state_dm - rho))

#------------------------------------------------------------------------------#
#----------------------------------- TEST 3 -----------------------------------#
#------------------------------------------------------------------------------#

# TEST3: see if the superoperator noise works
# PASSED: 12-12-21

# m = 2
# n = 2
# lattice = graphs.define_lattice(m = m, n = n)
#
# graph = graphs.create_random_connectivity(lattice)
#
# maxcut_obj = problems.MaxCut(lattice, lattice)
#
# d = 2
#
# # gamma = np.random.uniform(low = 0, high = 2 * np.pi, size = d)
# # beta = np.random.uniform(low = 0, high = np.pi, size = d)
#
# gamma = np.array([2.23703565, 5.16466766])
# beta = np.array([2.13005531, 2.02307449])
#
# p = 0.21
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, d = d, gamma = gamma, beta = beta, p = p)
#
# state = dual_obj.rho_init_tensor
# state = dual_obj.noise_layer(var_tensor = state)
# state = dual_obj.circuit_layer(layer_num = 0, var_tensor = state)
# state = dual_obj.noise_layer(var_tensor = state)
# state = dual_obj.circuit_layer(layer_num = 1, var_tensor = state)
#
# state_dm = dual_obj.tensor_2_mat(state.tensor)
#
# state_dm_2, primal_noisy = dual_obj.primal_noisy()
#
# X = []
# Y = []
# Z = []
#
# num_sites = m * n
#
# for site in lattice:
#
#     i = maxcut_obj.site_nums[site]
#
#     X.append(qutip.tensor([qutip.qeye(2)] * i + [qutip.sigmax()] + [qutip.qeye(2)] * (num_sites - i - 1)))
#     Y.append(qutip.tensor([qutip.qeye(2)] * i + [qutip.sigmay()] + [qutip.qeye(2)] * (num_sites - i - 1)))
#     Z.append(qutip.tensor([qutip.qeye(2)] * i + [qutip.sigmaz()] + [qutip.qeye(2)] * (num_sites - i - 1)))
#
# I = qutip.tensor([qutip.qeye(2)] * num_sites)
#
# def noise_layer(rho: qutip.Qobj, prob_obj: problems.MaxCut, p: float):
#
#     for site in prob_obj.lattice:
#
#         i = prob_obj.site_nums[site]
#
#         rho = (1 - 3 * p/4) * rho +\
#               (p/4) * (X[i] * rho * X[i] + Y[i] * rho * Y[i] + Z[i] * rho * Z[i])
#
#     return rho
#
# # psi_after_step = maxcut_obj.init_state()
# # psi_after_step = maxcut_obj.Up(gamma[0]) * psi_after_step
# # psi_after_step = maxcut_obj.Um(beta[0]) * psi_after_step
#
# # rho_after_step = psi_after_step * psi_after_step.dag()
#
# # rho_after_step = qutip.Qobj(init_mat, dims = (maxcut_obj.init_state() * maxcut_obj.init_state().dag()).dims)
# rho_after_step = maxcut_obj.init_state() * maxcut_obj.init_state().dag()
#
# rho_after_step = noise_layer(rho_after_step, maxcut_obj, p)
# rho_after_step = maxcut_obj.Up(gamma[0]).dag() * rho_after_step * maxcut_obj.Up(gamma[0])
# rho_after_step = maxcut_obj.Um(beta[0]).dag() * rho_after_step * maxcut_obj.Um(beta[0])
#
# rho_after_step = noise_layer(rho_after_step, maxcut_obj, p)
# rho_after_step = maxcut_obj.Up(gamma[1]).dag() * rho_after_step * maxcut_obj.Up(gamma[1])
# rho_after_step = maxcut_obj.Um(beta[1]).dag() * rho_after_step * maxcut_obj.Um(beta[1])
#
# rho = rho_after_step.full()
#
# print(np.linalg.norm(state_dm - rho))
# print(np.linalg.norm(state_dm_2 - rho))

#------------------------------------------------------------------------------#
#----------------------------------- TEST 4 -----------------------------------#
#------------------------------------------------------------------------------#

# TEST4: check the variable loading operations and entropy bounds

# lattice = graphs.define_lattice(m = 1, n = 2)
# maxcut_obj = problems.MaxCut(lattice, lattice)
# p = 5
#
# gamma0 = np.pi/13
# beta0 = np.pi/19
#
# p_noise = 0.9
#
# gamma = np.array([gamma0])
# beta = np.array([beta0])
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma, beta = beta, p_noise = p_noise)
#
# # vars = np.arange((1 + dual_obj.num_diag_elements + 2 * dual_obj.num_tri_elements) * p)
#
# dual_obj.assemble_vars_into_tensors(vars)
#
# print(dual_obj.entropy_bounds)

#------------------------------------------------------------------------------#
#----------------------------------- TEST 5 -----------------------------------#
#------------------------------------------------------------------------------#

# TEST5: check that the dual function at randomly picked values of the dual
# variables is real and less than the primal value.

# num_var_rand = 100
#
# for i in range(num_var_rand):
#
#     print(i)
#
#     lambdas_init = np.random.uniform(low = 1e-9, high = 1e3, size = p)
#     sigma_vars_init = np.random.uniform(low = -1e3, high = 1e3, size = dual_obj.len_vars - p)
#     vars_init = np.concatenate((lambdas_init, sigma_vars_init))
#
#     p_lb = dual_obj.objective(vars_init)
#     p_lb_jax = dual_jax.objective_external_dual_JAX(jnp.array(vars_init), dual_obj_jax)
#
#     if np.isnan(p_lb):
#
#         break
#
#     if -float(p_lb_jax) > p_noisy:
#
#         print("Error in dual function")
#
#     print("JAX val = ", -float(p_lb_jax))
#     print("np val = ", p_lb)

#------------------------------------------------------------------------------#
#----------------------------------- TEST 7 -----------------------------------#
#------------------------------------------------------------------------------#

# Check the optimised dual value obtained from the three duals when
# p = 1. Things to check - are they close to the primal? do they match the
# analytical dual solution? as they are relaxations of each other, check that
# they are ordered as expected.

# # ---- define the lattice/graph/problem ----- #
# lattice = graphs.define_lattice(m = 2, n = 3)
# graph = graphs.create_random_connectivity(lattice)
# maxcut_obj = problems.MaxCut(graph, lattice)
# d = 2
#
# # ---- calculate the actual maxcut solution ----- #
# primal_actual = np.min(maxcut_obj.H.full())
#
# # ---- calculate the clean maxcut solution ----- #
# gamma_init = np.random.uniform(low = 0, high = 2 * np.pi, size = d)
# beta_init = np.random.uniform(low = 0, high = np.pi, size = d)
# gamma_beta_init = np.concatenate((gamma_init, beta_init))
# mc_probs = np.zeros(graph.number_of_nodes())
# p = 0
# qaoa_obj = algorithms.QAOA(maxcut_obj, d, p, mc_probs)
# obj_over_opti, opt_result = algorithms.optimize_QAOA(gamma_beta_init, qaoa_obj)
# primal_clean = obj_over_opti[-1]
# gamma_beta_clean = opt_result.x
# gamma_clean = np.split(gamma_beta_clean, 2)[0]
# beta_clean = np.split(gamma_beta_clean, 2)[1]
#
# # ---- calculate the max dual value of full primal ----- #
# # noise probability
# p = 0.7
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, d = d, gamma = gamma_clean, beta = beta_clean, p = p)
# _, primal_noisy = dual_obj.primal_noisy()
# primal_noisy = np.real(primal_noisy)
#
# dual_obj_jax = dual_jax.MaxCutDualJAX(prob_obj = maxcut_obj, d = d,
#             gamma = jnp.array(gamma_clean), beta = jnp.array(beta_clean), p = p)
#
# _, primal_noisy_jax = dual_obj_jax.primal_noisy()
# primal_noisy_jax = float(jnp.real(primal_noisy_jax))
#
# a_vars_init = np.zeros(d)
# sigma_vars_init = np.zeros(dual_obj.len_vars - d)
# vars_init = np.concatenate((a_vars_init, sigma_vars_init))
#
# # dual_value, vars_opt = dual_jax.adam_external_dual_JAX(jnp.array(vars_init),
#                                     # dual_obj_jax, alpha = 0.1, num_steps = 100)
#
# # dual_obj.assemble_vars_into_tensors(np.array(vars_opt))
#
# obj_over_opti, opt_result = dual_jax.optimize_external_dual_JAX(vars_init, dual_obj_jax)
#
# print("primal = ", primal_noisy_jax)
# print("Dual = ", -obj_over_opti[-1])
# print("Trace = ", jnp.trace(dual_obj_jax.H_problem)/dual_obj_jax.dim)
#
# # ---- calculate the max dual value of global version of primal ----- #
#
# dual_obj_jax_global = dual_jax.MaxCutDualJAXGlobal(prob_obj = maxcut_obj, d = d, p = p)
#
# primal_noisy_global = float(jnp.real(dual_obj_jax_global.primal_noisy()))
#
# # dual_value_global, vars_opt_global = dual_jax.adam_external_dual_JAX(jnp.array(vars_init),
# #                                     dual_obj_jax_global, alpha = 0.08, num_steps = 200)
#
# obj_over_opti_global, opt_result_global = \
#             dual_jax.optimize_external_dual_JAX(vars_init, dual_obj_jax_global)
#
# print("Primal (global) = ", primal_noisy_global)
# print("Dual (global) = ", -obj_over_opti_global[-1])
#
# # ---- calculate the max dual value of no channel version of primal ----- #
#
# dual_obj_jax_nochannel = dual_jax.MaxCutDualJAXNoChannel(prob_obj = maxcut_obj, d = d, p = p)
#
# a_init = np.ones(d)
# # dual_value_nochannel, vars_opt_nochannel = dual_jax.adam_external_dual_JAX(jnp.array(a_init),
# #                                     dual_obj_jax_nochannel, alpha = 0.08, num_steps = 200)
#
# obj_over_opti_nc, opt_result_nc = \
#         dual_jax.optimize_external_dual_JAX(a_init, dual_obj_jax_nochannel)
#
# print("Primal = ", primal_noisy_jax)
# print("Dual (no channel) = ", -obj_over_opti_nc[-1])
#
# # ---- calculate the max dual value of no channel version of primal ----- #
#
# lmbda_range = np.linspace(0.01, 10000, 1000)
# g_array = []
# a_vec = a_init.copy()
# for l in lmbda_range:
#     a_vec[0] = l
#     g_array.append(float(dual_obj_jax_nochannel.objective(jnp.array(a_vec))))

#------------------------------------------------------------------------------#
#----------------------------------- TEST 8 -----------------------------------#
#------------------------------------------------------------------------------#

# # ---- define the lattice/graph/problem ----- #
# lattice = graphs.define_lattice(m = 2, n = 2)
# graph = graphs.create_random_connectivity(lattice)
# maxcut_obj = problems.MaxCut(lattice, lattice)
# d = 2
#
# # ---- calculate the actual maxcut solution ----- #
# primal_actual = np.min(maxcut_obj.H.full())
#
# # ---- calculate the max dual value of full primal ----- #
# # noise probability
# p = 0.21
#
# # case 1
# gamma = np.array([5.49778548, 4.07564258])
# beta = np.array([0.39269847, 0.        ])
#
# # gamma = np.array([0.0, 0.0])
# # beta = np.array([0.0, 0.0])
#
# # gamma = np.array([0.0])
# # beta = np.array([1.0])
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, d = d, gamma = gamma, beta = beta, p = p)
# _, primal_noisy = dual_obj.primal_noisy()
# primal_noisy = np.real(primal_noisy)
#
# dual_obj_jax = dual_jax.MaxCutDualJAX(prob_obj = maxcut_obj, d = d,
#             gamma = jnp.array(gamma), beta = jnp.array(beta), p = p)
# _, primal_noisy_jax = dual_obj_jax.primal_noisy()
# primal_noisy_jax = float(jnp.real(primal_noisy_jax))
#
# a_vars_init = np.zeros(d)
# sigma_vars_init = np.zeros(dual_obj.len_vars - d)
# vars_init = np.concatenate((a_vars_init, sigma_vars_init))
# dual_value, vars_opt = dual_jax.adam_external_dual_JAX(jnp.array(vars_init),
#                                     dual_obj_jax, alpha = 0.1, num_steps = 100)
#
# print("primal = ", primal_noisy_jax)
# print("Dual = ", -dual_value[-1])
#
# # case 2
# gamma = np.array([2.23703565, 5.16466766])
# beta = np.array([2.13005531, 2.02307449])
#
# # gamma = np.array([1.0, 1.0])
# # beta = np.array([1.0, 0.0])
#
# # gamma = np.array([1.0])
# # beta = np.array([1.0])
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, d = d, gamma = gamma, beta = beta, p = p)
# _, primal_noisy = dual_obj.primal_noisy()
# primal_noisy = np.real(primal_noisy)
#
# dual_obj_jax = dual_jax.MaxCutDualJAX(prob_obj = maxcut_obj, d = d,
#             gamma = jnp.array(gamma), beta = jnp.array(beta), p = p)
# _, primal_noisy_jax = dual_obj_jax.primal_noisy()
# primal_noisy_jax = float(jnp.real(primal_noisy_jax))
#
# a_vars_init = np.zeros(d)
# sigma_vars_init = np.zeros(dual_obj.len_vars - d)
# vars_init = np.concatenate((a_vars_init, sigma_vars_init))
# dual_value, vars_opt = dual_jax.adam_external_dual_JAX(jnp.array(vars_init),
#                                     dual_obj_jax, alpha = 0.1, num_steps = 100)
#
# print("primal = ", primal_noisy_jax)
# print("Dual = ", -dual_value[-1])
#
# dual_obj.assemble_vars_into_tensors(np.array(vars_opt))
#
# sigma1 = dual_obj.tensor_2_mat(dual_obj.Sigmas[0].tensor)
# epsilon_1_rho = dual_obj.rho_init_tensor
# epsilon_1_rho = dual_obj.circuit_layer(layer_num = 0, var_tensor = epsilon_1_rho, mode = 'primal')
# epsilon_1_rho = dual_obj.noise_layer(epsilon_1_rho)
# epsilon_1_rho = dual_obj.tensor_2_mat(epsilon_1_rho.tensor)
# init_state_term_1 = -np.trace(np.matmul(sigma1, epsilon_1_rho))
#
# rho_init_mat = dual_obj.tensor_2_mat(dual_obj.rho_init_tensor.tensor)
# epsilon_1_dag_sigma1 = dual_obj.Sigmas[0]
# epsilon_1_dag_sigma1 = dual_obj.noise_layer(epsilon_1_dag_sigma1)
# epsilon_1_dag_sigma1 = dual_obj.circuit_layer(layer_num = 0, var_tensor = epsilon_1_dag_sigma1, mode = 'dual')
# epsilon_1_dag_sigma1 = dual_obj.tensor_2_mat(epsilon_1_dag_sigma1.tensor)
# init_state_term_2 = -np.trace(np.matmul(epsilon_1_dag_sigma1, rho_init_mat))
#
# print(init_state_term_1)
# print(init_state_term_2)
