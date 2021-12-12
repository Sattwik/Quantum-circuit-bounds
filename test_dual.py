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


m = 2
n = 3
lattice = graphs.define_lattice(m = m, n = n)

graph = graphs.create_random_connectivity(lattice)

maxcut_obj = problems.MaxCut(graph, lattice)

p = 1

# TEST1: gamma = 0, beta = 0 in the circuit layer should not change the init state
# PASSED
# gamma = np.array([0])
# beta = np.array([0])
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma, beta = beta, p_noise = 0)
#
# state = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)
#
# print(np.linalg.norm(state.tensor - dual_obj.rho_init_tensor.tensor))

# TEST2: gamma != 0, beta != 0 in the circuit layer should match with
# direct computation
# FAILED: error in problem unitary in dual

# gamma = np.array([np.pi/4])
# beta = np.array([np.pi/19])
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma, beta = beta, p_noise = 0)
# state = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)
#
# state_dm = dual_obj.tensor_2_mat(state.tensor)
#
# psi_after_step = maxcut_obj.init_state()
# psi_after_step = maxcut_obj.Up(gamma[0]) * psi_after_step
# psi_after_step = maxcut_obj.Um(beta[0]) * psi_after_step
#
# rho_after_step = psi_after_step * psi_after_step.dag()
#
# rho = rho_after_step.full()
#
# print(np.linalg.norm(state_dm - rho))

# TEST3: gamma != 0, see if the superoperator unitary works

# gamma0 = np.pi/13
# beta0 = np.pi/19
#
# gamma1 = np.pi/4
# beta1 = 2 * np.pi/3
#
# gamma2 = 3 * np.pi/4
# beta2 = np.pi/5
#
# gamma = np.array([gamma0, gamma1, gamma2])
# beta = np.array([beta0, beta1, beta2])
#
# maxcut_obj = problems.MaxCut(graph, lattice)
# p = 3
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma, beta = beta, p_noise = 0)
# state_0 = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)
# state_1 = dual_obj.circuit_layer(layer_num = 1, var_tensor = state_0)
# state_2 = dual_obj.circuit_layer(layer_num = 2, var_tensor = state_1)
#
# state_dm = dual_obj.tensor_2_mat(state_2.tensor)
#
# psi_after_step = maxcut_obj.init_state()
# psi_after_step = maxcut_obj.Up(gamma[0]) * psi_after_step
# psi_after_step = maxcut_obj.Um(beta[0]) * psi_after_step
#
# psi_after_step = maxcut_obj.Up(gamma[1]) * psi_after_step
# psi_after_step = maxcut_obj.Um(beta[1]) * psi_after_step
#
# psi_after_step = maxcut_obj.Up(gamma[2]) * psi_after_step
# psi_after_step = maxcut_obj.Um(beta[2]) * psi_after_step
#
# rho_after_step = psi_after_step * psi_after_step.dag()
#
# rho_init_direct = (maxcut_obj.init_state() * maxcut_obj.init_state().dag()).full()
#
# rho = rho_after_step.full()
#
# # Hp = -0.5*(qutip.tensor(qutip.qeye(2), qutip.qeye(2)) - qutip.tensor(qutip.sigmaz(), qutip.sigmaz()))
# # U = (-1j * Hp * gamma0).expm().full()
# # U_dag = U.conj().T
#
# # # (ja jb) (ia ib) -> (ja jb ia ib)
# # U_tensor = U.flatten()
# # # (ja jb ia ib) -> (ja) (jb) (ia) (ib)
# # U_tensor = U_tensor.reshape((2,2,2,2))
# # # (ja) (jb) (ia) (ib) -> (ja) (ia) (jb) (ib)
# # U_tensor = U_tensor.transpose([0, 2, 1, 3])
# #
# # # (iap ibp) (kap kbp) -> (iap ibp kap kbp)
# # U_dag_tensor = U_dag.flatten()
# # # (iap ibp kap kbp) -> (iap) (ibp) (kap) (kbp)
# # U_dag_tensor = U_dag_tensor.reshape((2,2,2,2))
# # # (iap) (ibp) (kap) (kbp) -> (iap) (kap) (ibp) (kbp)
# # U_dag_tensor = U_dag_tensor.transpose([0, 2, 1, 3])
#
# # rho_tdot = dual_obj.rho_init_tensor.tensor
# # rho_tdot = np.tensordot(U_tensor, rho_tdot, axes = ([1, 3], [0, 2]))
# # rho_tdot = rho_tdot.transpose([])
# # rho_tdot = np.tensordot(U_dag_tensor, rho_tdot, axes = ([0, 2], [1, 3]))
# # rho_tdot = dual_obj.tensor_2_mat(rho_tdot)
#
# print(np.linalg.norm(state_dm - rho))

# TEST4: see if the superoperator noise works with just two
# sites

# lattice = graphs.define_lattice(m = 2, n = 3)
#
# gamma0 = np.pi/13
# beta0 = np.pi/19
#
# gamma1 = np.pi/4
# beta1 = 2 * np.pi/3
# p_noise = 0.9
#
# gamma = np.array([gamma0, gamma1])
# beta = np.array([beta0, beta1])
#
# maxcut_obj = problems.MaxCut(lattice, lattice)
# p = 2
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma, beta = beta, p_noise = p_noise)
# state_0 = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)
# state_1 = dual_obj.noise_layer(var_tensor = state_0)
# state_2 = dual_obj.circuit_layer(layer_num = 1, var_tensor = state_1)
# state_3 = dual_obj.noise_layer(var_tensor = state_2)
#
# state_dm = dual_obj.tensor_2_mat(state_3.tensor)
#
# X = []
# Y = []
# Z = []
#
# num_sites = 6
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
# def noise_layer(rho: qutip.Qobj, prob_obj):
#
#     for site in prob_obj.lattice:
#
#         i = prob_obj.site_nums[site]
#
#         rho = (1 - 3 * p_noise/4) * rho +\
#               (p_noise/4) * (X[i] * rho * X[i] + Y[i] * rho * Y[i] + Z[i] * rho * Z[i])
#
#     return rho
#
# psi_after_step = maxcut_obj.init_state()
# psi_after_step = maxcut_obj.Up(gamma[0]) * psi_after_step
# psi_after_step = maxcut_obj.Um(beta[0]) * psi_after_step
#
# rho_after_step = psi_after_step * psi_after_step.dag()
#
# rho_after_step = noise_layer(rho_after_step, maxcut_obj)
#
# rho_after_step = maxcut_obj.Up(gamma[1]) * rho_after_step * maxcut_obj.Up(gamma[1]).dag()
# rho_after_step = maxcut_obj.Um(beta[1]) * rho_after_step * maxcut_obj.Um(beta[1]).dag()
#
# rho_after_step = noise_layer(rho_after_step, maxcut_obj)
#
# rho = rho_after_step.full()
#
# print(np.linalg.norm(state_dm - rho))

# TEST5: check the variable loading operations and entropy bounds

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

# TEST5: check that the objective is real and smaller than the actual solution

lattice = graphs.define_lattice(m = 2, n = 2)
graph = graphs.create_random_connectivity(lattice)
maxcut_obj = problems.MaxCut(lattice, lattice)
p = 1

# gamma0 = np.pi/13
# beta0 = np.pi/19
# gamma1 = np.pi/4
# beta1 = 2 * np.pi/3

# gamma = np.array([gamma0, gamma1])
# beta = np.array([beta0, beta1])

p_actual = np.min(maxcut_obj.H.full())

gamma_init = np.random.uniform(low = 0, high = 2 * np.pi, size = p)
beta_init = np.random.uniform(low = 0, high = np.pi, size = p)
gamma_beta_init = np.concatenate((gamma_init, beta_init))
mc_probs = np.zeros(graph.number_of_nodes())
p_noise = 0
qaoa_obj = algorithms.QAOA(maxcut_obj, p, p_noise, mc_probs)
obj_over_opti, opt_result = algorithms.optimize_QAOA(gamma_beta_init, qaoa_obj)
p_clean = obj_over_opti[-1]
gamma_beta_clean = opt_result.x
gamma_clean = np.split(gamma_beta_clean, 2)[0]
beta_clean = np.split(gamma_beta_clean, 2)[1]

prob_noise = 0.5

# num_monte_carlo = int(1/prob_noise)
num_monte_carlo = 20
p_noisy = 0

for n_mc in range(num_monte_carlo):

    mc_probs = np.random.uniform(low = 0, high = 1, size = (p, graph.number_of_nodes()))
    qaoa_obj_noisy = algorithms.QAOA(maxcut_obj, p, prob_noise, mc_probs, noisy = True)

    p_noisy += algorithms.objective_external_QAOA(
                    gamma_beta_clean, qaoa_obj_noisy)/num_monte_carlo

dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma_clean, beta = beta_clean, p_noise = prob_noise)
# dual_obj_jax = dual_jax.MaxCutDualJAX(prob_obj = maxcut_obj, p = p, gamma = jnp.array(gamma_clean), beta = jnp.array(beta_clean), p_noise = prob_noise)

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


# vars_init = np.random.random(dual_obj.len_vars)
# if vars_init[0] <= 0:
#     vars_init[0] = 1
#
# if vars_init[1] <= 0:
#     vars_init[1] = 1



# dual_obj_jax = dual_jax.MaxCutDualJAX(prob_obj = maxcut_obj, p = p, gamma = jnp.array(gamma_clean), beta = jnp.array(beta_clean), p_noise = prob_noise)

# p_lb_jax = dual_obj_jax.objective(jnp.array(vars_init))
# p_lb_jax = dual_jax.objective_external_dual_JAX(jnp.array(vars_init), dual_obj_jax)

# deriv = grad(dual_obj_jax.objective)(jnp.array(vars_init))
# deriv = dual_jax.gradient_external_dual_JAX(jnp.array(vars_init), dual_obj_jax)

# positions = tuple(np.arange(dual_obj_jax.len_vars, dtype = int))
# positions = tuple(np.arange(100, dtype = int))

# fd_deriv = dual_jax.fd_gradient_dual_JAX(jnp.array(vars_init), positions, dual_obj_jax)

# vars_init = np.random.random(dual_obj.len_vars)

# deriv2 = grad(dual_obj_jax.objective)(jnp.array(vars_init))

sigma_bound = 3e1

lambdas_init = np.ones(p)
# sigma_vars_init = np.random.uniform(low = -sigma_bound, high = sigma_bound, size = dual_obj.len_vars - p)
sigma_vars_init = np.ones(dual_obj.len_vars - p) * (-1e1)
vars_init = np.concatenate((lambdas_init, sigma_vars_init))

# obj_over_opti, opt_result = dual_jax.optimize_external_dual_JAX(vars_init, dual_obj_jax)

# p_lb_jax_opti = -float(obj_over_opti[-1])

# dual_obj_opt = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma_clean, beta = beta_clean, p_noise = prob_noise)
# dual_obj_opt.assemble_vars_into_tensors(opt_result.x)
# sigma0 = dual_obj_opt.tensor_2_mat(dual_obj_opt.Sigmas[0].tensor)
# sigma1 = dual_obj_opt.tensor_2_mat(dual_obj_opt.Sigmas[1].tensor)

# obj_opt = dual_obj_opt.cost()

dual_obj_jax_global = dual_jax.MaxCutDualJAXGlobal(prob_obj = maxcut_obj, p = p, p_noise = prob_noise)
# obj_over_opti_global, opt_result_global = dual_jax.optimize_external_dual_JAX(vars_init, dual_obj_jax_global)

init_obj = dual_jax.unjaxify_obj(dual_jax.objective_external_dual_JAX)(vars_init, dual_obj_jax_global)
init_gradient = dual_jax.unjaxify_grad(dual_jax.gradient_external_dual_JAX)(vars_init, dual_obj_jax_global)

value, state = dual_jax.adam_external_dual_JAX(jnp.array(vars_init), dual_obj_jax_global, alpha = 0.1, num_steps = 100)

# p_lb_jax_opti_global = -float(obj_over_opti_global[-1])

# dual_obj_opt_global = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma_clean, beta = beta_clean, p_noise = prob_noise)
# dual_obj_opt_global.assemble_vars_into_tensors(opt_result_global.x)
# sigma0_global = dual_obj_opt_global.tensor_2_mat(dual_obj_opt_global.Sigmas[0].tensor)
# obj_opt_global = dual_obj_opt_global.cost()

dual_obj_jax_nochannel = dual_jax.MaxCutDualJAXNoChannel(prob_obj = maxcut_obj, p = p, p_noise = prob_noise)

lambdas_init = np.ones(p)
obj_over_opti_nochannel, opt_result_nochannel = dual_jax.optimize_external_dual_JAX(lambdas_init, dual_obj_jax_nochannel)

init_obj = dual_jax.unjaxify_obj(dual_jax.objective_external_dual_JAX)(lambdas_init, dual_obj_jax_nochannel)
init_gradient = dual_jax.unjaxify_grad(dual_jax.gradient_external_dual_JAX)(lambdas_init, dual_obj_jax_nochannel)

p_lb_jax_opti_nochannel = -float(obj_over_opti_nochannel[-1])

lmbda_range = np.linspace(0.01, 10000, 1000)
g_array = []
vars_vec = vars_init.copy()
for l in lmbda_range:
    vars_vec[0] = l
    g_array.append(float(dual_jax.objective_external_dual_JAX(vars_vec, dual_obj_jax_nochannel)))

# dual_obj_opt_nochannel = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma_clean, beta = beta_clean, p_noise = prob_noise)
# dual_obj_opt_nochannel.assemble_vars_into_tensors(opt_result_nochannel.x)
# sigma0_nochannel = dual_obj_opt_nochannel.tensor_2_mat(dual_obj_opt_nochannel.Sigmas[0].tensor)

# g = lambda lmbda, x, S: -lmbda * np.log(np.sum(np.exp(-x/lmbda))) + lmbda * S
# E = np.diag(maxcut_obj.H.full())
# lmbda_range = np.linspace(0.01, 20, 1000)
# g_array = []
#
# for l in lmbda_range:
#     g_array.append(g(l, E, 4 * np.log(2) * 0.1))
#
# g_array = np.array(g_array)
# plt.plot(lmbda_range, g_array)
# plt.show()

# vars = np.arange((1 + dual_obj.num_diag_elements + 2 * dual_obj.num_tri_elements) * p)

# dual_obj.assemble_vars_into_tensors(vars)

# print(dual_obj.entropy_bounds)
