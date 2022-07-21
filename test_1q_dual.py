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
import colorama

from fermions import gaussian, fermion_test_utils

#------- testing analysis ---------#

# def S2(p):
#     return -p * np.log(p) - (1-p) * np.log(1-p)
#
# def g(lmbda, sigma, p, eps = 1):
#     g = sigma * (1-p) \
#         - lmbda * np.log(np.exp(-(sigma + eps)/lmbda) + np.exp(sigma/lmbda)) \
#         + lmbda * S2(p/2)
#     return g
#
# def sigma_star(lmbda, p, eps):
#     return lmbda * np.log(np.sqrt((2-p)/p) * np.exp(-eps/(2 * lmbda)))
#
# def g_full(lmbda, p, eps):
#     s_star = sigma_star(lmbda, p, eps)
#     return g(lmbda, s_star, p, eps)
#
# def g_nc(lmbda, p, eps):
#     return g(lmbda, 0, p, eps)
#
# eps = 1
# p = 0.999
# lmbda_range = np.linspace(0.01, 500.0, 1000)
#
# primal = p * eps/2
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.plot(lmbda_range, g_full(lmbda_range, p, eps), label = 'Full')
# ax.plot(lmbda_range, g_nc(lmbda_range, p, eps), ls = '--', label = 'NC')
# ax.axhline(y = primal, ls = '-', color = 'k', label = 'Primal')
# ax.legend()
# plt.show()
#
# g_nc_max = np.max(g_nc(lmbda_range, p, eps))
# print("primal = ", primal)
# print("g_nc max = ", g_nc_max)
# print("primal - g_nc_max = ", primal - g_nc_max)

#------- testing analysis ---------#
rng = np.random.default_rng()
seed = rng.integers(low=0, high=100, size=1)[0]
key = jax.random.PRNGKey(seed)

N = 1
d = 1
local_d = 1
k = 1

circ_params = gaussian.PrimalParams(N, d, local_d, key, k = k)
key, subkey = jax.random.split(circ_params.key_after_ham_gen)

#-------- no unitary test -- passed ---------#
# circ_params.layer_hamiltonians = jnp.zeros(circ_params.layer_hamiltonians.shape)
# Ome = gaussian.Omega(N)
#
# eps = 2.8
# D = jnp.array([[0,0], [0, eps]])
#
# circ_params.h_parent = -1j * jnp.matmul(Ome, jnp.matmul(D, Ome.conj().T))
#
# p = 0.3
# k_dual = 1
# dual_params = gaussian.DualParams(circ_params, p, k_dual)
#
# num_steps = int(5e3)
# key, subkey = jax.random.split(key)
# dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N
# dual_obj_over_opti, dual_opt_result = \
#     gaussian.optimize(dual_vars_init, dual_params,
#                       gaussian.dual_obj_1q_test, gaussian.dual_grad_1q_test,
#                       num_iters = num_steps)
# noisy_bound = -gaussian.dual_obj_1q_test(jnp.array(dual_opt_result.x), dual_params)
#
# dual_vars_init_nc = jnp.array([0.0])
# dual_obj_over_opti_nc, dual_opt_result_nc = \
#     gaussian.optimize(dual_vars_init_nc, dual_params,
#                       gaussian.dual_obj_no_channel_1q_test, gaussian.dual_grad_no_channel_1q_test,
#                       num_iters = num_steps)
# noisy_bound_nc = -gaussian.dual_obj_no_channel_1q_test(jnp.array(dual_opt_result_nc.x), dual_params)

#-------- unitary test -- passed ---------#
# Ome = gaussian.Omega(N)
#
# H = np.pi/2 * (1/np.sqrt(2)) * jnp.array([[1, 1],[1, -1]])
# circ_params.layer_hamiltonians = jnp.reshape(-1j * jnp.matmul(Ome, jnp.matmul(H, Ome.conj().T)), (1,2,2))
#
# eps = 2.8
# D = jnp.array([[0,0], [0, eps]])
#
# circ_params.h_parent = -1j * jnp.matmul(Ome, jnp.matmul(D, Ome.conj().T))
#
# p = 0.3
# k_dual = 1
# dual_params = gaussian.DualParams(circ_params, p, k_dual)
#
# num_steps = int(5e3)
# key, subkey = jax.random.split(key)
# dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N
# dual_obj_over_opti, dual_opt_result = \
#     gaussian.optimize(dual_vars_init, dual_params,
#                       gaussian.dual_obj_1q_test, gaussian.dual_grad_1q_test,
#                       num_iters = num_steps)
# noisy_bound = -gaussian.dual_obj_1q_test(jnp.array(dual_opt_result.x), dual_params)
#
# dual_vars_init_nc = jnp.array([0.0])
# dual_obj_over_opti_nc, dual_opt_result_nc = \
#     gaussian.optimize(dual_vars_init_nc, dual_params,
#                       gaussian.dual_obj_no_channel_1q_test, gaussian.dual_grad_no_channel_1q_test,
#                       num_iters = num_steps)
# noisy_bound_nc = -gaussian.dual_obj_no_channel_1q_test(jnp.array(dual_opt_result_nc.x), dual_params)

#-------- studying dual variables ---------#

N = 3
d = 3
local_d = 3
k = 1

circ_params = gaussian.PrimalParams(N, d, local_d, key, k = k)
key, subkey = jax.random.split(circ_params.key_after_ham_gen)

p = 0.9
k_dual = 1
dual_params = gaussian.DualParams(circ_params, p, k_dual)

num_steps = int(5e3)
key, subkey = jax.random.split(key)
dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N

num_steps = int(5e3)
key, subkey = jax.random.split(key)
dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N
dual_obj_over_opti, dual_opt_result = \
    gaussian.optimize(dual_vars_init, dual_params,
                      gaussian.dual_obj, gaussian.dual_grad,
                      num_iters = num_steps)
noisy_bound = -gaussian.dual_obj(jnp.array(dual_opt_result.x), dual_params)

dual_vars_init_nc = jnp.array([0.0])
dual_obj_over_opti_nc, dual_opt_result_nc = \
    gaussian.optimize(dual_vars_init_nc, dual_params,
                      gaussian.dual_obj_no_channel, gaussian.dual_grad_no_channel,
                      num_iters = num_steps)
noisy_bound_nc = -gaussian.dual_obj_no_channel(jnp.array(dual_opt_result_nc.x), dual_params)

print(noisy_bound_nc - noisy_bound)

N = dual_params.circ_params.N
d = dual_params.circ_params.d
h_parent = dual_params.circ_params.h_parent
layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
p = dual_params.p
Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init

lambdas, sigmas = gaussian.unvec_and_process_dual_vars(jnp.array(dual_opt_result.x), dual_params)

cost = 0

i = 0
hi = sigmas.at[:,:,i].get() - \
     gaussian.noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(), sigmas.at[:,:,i+1].get(), p)
cost += -lambdas.at[i].get() * jnp.log(gaussian.trace_fgstate(-hi/lambdas.at[i].get()))
gaussian.trace_fgstate(-hi/lambdas.at[i].get())
# i = d-1
hi = h_parent + sigmas.at[:,:,d-1].get()
cost += -lambdas.at[d-1].get() * jnp.log(gaussian.trace_fgstate(-hi/lambdas[d-1]))

epsilon_1_dag_sigma1 = \
gaussian.noisy_dual_layer(layer_hamiltonians.at[0, :, :].get(), sigmas.at[:,:,0].get(), p)
cost += -gaussian.energy(Gamma_mjr_init, epsilon_1_dag_sigma1)

q = 1 - p
q_powers = jnp.array([q**i for i in range(d)])

entropy_bounds = N * p * jnp.log(2) * \
         jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])

cost += jnp.dot(lambdas, entropy_bounds)
