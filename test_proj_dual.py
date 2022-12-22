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
from matplotlib import rc

from fermions import gaussian, fermion_test_utils

#------------------------------------------------------------------------------#
#----------------------------------- TESTS ------------------------------------#
#------------------------------------------------------------------------------#

colorama.init()

N = 16
if N%2 == 0:
    d = N - 1
else:
    d = N

# d = 5
local_d = 1
k = 1

rng = np.random.default_rng()
seed = N + 1
key = jax.random.PRNGKey(seed)

circ_params = gaussian.PrimalParams(N, d, local_d, key, k = k)
key, subkey = jax.random.split(circ_params.key_after_ham_gen)


#------------------------------------------------------------------------------#
#---------------------------------- PARENT H ----------------------------------#
#---------------------------------- CLEAN SOL ---------------------------------#
#------------------------------------------------------------------------------#
#
# w_parent, v_parent = jnp.linalg.eig(1j * circ_params.h_parent)
# w_parent = np.real(w_parent)
# gs_energy_parent = sum(w_parent[w_parent < 0])
#
# print(colorama.Fore.GREEN + "gs energy = ", gs_energy_parent)
# print(colorama.Style.RESET_ALL)
#
# final_energy = gaussian.energy_after_circuit(circ_params)
# print(colorama.Fore.GREEN + "circ energy = ", final_energy)
# print(colorama.Style.RESET_ALL)
#
# #------------------------------------------------------------------------------#
# #---------------------------------- NOISY SOL ---------------------------------#
# #------------------------------------------------------------------------------#
#
# p = 0.1
#
# noisy_sol = gaussian.noisy_primal(circ_params, p)
#
# print(colorama.Fore.GREEN + "noisy sol = ", noisy_sol)
# print(colorama.Style.RESET_ALL)
#
# #------------------------------------------------------------------------------#
# #---------------------------------- DUAL SETUP --------------------------------#
# #------------------------------------------------------------------------------#
#
# k_dual = 1
# lambda_lower_bounds = (0.0) * jnp.ones(d)
# dual_params = gaussian.DualParams(circ_params, p, k_dual, lambda_lower_bounds)
# gaussian.set_all_sigmas(dual_params)
#
# #------------------------------------------------------------------------------#
# #---------------------------------- NO CHANNEL DUAL ---------------------------#
# #------------------------------------------------------------------------------#
#
# num_steps = int(1e3)
# dual_vars_init_nc = jnp.array([0.0])
#
#
# dual_obj_over_opti_nc, dual_opt_result_nc = \
#     gaussian.optimize(dual_vars_init_nc, dual_params,
#                       gaussian.dual_obj_no_channel, gaussian.dual_grad_no_channel,
#                       num_iters = num_steps)
# noisy_bound_nc = -gaussian.dual_obj_no_channel(jnp.array(dual_opt_result_nc.x), dual_params)
#
#
# #------------------------------------------------------------------------------#
# #---------------------------------- PROJ DUAL ---------------------------------#
# #------------------------------------------------------------------------------#
#
# key, subkey = jax.random.split(key)
# dual_vars_init = jnp.zeros((d,))
#
# alpha = 0.01
# num_steps = int(1e3)
#
# dual_obj_over_opti_proj, dual_opt_result_proj = \
#     gaussian.optimize(dual_vars_init, dual_params,
#                       gaussian.dual_obj_projected, gaussian.dual_grad_projected,
#                       num_iters = num_steps)
# noisy_bound_proj = -gaussian.dual_obj_projected(jnp.array(dual_opt_result_proj.x), dual_params)
#
# # plt.plot(-dual_obj_over_opti_proj)
# # plt.show()
#
# #------------------------------------------------------------------------------#
# #---------------------------------- FULL DUAL ---------------------------------#
# #------------------------------------------------------------------------------#
#
# key, subkey = jax.random.split(key)
# dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N
#
# alpha = 0.01
# num_steps = int(1e3)
#
# dual_obj_over_opti_phase1, dual_opt_result_phase1 = \
#     gaussian.optimize(dual_vars_init, dual_params,
#                       gaussian.dual_obj, gaussian.dual_grad,
#                       num_iters = num_steps)
# noisy_bound = -gaussian.dual_obj(jnp.array(dual_opt_result_phase1.x), dual_params)
#
# # plt.plot(-dual_obj_over_opti_phase1)
# # plt.show()
#
# print(colorama.Fore.GREEN + "sol AR = ", noisy_sol/final_energy)
# print(colorama.Fore.GREEN + "bound AR = ", noisy_bound/final_energy)
# print(colorama.Fore.GREEN + "proj bound AR = ", noisy_bound_proj/final_energy)
# print(colorama.Fore.GREEN + "nc AR = ", noisy_bound_nc/final_energy)
#
# print(colorama.Style.RESET_ALL)
#
# colorama.deinit()
