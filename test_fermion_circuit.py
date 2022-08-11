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

#------------------------------------------------------------------------------#
#----------------------------------- TESTS ------------------------------------#
#------------------------------------------------------------------------------#

colorama.init()

N = 20
if N%2 == 0:
    d = N - 1
else:
    d = N
# d = 2
local_d = 1
k = 1

rng = np.random.default_rng()
# seed = rng.integers(low=0, high=100, size=1)[0]
# seed = 69
seed = N + 0
key = jax.random.PRNGKey(seed)

circ_params = gaussian.PrimalParams(N, d, local_d, key, k = k)
key, subkey = jax.random.split(circ_params.key_after_ham_gen)

# circ_params.layer_hamiltonians = jnp.zeros(circ_params.layer_hamiltonians.shape)

#------------------------------------------------------------------------------#
#---------------------------------- PARENT H ----------------------------------#
#---------------------------------- CLEAN SOL ---------------------------------#
#------------------------------------------------------------------------------#

w_parent, v_parent = jnp.linalg.eig(1j * circ_params.h_parent)
w_parent = np.real(w_parent)
gs_energy_parent = sum(w_parent[w_parent < 0])

print(colorama.Fore.GREEN + "gs energy = ", gs_energy_parent)
print(colorama.Style.RESET_ALL)

final_energy = gaussian.energy_after_circuit(circ_params)
print(colorama.Fore.GREEN + "circ energy = ", final_energy)
print(colorama.Style.RESET_ALL)

#------------------------------------------------------------------------------#
#---------------------------------- NOISY SOL ---------------------------------#
#------------------------------------------------------------------------------#

p = 0.0

# noisy_sol_full = fermion_test_utils.primal_noisy_circuit_full(dual_params)
noisy_sol = gaussian.noisy_primal(circ_params, p)
# print(colorama.Fore.GREEN + "noisy sol full = ", noisy_sol_full)
print(colorama.Fore.GREEN + "noisy sol = ", noisy_sol)
# print(colorama.Fore.GREEN + "|noisy_sol - noisy_sol_full|= ", np.abs(noisy_sol-noisy_sol_full))
# print(colorama.Fore.GREEN + "noisy bound <= noisy sol? ")
# if noisy_bound <= float(np.real(noisy_sol)):
#     print(colorama.Fore.GREEN + "True")
# else:
#     print(colorama.Fore.RED + "False")
print(colorama.Style.RESET_ALL)

#------------------------------------------------------------------------------#
#---------------------------------- DUAL SETUP --------------------------------#
#------------------------------------------------------------------------------#

k_dual = 1
lambda_lower_bounds = (0.0) * jnp.ones(d)
dual_params = gaussian.DualParams(circ_params, p, k_dual, lambda_lower_bounds)

# # p_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# p_list = [0.0]
# lambda_nc_list = []
# lambda_nc_direct_list = []
#
# for p in p_list:
#     k_dual = 1
#     lambda_lower_bounds = (0.0) * jnp.ones(d)
#     dual_params = gaussian.DualParams(circ_params, p, k_dual, lambda_lower_bounds)
#
# #------------------------------------------------------------------------------#
# #---------------------------------- NO CHANNEL DUAL ---------------------------#
# #------------------------------------------------------------------------------#
#
#     # alpha = 0.01
num_steps = int(5e3)
dual_vars_init_nc = jnp.array([0.0])
# dual_obj_over_opti_nc, dual_opt_result_nc = \
#     gaussian.adam_optimize(gaussian.dual_obj_no_channel, gaussian.dual_grad_no_channel,
#                            dual_vars_init_nc, dual_params,
#                            alpha, num_steps)
# noisy_bound_nc = -gaussian.dual_obj_no_channel(jnp.array(dual_opt_result_nc), dual_params)

dual_obj_over_opti_nc, dual_opt_result_nc = \
    gaussian.optimize(dual_vars_init_nc, dual_params,
                      gaussian.dual_obj_no_channel, gaussian.dual_grad_no_channel,
                      num_iters = num_steps)
noisy_bound_nc = -gaussian.dual_obj_no_channel(jnp.array(dual_opt_result_nc.x), dual_params)
#
#     dual_vars_init_nc_direct = jnp.log(1 + jnp.exp(dual_vars_init_nc))
#     lower_bounds_nc = np.array([0.0])
#     upper_bounds_nc = np.array([np.inf])
#     bnds = scipy.optimize.Bounds(lb = lower_bounds_nc, ub = upper_bounds_nc, keep_feasible = True)
#
#     dual_obj_over_opti_nc_direct, dual_opt_result_nc_direct = \
#         gaussian.optimize(dual_vars_init_nc_direct, dual_params,
#                           gaussian.dual_obj_no_channel_direct_lambda, gaussian.dual_grad_no_channel_direct_lambda,
#                           num_iters = num_steps, bounds = bnds)
#     noisy_bound_nc_direct = -gaussian.dual_obj_no_channel_direct_lambda(jnp.array(dual_opt_result_nc_direct.x), dual_params)
#
# a_range = jnp.linspace(-4, 5, 1000)
# lmbda_range = jnp.log(1 + jnp.exp(a_range))
# a_range = jnp.reshape(a_range, (1000, 1))
# dual_grad_nc_array = vmap(gaussian.dual_grad_no_channel, in_axes = [0, None])(a_range, dual_params)
# dual_obj_nc_array = vmap(gaussian.dual_obj_no_channel, in_axes = [0, None])(a_range, dual_params)
#
# zero_index = jnp.argmin(jnp.abs(dual_grad_nc_array))
# lmbda_grad = lmbda_range.at[zero_index].get()
#
#     # plt.plot(-dual_obj_over_opti_nc)
#     # plt.show()
#
#
#     lambda_nc = jnp.log(1 + jnp.exp(dual_opt_result_nc.x))
#     print('lambda_nc = ', lambda_nc)
#     print('lambda_nc_direct = ', dual_opt_result_nc_direct.x)
#     # print('lambda grad = ', lmbda_grad)
#
#     # plt.plot(lmbda_range, dual_grad_nc_array)
#     # plt.axhline(y = 0, ls = ':', color = 'k')
#     # plt.show()
#
#     lambda_nc_list.append(lambda_nc)
#     lambda_nc_direct_list.append(dual_opt_result_nc_direct.x)
#     # lambda_grad_list.append(lmbda_grad)
#
# plt.plot(p_list, lambda_nc_list, label = 'parametrized')
# plt.plot(p_list, lambda_nc_direct_list, label = 'direct')
# plt.legend()
# plt.yscale('log')
# plt.show()

#------------------------------------------------------------------------------#
#---------------------------------- FULL DUAL ---------------------------------#
#------------------------------------------------------------------------------#

key, subkey = jax.random.split(key)
dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N
# dual_vars_init = jnp.arange(0, dual_params.total_num_dual_vars)

# lambdas, sigmas = gaussian.unvec_and_process_dual_vars(dual_vars_init, dual_params)

# obj_init_fori = gaussian.dual_obj(dual_vars_init, dual_params)
# print(colorama.Fore.GREEN + "new dual = ", obj_init_fori)
# obj_init_full = fermion_test_utils.dual_full(dual_vars_init, dual_params)
# print("full dual = ", obj_init_full)
# print(colorama.Style.RESET_ALL)

alpha = 0.01
num_steps = int(5e3)

#phase 1
# dual_obj_over_opti_adam, dual_opt_result_adam = \
#     gaussian.adam_optimize(gaussian.dual_obj, gaussian.dual_grad,
#                            dual_vars_init, dual_params,
#                            alpha, num_steps)
# noisy_bound_adam_p1 = -gaussian.dual_obj(jnp.array(dual_opt_result_adam), dual_params)

#phase 2
# alpha = 0.001
# dual_vars_init = jnp.array(dual_opt_result_adam)
# dual_obj_over_opti_adam, dual_opt_result_adam = \
#     gaussian.adam_optimize(gaussian.dual_obj, gaussian.dual_grad,
#                            dual_vars_init, dual_params,
#                            alpha, num_steps)
# noisy_bound_adam_p2 = -gaussian.dual_obj(jnp.array(dual_opt_result_adam), dual_params)

# phase 1
dual_obj_over_opti_phase1, dual_opt_result_phase1 = \
    gaussian.optimize(dual_vars_init, dual_params,
                      gaussian.dual_obj, gaussian.dual_grad,
                      num_iters = num_steps, opt_method = "BFGS")
noisy_bound = -gaussian.dual_obj(jnp.array(dual_opt_result_phase1.x), dual_params)

print("noisy bound after 5e3 steps = ", noisy_bound)

# phase 2
dual_vars_init = jnp.array(dual_opt_result_phase1.x)
dual_obj_over_opti_phase2, dual_opt_result_phase2 = \
    gaussian.optimize(dual_vars_init, dual_params,
                      gaussian.dual_obj, gaussian.dual_grad,
                      num_iters = num_steps, opt_method = "BFGS")
noisy_bound = -gaussian.dual_obj(jnp.array(dual_opt_result_phase2.x), dual_params)

print("noisy bound after 10e3 steps = ", noisy_bound)

plt.plot(-dual_obj_over_opti_phase1)
plt.show()

plt.plot(-dual_obj_over_opti_phase2)
plt.show()


# dual_vars_init_direct = dual_vars_init
# dual_vars_init_direct = \
# dual_vars_init_direct.at[:d].set(jnp.log(1 + jnp.exp(dual_vars_init.at[:d].get())))
#
# sigma_bound = np.inf
# lower_bounds = np.array([0.0] * d + [-sigma_bound] * (dual_params.total_num_dual_vars - d))
# # lower_bounds[:d] = 0.0
# upper_bounds = np.array([np.inf] * d + [sigma_bound] * (dual_params.total_num_dual_vars - d))
# # upper_bounds = np.array([np.inf] * dual_params.total_num_dual_vars)
# bnds = scipy.optimize.Bounds(lb = lower_bounds, ub = upper_bounds, keep_feasible = True)
#
# dual_obj_over_opti_direct, dual_opt_result_direct = \
#     gaussian.optimize(dual_vars_init_direct, dual_params,
#                       gaussian.dual_obj_direct_lambda, gaussian.dual_grad_direct_lambda,
#                       num_iters = num_steps, bounds = bnds)
# noisy_bound_direct = -gaussian.dual_obj_direct_lambda(jnp.array(dual_opt_result_direct.x), dual_params)

# dual_vars_opt_result = jnp.array(dual_opt_result.x)
# dual_vars_opt_result = dual_vars_opt_result.at[:d].set(jnp.log(1 + jnp.exp(dual_vars_opt_result.at[:d].get())))
#
# noisy_bound_direct = -gaussian.dual_obj_direct_lambda(dual_vars_opt_result, dual_params)

print(colorama.Fore.GREEN + "noisy bound = ", noisy_bound)
print(colorama.Fore.GREEN + "noisy bound nc = ", noisy_bound_nc)
print(colorama.Fore.GREEN + "noisy bound nc <= noisy bound? ")
if noisy_bound_nc <= noisy_bound:
    print(colorama.Fore.GREEN + "True")
else:
    print(colorama.Fore.RED + "False")
print(colorama.Style.RESET_ALL)

print("lambdas = ", lambda_lower_bounds + dual_vars_opt_result[:d])


# plt.plot(-dual_obj_over_opti_direct)
# plt.show()

# print("Computing grads...")
#
# fd_grad = gaussian.fd_dual_grad_direct_lambda(jnp.array(dual_opt_result_direct.x), dual_params)
# autodiff_grad = gaussian.dual_grad_direct_lambda(jnp.array(dual_opt_result_direct.x), dual_params)
#
# print("Grad diff = ", jnp.linalg.norm(fd_grad - autodiff_grad))

# print(colorama.Fore.GREEN + "noisy bound = ", noisy_bound)
# print(colorama.Fore.GREEN + "noisy bound <= clean sol? ")
# clean_sol = gs_energy_parent
# if noisy_bound <= clean_sol:
#     print(colorama.Fore.GREEN + "True")
# else:
#     print(colorama.Fore.RED + "False")
# print(colorama.Style.RESET_ALL)
#
# noisy_bound_full = -fermion_test_utils.dual_full(jnp.array(dual_opt_result), dual_params)
# noisy_bound_full = -fermion_test_utils.dual_full(jnp.array(dual_opt_result.x), dual_params)
# print(colorama.Fore.GREEN + "noisy bound full = ", noisy_bound_full)
# print(colorama.Fore.GREEN + "noisy bound - noisy bound full = ", noisy_bound - noisy_bound_full)
# print(colorama.Style.RESET_ALL)


#-------------------- old stuff -----------------------#


# sigma_bound = 10.0
# lambda_bound = 100.0
# # dual_vars_init = dual_vars_inits.at[:d].set()
#
# # fd_grad = gaussian.fd_dual_grad(dual_vars_init, dual_params)
#



# #
# # start = time.time()
# # grad_init = gaussian.dual_grad(dual_vars_init, dual_params)
# # end = time.time()
# # print("grad init = ", grad_init)
# # print("grad exec time (s) = ", end - start)
#
# dual_bnds = scipy.optimize.Bounds(lb = [-lambda_bound] * d + [-sigma_bound] * (dual_params.total_num_dual_vars - d),
#                                   ub = [lambda_bound]  * d + [sigma_bound]  * (dual_params.total_num_dual_vars - d))
#
# dual_obj_over_opti, dual_opt_result = gaussian.optimize_dual(dual_vars_init, dual_params, dual_bnds)
#
# noisy_bound = -dual_obj_over_opti[-1]/scale
# print(colorama.Fore.GREEN + "noisy bound = ", noisy_bound)
# print(colorama.Style.RESET_ALL)
#
# print(colorama.Fore.GREEN + "noisy bound <= clean sol? ")
# if noisy_bound <= clean_sol:
#     print(colorama.Fore.GREEN + "True")
# else:
#     print(colorama.Fore.RED + "False")
# print(colorama.Style.RESET_ALL)
#
# # noisy_sol = fermion_test_utils.primal_noisy_circuit_full(dual_params)
# # print(colorama.Fore.GREEN + "noisy sol = ", noisy_sol)
# # print(colorama.Style.RESET_ALL)
#
# # print(colorama.Fore.GREEN + "noisy bound <= noisy sol? ")
# # if noisy_bound <= noisy_sol:
# #     print(colorama.Fore.GREEN + "True")
# # else:
# #     print(colorama.Fore.RED + "False")
# # print(colorama.Style.RESET_ALL)
#
# # plt.plot(dual_obj_over_opti)
# # plt.show()
#

# #

lambdas, sigmas = gaussian.unvec_and_process_dual_vars(jnp.array(dual_opt_result.x), dual_params)
layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
h_parent = dual_params.circ_params.h_parent
hi = sigmas.at[:,:,i].get() - \
     gaussian.noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(),
                      sigmas.at[:,:,i+1].get(), p)
gaussian.trace_fgstate(-hi/lambdas.at[i].get())


colorama.deinit()


# plt.plot(obj_over_opti)
# plt.show()
