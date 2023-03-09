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

N = 10
d = 2 * 1
local_d = 2 * 1
k = 1

rng = np.random.default_rng()
# seed = rng.integers(low=0, high=100, size=1)[0]
# seed = 69
seed = N + 9
key = jax.random.PRNGKey(seed)

print(key)

circ_params = gaussian.PrimalParams(N, d, local_d, key, k = k, mode = "NN")
key, subkey = jax.random.split(circ_params.key_after_ham_gen)

print(key)

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

p = 0.05
noisy_sol = gaussian.noisy_primal(circ_params, p)
print(colorama.Fore.GREEN + "noisy sol = ", noisy_sol)
print(colorama.Style.RESET_ALL)

#------------------------------------------------------------------------------#
#---------------------------------- DUAL SETUP --------------------------------#
#------------------------------------------------------------------------------#

k_dual = N

lambda_lower_bounds = (0.0) * jnp.ones(d)
dual_params = gaussian.DualParams(circ_params, p, k_dual, lambda_lower_bounds)

#------------------------------------------------------------------------------#
#---------------------------------- NO CHANNEL DUAL ---------------------------#
#------------------------------------------------------------------------------#

num_steps = int(5e3)
dual_vars_init_nc = jnp.array([0.0])

dual_obj_over_opti_nc, dual_opt_result_nc = \
    gaussian.optimize(dual_vars_init_nc, dual_params,
                    gaussian.dual_obj_no_channel, gaussian.dual_grad_no_channel,
                    num_iters = num_steps)
noisy_bound_nc = -gaussian.dual_obj_no_channel(jnp.array(dual_opt_result_nc.x), dual_params)

#------------------------------------------------------------------------------#
#---------------------------------- FULL DUAL ---------------------------------#
#------------------------------------------------------------------------------#


gaussian.set_all_sigmas(dual_params)
proj_sigmas_vec = gaussian.sigmas_to_vec(dual_params.sigmas_proj, dual_params)

dual_vars_init = jnp.zeros((dual_params.total_num_dual_vars,))
dual_vars_init = dual_vars_init.at[d:].set(proj_sigmas_vec)

# lmbda_list = np.logspace(-7, 0, 100)
# nb_list = []

# for lmbda in lmbda_list:
#     dual_vars = jnp.zeros((dual_params.total_num_dual_vars,))
#     dual_vars = dual_vars.at[d:].set(proj_sigmas_vec)
#     dual_vars = dual_vars.at[:d].set(lmbda)
#     nb_list.append(-gaussian.dual_obj(dual_vars, dual_params))

# plt.plot(lmbda_list, nb_list)
# plt.xscale('log')
# plt.show()


# key, subkey = jax.random.split(key)
# print(key)
# dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N

# dual_vars_init = jnp.arange(0, dual_params.total_num_dual_vars)

# lambdas, sigmas = gaussian.unvec_and_process_dual_vars(dual_vars_init, dual_params)

# alpha = 0.01
# num_steps = int(5e3)

dual_obj_over_opti_phase1, dual_opt_result_phase1 = \
    gaussian.optimize(dual_vars_init, dual_params,
                    gaussian.dual_obj, gaussian.dual_grad,
                    num_iters = num_steps)
noisy_bound = -gaussian.dual_obj(jnp.array(dual_opt_result_phase1.x), dual_params)


#------------------------------------------------------------------------------#
#--------------------------------- PURITY DUAL --------------------------------#
#------------------------------------------------------------------------------#

lambda_lower_bounds_purity = jnp.array([0.0])
dual_params_purity = gaussian.DualParamsPurity(circ_params, p, k_dual, lambda_lower_bounds_purity)

# alpha = 0.01
# num_steps = int(5e3)

# dual_vars_init_purity = 1e-9 * jnp.ones((dual_params_purity.total_num_dual_vars,))
# dual_vars_init_purity = dual_vars_init_purity.at[0].set(dual_opt_result_nc.x[0])

# key, subkey = jax.random.split(key)
# dual_vars_init_purity = jax.random.uniform(key, shape = (dual_params_purity.total_num_dual_vars,))/N

dual_vars_init_purity = jnp.zeros((dual_params_purity.total_num_dual_vars,))
dual_vars_init_purity = dual_vars_init_purity.at[1:].set(proj_sigmas_vec)

# init_obj = gaussian.dual_obj_purity(dual_vars_init_purity, dual_params_purity)

# lmbda_list = np.logspace(-7, 0, 100)
# nb_pur_list = []

# for lmbda in lmbda_list:
#     dual_vars_purity = jnp.zeros((dual_params_purity.total_num_dual_vars,))
#     dual_vars_purity = dual_vars_purity.at[1:].set(proj_sigmas_vec)
#     dual_vars_purity = dual_vars_purity.at[:1].set(lmbda)
#     nb_pur_list.append(-gaussian.dual_obj_purity(dual_vars_purity, dual_params_purity))

# plt.plot(lmbda_list, nb_pur_list)
# plt.xscale('log')
# plt.show()


dual_obj_over_opti_purity, dual_opt_result_purity = \
    gaussian.optimize(dual_vars_init_purity, dual_params_purity,
                    gaussian.dual_obj_purity, gaussian.dual_grad_purity,
                    num_iters = num_steps, tol_scale = 1e-7)
noisy_bound_purity = -gaussian.dual_obj_purity(jnp.array(dual_opt_result_purity.x), dual_params_purity)


print(colorama.Fore.GREEN + "noisy bound = ", noisy_bound)
print(colorama.Fore.GREEN + "noisy bound purity = ", noisy_bound_purity)
print(colorama.Fore.GREEN + "noisy bound nc = ", noisy_bound_nc)
print(colorama.Fore.GREEN + "noisy bound nc <= noisy bound? ")
if noisy_bound_nc <= noisy_bound:
    print(colorama.Fore.GREEN + "True")
else:
    print(colorama.Fore.RED + "False")
print(colorama.Fore.GREEN + "noisy bound nc <= noisy bound purity? ")
if noisy_bound_nc <= noisy_bound_purity:
    print(colorama.Fore.GREEN + "True")
else:
    print(colorama.Fore.RED + "False")
print(colorama.Style.RESET_ALL)

# lambdas, sigmas = gaussian.unvec_and_process_dual_vars(jnp.array(dual_opt_result_phase1.x), dual_params)

# print("lambdas = ", lambdas)


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
#
# lambdas, sigmas = gaussian.unvec_and_process_dual_vars(jnp.array(dual_opt_result.x), dual_params)
# layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
# h_parent = dual_params.circ_params.h_parent
# hi = sigmas.at[:,:,i].get() - \
#      gaussian.noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(),
#                       sigmas.at[:,:,i+1].get(), p)
# gaussian.trace_fgstate(-hi/lambdas.at[i].get())


colorama.deinit()


# plt.plot(obj_over_opti)
# plt.show()
