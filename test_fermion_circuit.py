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

rng = np.random.default_rng()
seed = rng.integers(low=0, high=100, size=1)[0]
key = jax.random.PRNGKey(seed)

N = 25
d = N
local_d = 7
k = 1

circ_params = gaussian.PrimalParams(N, d, local_d, key, k = k)
key, subkey = jax.random.split(circ_params.key_after_ham_gen)

w_parent, v_parent = jnp.linalg.eig(1j * circ_params.h_parent)
w_parent = np.real(w_parent)
gs_energy_parent = sum(w_parent[w_parent < 0])

print(colorama.Fore.GREEN + "gs energy = ", gs_energy_parent)
print(colorama.Style.RESET_ALL)

final_energy = gaussian.energy_after_circuit(circ_params)
print(colorama.Fore.GREEN + "circ energy = ", final_energy)
print(colorama.Style.RESET_ALL)

p = 0.001
k_dual = 1
dual_params = gaussian.DualParams(circ_params, p, k_dual)

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
num_steps = 750
dual_obj_over_opti_adam, dual_opt_result_adam = \
    gaussian.adam_optimize_dual(dual_vars_init, dual_params,
                               alpha, num_steps)

adam_noisy_bound = -gaussian.dual_obj(jnp.array(dual_opt_result_adam), dual_params)

plt.plot(-dual_obj_over_opti_adam)
plt.show()

print(colorama.Fore.GREEN + "adam noisy bound = ", adam_noisy_bound)

print(colorama.Fore.GREEN + "(adam) noisy bound <= clean sol? ")
clean_sol = gs_energy_parent
if adam_noisy_bound <= clean_sol:
    print(colorama.Fore.GREEN + "True")
else:
    print(colorama.Fore.RED + "False")
print(colorama.Style.RESET_ALL)

# noisy_sol = fermion_test_utils.primal_noisy_circuit_full(dual_params)
# print(colorama.Fore.GREEN + "adam noisy bound <= noisy sol? ")
# if adam_noisy_bound <= float(np.real(noisy_sol)):
#     print(colorama.Fore.GREEN + "True")
# else:
#     print(colorama.Fore.RED + "False")
# print(colorama.Style.RESET_ALL)

# noisy_bound_full = -fermion_test_utils.dual_full(jnp.array(dual_opt_result_adam), dual_params)
# print(colorama.Fore.GREEN + "noisy bound full = ", noisy_bound_full)
# print(colorama.Style.RESET_ALL)
#
# print(colorama.Fore.GREEN + "adam noisy bound - noisy bound full = ", adam_noisy_bound - noisy_bound_full)
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


colorama.deinit()


# plt.plot(obj_over_opti)
# plt.show()
