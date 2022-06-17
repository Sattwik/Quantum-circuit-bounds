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

N = 100
d = 5
seed = rng.integers(low=0, high=100, size=1)[0]
key = jax.random.PRNGKey(seed)

circ_params = gaussian.PrimalParams(N, d, key)

key, subkey = jax.random.split(circ_params.key_after_ham_gen)
theta_init = jax.random.uniform(key, shape = (d,), minval = 0.0, maxval = 2 * np.pi)

# objective = gaussian.circ_obj(theta_init, params)

circ_obj_over_opti, circ_opt_result = gaussian.optimize_circuit(theta_init, circ_params)

w_target, v_target = jnp.linalg.eig(1j * circ_params.h_target)
w_target = np.real(w_target)
gs_energy_target = sum(w_target[w_target < 0])

actual_sol = gs_energy_target
print(colorama.Fore.GREEN + "gs energy = ", actual_sol)
print(colorama.Style.RESET_ALL)

clean_sol = circ_obj_over_opti[-1]
print(colorama.Fore.GREEN + "clean sol = ", clean_sol)
print(colorama.Style.RESET_ALL)

p = 0.001
scale = 1
dual_params = gaussian.DualParams(circ_params, jnp.array(circ_opt_result.x), p,
                                  scale = scale)

key, subkey = jax.random.split(key)
# dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N
sigma_bound = 10.0
lambda_bound = 100.0
dual_vars_init = jnp.ones((dual_params.total_num_dual_vars,))
# dual_vars_init = dual_vars_inits.at[:d].set()

# fd_grad = gaussian.fd_dual_grad(dual_vars_init, dual_params)

# obj_init = gaussian.dual_obj(dual_vars_init, dual_params)
# print("obj init = ", obj_init)
#
# start = time.time()
# grad_init = gaussian.dual_grad(dual_vars_init, dual_params)
# end = time.time()
# print("grad init = ", grad_init)
# print("grad exec time (s) = ", end - start)

dual_bnds = scipy.optimize.Bounds(lb = [-lambda_bound] * d + [-sigma_bound] * (dual_params.total_num_dual_vars - d),
                                  ub = [lambda_bound]  * d + [sigma_bound]  * (dual_params.total_num_dual_vars - d))

dual_obj_over_opti, dual_opt_result = gaussian.optimize_dual(dual_vars_init, dual_params, dual_bnds)

noisy_bound = -dual_obj_over_opti[-1]/scale
print(colorama.Fore.GREEN + "noisy bound = ", noisy_bound)
print(colorama.Style.RESET_ALL)

print(colorama.Fore.GREEN + "noisy bound <= clean sol? ")
if noisy_bound <= clean_sol:
    print(colorama.Fore.GREEN + "True")
else:
    print(colorama.Fore.RED + "False")
print(colorama.Style.RESET_ALL)

# noisy_sol = fermion_test_utils.primal_noisy_circuit_full(dual_params)
# print(colorama.Fore.GREEN + "noisy sol = ", noisy_sol)
# print(colorama.Style.RESET_ALL)

# print(colorama.Fore.GREEN + "noisy bound <= noisy sol? ")
# if noisy_bound <= noisy_sol:
#     print(colorama.Fore.GREEN + "True")
# else:
#     print(colorama.Fore.RED + "False")
# print(colorama.Style.RESET_ALL)

# plt.plot(dual_obj_over_opti)
# plt.show()

# alpha = 0.01
# num_steps = 500
#
# dual_obj_over_opti_adam, dual_opt_result_adam = \
#     gaussian.adam_optimize_dual(dual_vars_init, dual_params,
#                                alpha, num_steps)
#
# adam_noisy_bound = -float(dual_obj_over_opti_adam[-1])
#
# plt.plot(dual_obj_over_opti_adam)
# plt.show()
#
# print(colorama.Fore.GREEN + "adam noisy bound = ", adam_noisy_bound)
#
# print(colorama.Fore.GREEN + "(adam) noisy bound <= clean sol? ")
# if adam_noisy_bound <= clean_sol:
#     print(colorama.Fore.GREEN + "True")
# else:
#     print(colorama.Fore.RED + "False")
# print(colorama.Style.RESET_ALL)

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

# plt.plot(dual_obj_over_opti)
# plt.show()

colorama.deinit()


# plt.plot(obj_over_opti)
# plt.show()
