
from functools import partial

import pickle
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import jax.numpy as jnp
from jax.config import config
from jax import jit, grad, vmap, value_and_grad
config.update("jax_enable_x64", True)

from fermions import gaussian

class NCDualParams():
    def __init__(self, N: int, p: float, d: int):
        self.N = N
        self.p = p
        self.d = d
        q = 1-p

        self.Sd = N * np.log(2) * (1 - q ** d)

        k = 4
        f1 = 2
        self.lambda_lower_bound = 8 * np.exp(3) * k * f1
        # self.lambda_lower_bound = 0

@partial(jit, static_argnums = (1,))
def dual_obj_nc(dual_vars: jnp.array, dual_params: NCDualParams):
    a = dual_vars.at[0].get()

    lmbda = dual_params.lambda_lower_bound + jnp.exp(a)

    N = dual_params.N
    p = dual_params.p
    d = dual_params.d
    Sd = dual_params.Sd

    cost = -lmbda * N * jnp.log(jnp.exp(-1/lmbda) + jnp.exp(1/lmbda)) + lmbda * Sd

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad_nc(dual_vars: np.array, dual_params: NCDualParams):
    return grad(dual_obj_nc, argnums = 0)(dual_vars, dual_params)

N = 32
# p_list = np.linspace(0.03, 0.3, 5)
# p_list = [0.03, 0.05]
# d_list = np.concatenate((np.array(np.linspace(4, 24, 11), dtype = int), np.array(np.linspace(24, 240, 11), dtype = int)))

p_list = np.linspace(0.03, 0.3, 10)
d_list = np.array(np.linspace(4, 24, 11), dtype = int)

depth_list = [1 + 2 * d for d in d_list]

data_path = "../vqa_data/0510/20230510-145039/"

clean_sol = -N
norm = 2 * N

for p in p_list:
    entropic_bounds_new = []
    for depth in depth_list:
        dual_params = NCDualParams(N, p, int(depth))
        dual_vars_init = jnp.zeros((1,))

        num_steps = int(5e3)
        dual_obj_over_opti, dual_opt_result = \
            gaussian.optimize(dual_vars_init, dual_params,
                            dual_obj_nc, dual_grad_nc,
                            num_iters = num_steps)
        noisy_bound_nc = -dual_obj_nc(np.array(dual_opt_result.x), dual_params)
        entropic_bounds_new.append(noisy_bound_nc)

    fname = "entropic_bound_noise_bounded_temp_" + str(p) + ".npy"
    with open(os.path.join(data_path, fname), 'wb') as result_file:
        np.save(result_file, np.array(entropic_bounds_new))

# entropic_bounds_scaled = (np.array(entropic_bounds_new) - clean_sol)/norm

# for p in p_list:
#     data_path = "./../vqa_data/results_sattwik/results_quantumHamiltonian_cnot/"
#     fname = 'res.pkl'

#     fname = "entropic_bound_noise" + str(p) + ".npy"
#     with open(os.path.join(data_path, fname), 'rb') as result_file:
#         eb_array = np.load(result_file)

#     d_list = [2 + 2 * eb_tuple[0] for eb_tuple in eb_array]
#     entropic_bounds_old = eb_array[:, 1]

#     entropic_bounds_new = []

#     for d in d_list:
#         dual_params = NCDualParams(N, p, int(d))
#         dual_vars_init = jnp.zeros((1,))

#         num_steps = int(5e3)
#         dual_obj_over_opti, dual_opt_result = \
#             gaussian.optimize(dual_vars_init, dual_params,
#                             dual_obj_nc, dual_grad_nc,
#                             num_iters = num_steps)
#         noisy_bound_nc = -dual_obj_nc(np.array(dual_opt_result.x), dual_params)
#         entropic_bounds_new.append(noisy_bound_nc)

#     fname = "entropic_bound_noise_bounded_temp_" + str(p) + ".npy"
#     with open(os.path.join(data_path, fname), 'wb') as result_file:
#         np.save(result_file, np.array(entropic_bounds_new))


# d = d_list[0]

# dual_params = NCDualParams(N, p, int(d))
# dual_vars_init = jnp.zeros((1,))

# num_steps = int(5e3)
# dual_obj_over_opti, dual_opt_result = \
#     gaussian.optimize(dual_vars_init, dual_params,
#                         dual_obj_nc, dual_grad_nc,
#                         num_iters = num_steps)
# noisy_bound_nc = -dual_obj_nc(np.array(dual_opt_result.x), dual_params)
# entropic_bounds_new.append(noisy_bound_nc)




