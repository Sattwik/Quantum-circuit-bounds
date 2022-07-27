from datetime import datetime
from datetime import date
import time
import pickle
import sys
import argparse
import io
import copy
import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import jax.numpy as jnp
import jax
import scipy
from jax.config import config
config.update("jax_enable_x64", True)

from fermions import gaussian, fermion_test_utils

parser = argparse.ArgumentParser(description='System size(N), noise probability(p), seed')
parser.add_argument('--N', type = str)
parser.add_argument('--seed', type = str)
parser.add_argument('--p', type = str)
parser.add_argument('--k_dual', type = str)
parser.add_argument('--result_save_path', type = str)
cliargs = parser.parse_args()

key = jax.random.PRNGKey(int(cliargs.seed))
N = int(cliargs.N)

print('N = ', N)
print('seed = ', seed)

if N%2 == 0:
    d = N - 1
else:
    d = N
local_d = 1
k = 1

circ_params = gaussian.PrimalParams(N, d, local_d, key, k = k)
key, subkey = jax.random.split(circ_params.key_after_ham_gen)

final_energy = gaussian.energy_after_circuit(circ_params)
clean_sol = final_energy
print("clean sol = ", clean_sol)

p = float(cliargs.p)
k_dual = int(cliargs.k_dual)
lambda_lower_bounds = 0.1 * jnp.ones(d)
dual_params = gaussian.DualParams(circ_params, p, k_dual, lambda_lower_bounds)

key, subkey = jax.random.split(key)
dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N

num_steps = int(5e3)
dual_obj_over_opti, dual_opt_result = \
    gaussian.optimize(dual_vars_init, dual_params,
                      gaussian.dual_obj, gaussian.dual_grad,
                      num_iters = num_steps)
noisy_bound = -gaussian.dual_obj(jnp.array(dual_opt_result.x), dual_params)
print("noisy bound = ", noisy_bound)

noisy_sol = gaussian.noisy_primal(circ_params, p)
print("noisy sol = ", noisy_sol)
print("noisy bound <= noisy sol? ")
if noisy_bound <= float(np.real(noisy_sol)):
    print("True")
else:
    print("False")

num_steps = int(5e3)
dual_vars_init_nc = jnp.array([0.0])
dual_obj_over_opti_nc, dual_opt_result_nc = \
    gaussian.optimize(dual_vars_init_nc, dual_params,
                      gaussian.dual_obj_no_channel, gaussian.dual_grad_no_channel,
                      num_iters = num_steps)
noisy_bound_nc = -gaussian.dual_obj_no_channel(jnp.array(dual_opt_result_nc.x), dual_params)

print("noisy bound nc = ", noisy_bound_nc)
print("noisy bound nc <= noisy bound? ")
if noisy_bound_nc <= noisy_bound:
    print("True")
else:
    print("False")

data_list = [clean_sol, noisy_sol, noisy_bound, noisy_bound_nc,
             lambda_lower_bounds,
             dual_obj_over_opti, dual_obj_over_opti_nc]

data_file_name = "fermion1D-N-" + str(N) + "-seed-" + cliargs.seed + "-p-" + str(p) + "-kdual-" + \
                 str(k_dual) + ".pkl"

with open(os.path.join(cliargs.result_save_path, data_file_name), "wb") as f_for_pkl:
    pickle.dump(data_list, f_for_pkl)
