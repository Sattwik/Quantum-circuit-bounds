from datetime import datetime
from datetime import date
import argparse
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import jax.numpy as jnp
import jax
from jax.config import config
config.update("jax_enable_x64", True)

from fermions import gaussian2D

parser = argparse.ArgumentParser(description='System size(N), noise probability(p), seed')
parser.add_argument('--N', type = str)
parser.add_argument('--d', type = str)
parser.add_argument('--seed', type = str)
parser.add_argument('--p', type = str)
parser.add_argument('--k_dual', type = str)
parser.add_argument('--result_save_path', type = str)
cliargs = parser.parse_args()

key = jax.random.PRNGKey(int(cliargs.seed))

N = int(cliargs.N)
d = int(cliargs.d)

print('N = ', N)
print('seed = ', cliargs.seed)
print('d = ', d)

local_d = 0
k = 1

#--> set up circuit 
circ_params = gaussian2D.PrimalParams(N, N, d, local_d, key, k = k, mode = "ssh", h_mode = "ssh", init_state_desc = "GS")
key, subkey = jax.random.split(circ_params.key_after_ham_gen)

#--> noiseless solution
final_energy = gaussian2D.energy_after_circuit(circ_params)
clean_sol = final_energy
print("clean sol = ", clean_sol)

#--> set up dual parameters
p = float(cliargs.p)
k_dual = int(cliargs.k_dual)
lambda_lower_bounds = 0.0 * jnp.ones(d)
dual_params = gaussian2D.DualParams(circ_params, p, k_dual, lambda_lower_bounds)

#--> compute heisenberg picture projections on to local Hams
gaussian2D.set_all_sigmas(dual_params)
proj_sigmas_vec = gaussian2D.sigmas_to_vec(dual_params.sigmas_proj, dual_params)

#--> full von neumann dual
dual_vars_init = jnp.zeros((dual_params.total_num_dual_vars,))
dual_vars_init = dual_vars_init.at[d:].set(proj_sigmas_vec)
dual_vars_init = dual_vars_init.at[:d].set(-3)

# key, subkey = jax.random.split(key)
# dual_vars_init = jax.random.uniform(key, shape = (dual_params.total_num_dual_vars,))/N

num_steps = int(3e3)
dual_obj_over_opti, dual_opt_result = \
    gaussian2D.optimize(dual_vars_init, dual_params,
                      gaussian2D.dual_obj, gaussian2D.dual_grad,
                      num_iters = num_steps)
noisy_bound = -gaussian2D.dual_obj(jnp.array(dual_opt_result.x), dual_params)
print("noisy bound = ", noisy_bound)

noisy_sol = gaussian2D.noisy_primal(circ_params, p)
print("noisy sol = ", noisy_sol)
print("noisy bound <= noisy sol? ")
if noisy_bound <= float(np.real(noisy_sol)):
    print("True")
else:
    print("False")

#--> no channel dual
num_steps = int(10e3)
dual_vars_init_nc = jnp.array([0.0])
dual_obj_over_opti_nc, dual_opt_result_nc = \
    gaussian2D.optimize(dual_vars_init_nc, dual_params,
                      gaussian2D.dual_obj_no_channel, gaussian2D.dual_grad_no_channel,
                      num_iters = num_steps)
noisy_bound_nc = -gaussian2D.dual_obj_no_channel(jnp.array(dual_opt_result_nc.x), dual_params)

print("noisy bound nc = ", noisy_bound_nc)
print("noisy bound nc <= noisy bound? ")
if noisy_bound_nc <= noisy_bound:
    print("True")
else:
    print("False")

#--> full purity dual
# lambda_lower_bounds_purity = jnp.array([0.0])
# dual_params_purity = gaussian2D.DualParamsPurity(circ_params, p, k_dual, lambda_lower_bounds_purity)

# step_size = 0.001
# num_steps = int(10e3)

# dual_vars_init_purity = jnp.zeros((dual_params_purity.total_num_dual_vars,))
# dual_vars_init_purity = dual_vars_init_purity.at[1:].set(proj_sigmas_vec)

# # dual_vars_init_purity = 1e-9 * jnp.ones((dual_params_purity.total_num_dual_vars,))
# # dual_vars_init_purity = dual_vars_init_purity.at[0].set(dual_opt_result_nc.x[0])

# # dual_obj_over_opti_purity, dual_opt_result_purity = \
# #     gaussian2D.optax_optimize(dual_vars_init_purity, dual_params_purity,
# #                       gaussian2D.dual_obj_purity, gaussian2D.dual_grad_purity,
# #                       num_iters = num_steps, tol_scale = 1e-7)
# # noisy_bound_purity = -gaussian2D.dual_obj_purity(jnp.array(dual_opt_result_purity.x), dual_params_purity)

# dual_obj_over_opti_purity_optax, dual_opt_result_purity_optax = \
# gaussian2D.optax_optimize(gaussian2D.dual_obj_purity, gaussian2D.dual_grad_purity,
#                   dual_vars_init_purity, dual_params_purity,
#                   step_size, num_steps, method = "adam")
# noisy_bound_purity = -jnp.min(dual_obj_over_opti_purity_optax)

#--> full purity dual (smooth)
# lambda_lower_bounds_purity = jnp.array([0.0])
# dual_params_purity_smooth = gaussian2D.DualParamsPuritySmooth(circ_params, p, k_dual, lambda_lower_bounds_purity)

# num_steps = 15e3 
# dual_vars_init_purity_smooth = jnp.zeros((dual_params_purity_smooth.total_num_dual_vars,))
# dual_vars_init_purity_smooth = dual_vars_init_purity_smooth.at[d:].set(proj_sigmas_vec)
# dual_vars_init_purity_smooth = dual_vars_init_purity_smooth.at[:d].set(-4)

# dual_obj_over_opti_purity_smooth, dual_opt_result_purity_smooth = \
#     gaussian2D.optimize(dual_vars_init_purity_smooth, dual_params_purity_smooth,
#                     gaussian2D.dual_obj_purity_smooth, gaussian2D.dual_grad_purity_smooth,
#                     num_iters = num_steps, tol_scale = 1e-7)
# noisy_bound_purity = -gaussian2D.dual_obj_purity_smooth(jnp.array(dual_opt_result_purity_smooth.x), dual_params_purity_smooth)

noisy_bound_purity = -69

print(noisy_sol)
print(noisy_bound)
print(noisy_bound_purity)
print(noisy_bound_nc)

data_list = [clean_sol, noisy_sol, noisy_bound, noisy_bound_nc, noisy_bound_purity,
             lambda_lower_bounds,
             dual_obj_over_opti, dual_obj_over_opti_nc]

data_file_name = "fermion2D-block-purity-N-" + str(N) + "-d-" + str(d) + "-seed-" + cliargs.seed + "-p-" + str(p) + "-kdual-" + \
                 str(k_dual) + ".pkl"

with open(os.path.join(cliargs.result_save_path, data_file_name), "wb") as f_for_pkl:
    pickle.dump(data_list, f_for_pkl)
