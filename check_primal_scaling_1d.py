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
import matplotlib

from fermions import gaussian, gaussian2D, fermion_test_utils

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rcParams["image.cmap"] = "inferno"
matplotlib.rcParams["axes.titlesize"] = 16
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

#------------------------------------------------------------------------------#

colorama.init()

N_list = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

clean_sols_1D = np.zeros((len(N_list), 5))
noisy_sols_nc_1D = np.zeros((len(N_list), 5))
noisy_sols_1D = np.zeros((len(N_list), 5))

# clean_sols_2D = np.zeros((len(N_list), 5))
# noisy_sols_nc_2D = np.zeros((len(N_list), 5))
# noisy_sols_2D = np.zeros((len(N_list), 5))

for i_N, N in enumerate(N_list):

    local_d = 0
    k = 1
    d = N
    p = 0.1

    seed_list = N + np.array(range(5))

    for i_seed, seed in enumerate(seed_list):
        print('seed = ', seed)
        rng = np.random.default_rng()
        key = jax.random.PRNGKey(seed)

        # 1D, full circuit
        circ_params = gaussian.PrimalParams(N, d, local_d, key, k = k, mode = 'block')
        key, subkey = jax.random.split(circ_params.key_after_ham_gen)

        w_parent, v_parent = jnp.linalg.eig(1j * circ_params.h_parent)
        w_parent = np.real(w_parent)
        gs_energy_parent = sum(w_parent[w_parent < 0])

        # print(colorama.Fore.GREEN + "gs energy = ", gs_energy_parent)
        # print(colorama.Style.RESET_ALL)
        #
        # final_energy = gaussian.energy_after_circuit(circ_params)
        # print(colorama.Fore.GREEN + "circ energy = ", final_energy)
        # print(colorama.Style.RESET_ALL)

        noisy_sol = gaussian.noisy_primal(circ_params, p)

        clean_sols_1D[i_N, i_seed] = gs_energy_parent
        noisy_sols_1D[i_N, i_seed] = noisy_sol
        # print(colorama.Fore.GREEN + "noisy sol = ", noisy_sol)

        # 1D, only noise
        circ_params_nc = gaussian.PrimalParams(N, d, local_d, key, k = k, mode = 'block')
        circ_params_nc.layer_hamiltonians = 0 * circ_params_nc.layer_hamiltonians
        key, subkey = jax.random.split(circ_params_nc.key_after_ham_gen)
        noisy_sol_nc = gaussian.noisy_primal(circ_params_nc, p)

        noisy_sols_nc_1D[i_N, i_seed] = noisy_sol_nc

        # 2D, full circuit
        # circ_params_2D = gaussian2D.PrimalParams(N, N, d, local_d, key, k = k + 2, mode = 'block')
        # key, subkey = jax.random.split(circ_params_2D.key_after_ham_gen)
        #
        # clean_sol_2D = gaussian2D.energy_after_circuit(circ_params_2D)
        # clean_sols_2D[i_N, i_seed] = clean_sol_2D
        #
        # noisy_sol_2D = gaussian2D.noisy_primal(circ_params_2D, p)
        # noisy_sols_2D[i_N, i_seed] = noisy_sol_2D
        #
        # # 2D, only noise
        # circ_params_2D_nc = gaussian2D.PrimalParams(N, N, d, local_d, key, k = k, mode = 'block')
        # key, subkey = jax.random.split(circ_params_2D_nc.key_after_ham_gen)
        # circ_params_2D_nc.layer_hamiltonians = 0 * circ_params_2D_nc.layer_hamiltonians
        #
        # noisy_sol_2D_nc = gaussian2D.noisy_primal(circ_params_2D_nc, p)
        # noisy_sols_nc_2D[i_N, i_seed] = noisy_sol_2D_nc


# print(colorama.Style.RESET_ALL)

approx_ratio_noisy_sol_nc_1D = noisy_sols_nc_1D/clean_sols_1D
approx_ratio_noisy_sol_1D = noisy_sols_1D/clean_sols_1D
# approx_ratio_noisy_sol_2D = noisy_sols_2D/clean_sols_2D
# approx_ratio_noisy_sol_nc_2D = noisy_sols_nc_2D/clean_sols_2D

mean_approx_noisy_sol_1D = np.mean(approx_ratio_noisy_sol_1D, axis = 1)
std_approx_noisy_sol_1D = np.std(approx_ratio_noisy_sol_1D, axis = 1)

mean_approx_noisy_sol_nc_1D = np.mean(approx_ratio_noisy_sol_nc_1D, axis = 1)
std_approx_noisy_sol_nc_1D = np.std(approx_ratio_noisy_sol_nc_1D, axis = 1)

# mean_approx_noisy_sol_2D = np.mean(approx_ratio_noisy_sol_2D, axis = 1)
# std_approx_noisy_sol_2D = np.std(approx_ratio_noisy_sol_2D, axis = 1)
#
# mean_approx_noisy_sol_nc_2D = np.mean(approx_ratio_noisy_sol_nc_2D, axis = 1)
# std_approx_noisy_sol_nc_2D = np.std(approx_ratio_noisy_sol_nc_2D, axis = 1)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(N_list, mean_approx_noisy_sol_nc_1D, marker = '^', color = CB_color_cycle[0], label = r'1D (only noise)')

ax.plot(N_list, mean_approx_noisy_sol_1D, marker = 'o', color = CB_color_cycle[1], label = r'1D')
ax.fill_between(N_list,(mean_approx_noisy_sol_1D - std_approx_noisy_sol_1D), (mean_approx_noisy_sol_1D + std_approx_noisy_sol_1D), color = CB_color_cycle[1], alpha = 0.1)

# ax.plot(N_list, mean_approx_noisy_sol_nc_2D, marker = '^', color = CB_color_cycle[2], label = r'2D (only noise)', ls = '--')
#
# ax.plot(N_list, mean_approx_noisy_sol_2D, marker = 'o', color = CB_color_cycle[3], label = r'2D')
# ax.fill_between(N_list, (mean_approx_noisy_sol_2D - std_approx_noisy_sol_2D), (mean_approx_noisy_sol_2D + std_approx_noisy_sol_2D), color = CB_color_cycle[3], alpha = 0.1)



ax.set_ylabel('Approx. ratio')
ax.set_xlabel(r'$N$')
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig("./../vqa_data/fermion_primal_scaling.pdf", format = 'pdf')
