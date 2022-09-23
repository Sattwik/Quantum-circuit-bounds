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

from fermions import gaussian, fermion_test_utils

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

N_list = [14, 16, 18, 20, 22, 24, 26, 28, 30]

clean_sols = np.zeros((len(N_list), 5))
noisy_sols = np.zeros((len(N_list), 5))

for i_N, N in enumerate(N_list):

    print('N = ', N)
    if N%2 == 0:
        d = N - 1
    else:
        d = N
    local_d = 1
    k = 1

    seed_list = N + np.array(range(5))

    for i_seed, seed in enumerate(seed_list):
        print('seed = ', seed)
        rng = np.random.default_rng()
        key = jax.random.PRNGKey(seed)

        circ_params = gaussian.PrimalParams(N, d, local_d, key, k = k, mode = 'block')
        key, subkey = jax.random.split(circ_params.key_after_ham_gen)

        w_parent, v_parent = jnp.linalg.eig(1j * circ_params.h_parent)
        w_parent = np.real(w_parent)
        gs_energy_parent = sum(w_parent[w_parent < 0])

        print(colorama.Fore.GREEN + "gs energy = ", gs_energy_parent)
        print(colorama.Style.RESET_ALL)

        final_energy = gaussian.energy_after_circuit(circ_params)
        print(colorama.Fore.GREEN + "circ energy = ", final_energy)
        print(colorama.Style.RESET_ALL)

        p = 0.1
        noisy_sol = gaussian.noisy_primal(circ_params, p)

        print(colorama.Fore.GREEN + "noisy sol = ", noisy_sol)

        clean_sols[i_N, i_seed] = gs_energy_parent
        noisy_sols[i_N, i_seed] = noisy_sol

print(colorama.Style.RESET_ALL)

approx_ratio_noisy_sol = noisy_sols/clean_sols
fig = plt.figure()
ax = fig.add_subplot(111)

mean_approx_noisy_sol = np.mean(approx_ratio_noisy_sol, axis = 1)
std_approx_noisy_sol = np.std(approx_ratio_noisy_sol, axis = 1)

ax.plot(N_list, mean_approx_noisy_sol, marker = '^', color = CB_color_cycle[3], label = r'Primal')
ax.fill_between(N_list,
                (mean_approx_noisy_sol - std_approx_noisy_sol),
                (mean_approx_noisy_sol + std_approx_noisy_sol),
                color = CB_color_cycle[3], alpha = 0.1)

ax.set_ylabel('Approx. ratio')
ax.set_xlabel(r'$N$')
ax.legend()
ax.set_yscale('log')
