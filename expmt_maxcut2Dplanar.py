import pickle
import os
from datetime import datetime
from datetime import date
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import qutip
from matplotlib import rc
import matplotlib
import jax.numpy as jnp

from vqa import graphs, problems, algorithms, dual, dual_jax

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rcParams["image.cmap"] = "inferno"
matplotlib.rcParams["axes.titlesize"] = 16
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

m = 2
n = 3
lattice = graphs.define_lattice(m = m, n = n)

d = 2

num_random_graphs = 20
num_init_states = 1
p_noise_list = [0.001, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.00]
num_noise_probs = len(p_noise_list)
opt_method = 'L-BFGS-B'
# in ['CG', 'Newton-CG', 'BFGS', 'L-BFGS-B']

actual_sol = np.zeros((num_random_graphs, num_init_states))
clean_sol = np.zeros((num_random_graphs, num_init_states))
noisy_sol = np.zeros((num_noise_probs, num_random_graphs, num_init_states))
noisy_bound = np.zeros((num_noise_probs, num_random_graphs, num_init_states))
noisy_bound_nc = np.zeros((num_noise_probs, num_random_graphs, num_init_states))
noisy_sol_global = np.zeros((num_noise_probs, num_random_graphs, num_init_states))
noisy_bound_global = np.zeros((num_noise_probs, num_random_graphs, num_init_states))

graphs_list = []

for n_graph in range(num_random_graphs):

    graph = graphs.create_random_connectivity(lattice)
    graphs_list.append(graph)

    maxcut_obj = problems.MaxCut(graph, lattice)

    mc_probs = np.zeros(graph.number_of_nodes())
    p = 0
    qaoa_obj = algorithms.QAOA(maxcut_obj, d, p, mc_probs)

    start = time.time()

    for n_init in range(num_init_states):

        if n_init == 0:
            gamma_init = np.ones(d) * 1.1
            beta_init = np.ones(d) * 1.1
            gamma_beta_init = np.concatenate((gamma_init, beta_init))

        else:
            gamma_init = np.random.uniform(low = 0, high = 2 * np.pi, size = d)
            beta_init = np.random.uniform(low = 0, high = np.pi, size = d)
            gamma_beta_init = np.concatenate((gamma_init, beta_init))

        print(gamma_beta_init)

        obj_over_opti, opt_result = algorithms.optimize_QAOA(gamma_beta_init,
                                                             qaoa_obj)

        actual_sol[n_graph, n_init] = np.min(maxcut_obj.H)
        clean_sol[n_graph, n_init] = obj_over_opti[-1]

        gamma_beta_clean = opt_result.x
        gamma_clean = np.split(gamma_beta_clean, 2)[0]
        beta_clean = np.split(gamma_beta_clean, 2)[1]

        for n_noise, p in enumerate(p_noise_list):

            print("***********************************************************")
            print("---------- Graph number ", n_graph,
                                    ", init. ", n_init, ", p = ", p, "------")
            print("***********************************************************")

            dual_obj_jax = dual_jax.MaxCutDualJAX(prob_obj = maxcut_obj, d = d,
                        gamma = jnp.array(gamma_clean),
                        beta = jnp.array(beta_clean), p = p)

            a_vars_init = np.zeros(d)
            sigma_vars_init = np.zeros(dual_obj_jax.len_vars - d)
            vars_init = np.concatenate((a_vars_init, sigma_vars_init))

            #-------- complete dual --------#

            _, primal_noisy_jax = dual_obj_jax.primal_noisy()
            primal_noisy_jax = float(jnp.real(primal_noisy_jax))
            noisy_sol[n_noise, n_graph, n_init] = primal_noisy_jax

            obj_over_opti, opt_result = \
                    dual_jax.optimize_external_dual_JAX(vars_init,
                                                        dual_obj_jax,
                                                        opt_method)

            noisy_bound[n_noise, n_graph, n_init] = -obj_over_opti[-1]

            #-------- global dual --------#

            dual_obj_jax_global = dual_jax.MaxCutDualJAXGlobal(
                                            prob_obj = maxcut_obj, d = d, p = p)

            _, primal_noisy_global = dual_obj_jax_global.primal_noisy()
            primal_noisy_global = float(jnp.real(primal_noisy_global))
            noisy_sol_global[n_noise, n_graph, n_init] = primal_noisy_global

            obj_over_opti_global, opt_result_global = \
            dual_jax.optimize_external_dual_JAX(vars_init,
                                                dual_obj_jax_global,
                                                opt_method)

            noisy_bound_global[n_noise, n_graph, n_init] =\
                                                    -obj_over_opti_global[-1]

            #-------- no channel dual --------#

            dual_obj_jax_nochannel = \
            dual_jax.MaxCutDualJAXNoChannel(prob_obj = maxcut_obj, d = d, p = p)

            obj_over_opti_nc, opt_result_nc = \
            dual_jax.optimize_external_dual_JAX(a_vars_init,
                                                dual_obj_jax_nochannel,
                                                opt_method)

            noisy_bound_nc[n_noise, n_graph, n_init] = -obj_over_opti_nc[-1]

    end = time.time()

    print("Time (min) for one graph= ", (end - start)/60)

# Save data
today = date.today()
mmdd =  today.strftime("%m%d%y")[:4]

data_folder_path = os.path.join('./../vqa_data', mmdd)

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

now = datetime.now().strftime("%Y%m%d-%H%M%S")
result_save_path = os.path.join(data_folder_path, now)

if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

file_name = "qaoa_maxcut_exact.pkl"

data_list = [m, n, d, p_noise_list,
             actual_sol, clean_sol,
             noisy_sol, noisy_bound, noisy_bound_nc,
             noisy_sol_global, noisy_bound_global,
             graphs_list]

with open(os.path.join(result_save_path, file_name), "wb") as f_for_pkl:
    pickle.dump(data_list, f_for_pkl)

# approx_ratio_clean = clean_sol/actual_sol
# approx_ratio_noisy = noisy_sol/actual_sol
# approx_ratio_noisy_bound = noisy_bound/actual_sol
#
# approx_ratio_clean = np.reshape(approx_ratio_clean, (num_random_graphs * num_init_states))
# approx_ratio_noisy = np.reshape(approx_ratio_noisy, (num_noise_probs, num_random_graphs * num_init_states))
# approx_ratio_noisy_bound = np.reshape(approx_ratio_noisy_bound, (num_noise_probs, num_random_graphs * num_init_states))
#
# mean_approx_clean = np.mean(approx_ratio_clean)
# std_approx_clean = np.std(approx_ratio_clean)
#
# mean_approx_noisy = np.mean(approx_ratio_noisy, axis = 1)
# std_approx_noisy = np.std(approx_ratio_noisy, axis = 1)
#
# mean_approx_noisy_bound = np.mean(approx_ratio_noisy_bound, axis = 1)
# std_approx_noisy_bound = np.std(approx_ratio_noisy_bound, axis = 1)
#
# fig = plt.figure()
# num_bins = 30
# ax = fig.add_subplot(411)
# ax.hist(approx_ratio_clean, label = 'Clean', alpha = 0.75,
#         color = CB_color_cycle[0], bins = num_bins)
# ax.legend(fontsize = 8)
# ax.set_xlim((0.4, 1))
# ax.set_ylim((0, 6))
# ax.axvline(x = mean_approx_clean, ls = '--', c = 'k')
#
# subplot_list = [412, 413, 414]
#
# for n_noise in range(num_noise_probs):
#     ax = fig.add_subplot(subplot_list[n_noise])
#     ax.hist(approx_ratio_noisy[n_noise, :],
#             label = 'Noisy soln., p = ' + str(p_noise_list[n_noise]),
#             alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins)
#
#     ax.hist(approx_ratio_noisy_bound[n_noise, :],
#             label = 'Noisy bound, p = ' + str(p_noise_list[n_noise]),
#             alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins,
#              hatch='/', edgecolor='k', fill=True)
#
#     ax.set_xlim((0.4, 1))
#     ax.set_ylim((0, 6))
#     ax.axvline(x = mean_approx_noisy[n_noise], ls = '--', c = 'k')
#     ax.axvline(x = mean_approx_noisy[n_noise], ls = '-.', c = 'k')
#     ax.legend(fontsize = 8)
#
# ax.set_xlabel(r'$\alpha$')
# plt.tight_layout()
# plt.show()
