import pickle
import os
from datetime import datetime
from datetime import date

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import qutip
from matplotlib import rc
import matplotlib

from vqa import graphs
from vqa import problems
from vqa import algorithms

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

p = 3

num_random_graphs = 5
num_init_states = 5
num_noise_probs = 3
p_noise_list = [0.001, 0.01, 0.1]

actual_maxcut_sol = np.zeros((num_random_graphs, num_init_states))
clean_QAOA_maxcut_sol = np.zeros((num_random_graphs, num_init_states))
noisy_QAOA_maxcut_sol = np.zeros((num_noise_probs, num_random_graphs, num_init_states))

for n_graph in range(num_random_graphs):

    graph = graphs.create_random_connectivity(lattice)

    maxcut_obj = problems.MaxCut(graph, lattice)

    mc_probs = np.zeros(graph.number_of_nodes())
    p_noise = 0
    qaoa_obj = algorithms.QAOA(maxcut_obj, p, p_noise, mc_probs)

    for n_init in range(num_init_states):

        gamma_init = np.random.uniform(low = 0, high = 2 * np.pi, size = p)
        beta_init = np.random.uniform(low = 0, high = np.pi, size = p)
        gamma_beta_init = np.concatenate((gamma_init, beta_init))

        obj_over_opti, opt_result = algorithms.optimize_QAOA(gamma_beta_init, qaoa_obj)

        actual_maxcut_sol[n_graph, n_init] = np.min(maxcut_obj.H)
        clean_QAOA_maxcut_sol[n_graph, n_init] = obj_over_opti[-1]

        gamma_beta_clean = opt_result.x

        for n_noise, p_noise in enumerate(p_noise_list):

            num_monte_carlo = int(1/p_noise)
            noisy_sol = 0

            for n_mc in range(num_monte_carlo):

                mc_probs = np.random.uniform(low = 0, high = 1, size = (p, graph.number_of_nodes()))
                qaoa_obj_noisy = algorithms.QAOA(maxcut_obj, p, p_noise, mc_probs, noisy = True)

                noisy_sol += algorithms.objective_external_QAOA(
                                gamma_beta_clean, qaoa_obj_noisy)/num_monte_carlo

            noisy_QAOA_maxcut_sol[n_noise, n_graph, n_init] = noisy_sol

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

data_list = [m, n, p, p_noise_list,
             actual_maxcut_sol, clean_QAOA_maxcut_sol, noisy_QAOA_maxcut_sol]

with open(os.path.join(result_save_path, file_name), "wb") as f_for_pkl:
    pickle.dump(data_list, f_for_pkl)

approx_ratio_clean = clean_QAOA_maxcut_sol/actual_maxcut_sol
approx_ratio_noisy = noisy_QAOA_maxcut_sol/actual_maxcut_sol

approx_ratio_clean = np.reshape(approx_ratio_clean, (num_random_graphs * num_init_states))
approx_ratio_noisy = np.reshape(approx_ratio_noisy, (num_noise_probs, num_random_graphs * num_init_states))

mean_approx_clean = np.mean(approx_ratio_clean)
std_approx_clean = np.std(approx_ratio_clean)

mean_approx_noisy = np.mean(approx_ratio_noisy, axis = 1)
std_approx_noisy = np.std(approx_ratio_noisy, axis = 1)

fig = plt.figure()
num_bins = 30
ax = fig.add_subplot(411)
ax.hist(approx_ratio_clean, label = 'Clean', alpha = 0.75,
        color = CB_color_cycle[0], bins = num_bins)
ax.legend(fontsize = 8)
ax.set_xlim((0.4, 1))
ax.set_ylim((0, 6))
ax.axvline(x = mean_approx_clean, ls = '--', c = 'k')

subplot_list = [412, 413, 414]

for n_noise in range(num_noise_probs):
    ax = fig.add_subplot(subplot_list[n_noise])
    ax.hist(approx_ratio_noisy[n_noise, :],
            label = 'Noisy, p = ' + str(p_noise_list[n_noise]),
            alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins)
    ax.set_xlim((0.4, 1))
    ax.set_ylim((0, 6))
    ax.axvline(x = mean_approx_noisy[n_noise], ls = '--', c = 'k')
    ax.legend(fontsize = 8)

ax.set_xlabel(r'$\alpha$')
plt.tight_layout()
plt.show()
