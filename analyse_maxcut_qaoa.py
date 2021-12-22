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

data_paths = ["./../vqa_data/1215/20211215-184159/", "./../vqa_data/1215/20211215-201615/"]
data_file_name = "qaoa_maxcut_exact.pkl"
data_file_name = "qaoa_maxcut_exact.pkl"

p_noise_lists = []
actual_sol_list = []
clean_sol_list = []
noisy_sol_list = []
noisy_bound_list = []

for data_path in data_paths:

    with open(os.path.join(data_path, data_file_name), "rb") as f_for_pkl:

        m, n, d, p_noise_list,\
        actual_sol, clean_sol, noisy_sol, noisy_bound = pickle.load(f_for_pkl)

num_random_graphs = actual_sol.shape[0]
num_init_states = actual_sol.shape[1]
num_noise_probs = noisy_sol.shape[0]

approx_ratio_clean = clean_sol/actual_sol
approx_ratio_noisy = noisy_sol/actual_sol
approx_ratio_noisy_bound = noisy_bound/actual_sol

approx_ratio_clean = np.reshape(approx_ratio_clean, (num_random_graphs * num_init_states))
approx_ratio_noisy = np.reshape(approx_ratio_noisy, (num_noise_probs, num_random_graphs * num_init_states))
approx_ratio_noisy_bound = np.reshape(approx_ratio_noisy_bound, (num_noise_probs, num_random_graphs * num_init_states))

mean_approx_clean = np.mean(approx_ratio_clean)
std_approx_clean = np.std(approx_ratio_clean)

mean_approx_noisy = np.mean(approx_ratio_noisy, axis = 1)
std_approx_noisy = np.std(approx_ratio_noisy, axis = 1)

mean_approx_noisy_bound = np.mean(approx_ratio_noisy_bound, axis = 1)
std_approx_noisy_bound = np.std(approx_ratio_noisy_bound, axis = 1)

# ---- Plotting the approximation ratios ----- #

fig = plt.figure()
num_bins = 30
ax = fig.add_subplot(411)
ax.hist(approx_ratio_clean, label = 'p = 0', alpha = 0.75,
        color = CB_color_cycle[0], bins = num_bins)
ax.legend(fontsize = 8)
ax.set_xlim((0.4, 1))
ax.set_ylim((0, 6))
ax.axvline(x = mean_approx_clean, ls = '--', c = 'k')

subplot_list = [412, 413, 414]

for n_noise in range(num_noise_probs):
    ax = fig.add_subplot(subplot_list[n_noise])
    ax.hist(approx_ratio_noisy[n_noise, :],
            label = 'p = ' + str(p_noise_list[n_noise]),
            alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins)

    ax.hist(approx_ratio_noisy_bound[n_noise, :],
            label = 'Bound, p = ' + str(p_noise_list[n_noise]),
            alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins,
             hatch='//', edgecolor='k', fill=True)

    ax.set_xlim((0.4, 1))
    ax.set_ylim((0, 6))
    ax.axvline(x = mean_approx_noisy[n_noise], ls = '--', c = 'k', lw = 0.75)
    ax.axvline(x = mean_approx_noisy_bound[n_noise], ls = '-.', c = 'k', lw = 0.75)
    ax.legend(fontsize = 8)

ax.set_xlabel(r'$\alpha$')
plt.tight_layout()
figname = "solutions_and_bounds.pdf"
plt.savefig(os.path.join(data_path, figname), format = 'pdf')

# ---- Plotting the duality gap ----- #
duality_gap = noisy_sol - noisy_bound
duality_gap = np.reshape(duality_gap, (num_noise_probs, num_random_graphs * num_init_states))
mean_duality_gap = np.mean(duality_gap, axis = 1)
std_duality_gap = np.std(duality_gap, axis = 1)

fig = plt.figure()
num_bins = 30
subplot_list = [311, 312, 313]

for n_noise in range(num_noise_probs):
    ax = fig.add_subplot(subplot_list[n_noise])
    ax.hist(duality_gap[n_noise, :],
            label = 'p = ' + str(p_noise_list[n_noise]),
            alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins)

    ax.set_ylim((0, 5))
    ax.axvline(x = mean_duality_gap[n_noise], ls = '--', c = 'k', lw = 0.75)
    ax.legend(fontsize = 8)

ax.set_xlabel('Duality gap')
plt.tight_layout()
figname = "duality_gap.pdf"
plt.savefig(os.path.join(data_path, figname), format = 'pdf')
