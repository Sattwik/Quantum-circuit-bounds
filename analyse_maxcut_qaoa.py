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

file_name = "./../vqa_data/1104/20211104-220940/qaoa_maxcut_exact.pkl"

with open(os.path.join(file_name), "rb") as f_for_pkl:

    m, n, p, p_noise_list,\
    actual_maxcut_sol,\
    clean_QAOA_maxcut_sol, noisy_QAOA_maxcut_sol = pickle.load(f_for_pkl)

num_random_graphs = actual_maxcut_sol.shape[0]
num_init_states = actual_maxcut_sol.shape[1]
num_noise_probs = noisy_QAOA_maxcut_sol.shape[0]

approx_ratio_clean = clean_QAOA_maxcut_sol/actual_maxcut_sol
approx_ratio_noisy = noisy_QAOA_maxcut_sol/actual_maxcut_sol

approx_ratio_clean = np.reshape(approx_ratio_clean,
                                (num_random_graphs * num_init_states))
approx_ratio_noisy = np.reshape(approx_ratio_noisy,
                                (num_noise_probs, num_random_graphs * num_init_states))

mean_approx_clean = np.mean(approx_ratio_clean)
std_approx_clean = np.std(approx_ratio_clean)

mean_approx_noisy = np.mean(approx_ratio_noisy, axis = 1)
std_approx_noisy = np.std(approx_ratio_noisy, axis = 1)

fig = plt.figure()
num_bins = 50
ax = fig.add_subplot(411)
ax.hist(approx_ratio_clean, label = 'Clean QAOA', alpha = 0.75,
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
