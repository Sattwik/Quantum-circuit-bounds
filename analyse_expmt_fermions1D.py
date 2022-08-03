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

data_path = "./../vqa_data/0727/20220727-125505/"

N_list = [10,15,20,25,30]
p_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

num_k_duals = 2
num_N = len(N_list)
num_p = len(p_list)
num_seeds = 5

clean_sol_list = np.zeros((num_N, num_seeds, num_p, num_k_duals), dtype = float)
noisy_sol_list = np.zeros((num_N, num_seeds, num_p, num_k_duals), dtype = float)
noisy_bound_list = np.zeros((num_N, num_seeds, num_p, num_k_duals), dtype = float)
noisy_bound_nc_list = np.zeros((num_N, num_seeds, num_p, num_k_duals), dtype = float)

for i_N, N in enumerate(N_list):
    for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
        for i_p, p in enumerate(p_list):
            for i_k, k_dual in enumerate([1, N]):
                fname = "fermion1D-N-" + str(N) + "-seed-" + str(seed) + \
                        "-p-" + str(p) + "-kdual-" + \
                        str(k_dual) + ".pkl"
                with open(os.path.join(data_path, fname), 'rb') as result_file:
                    print("Reading file: " + os.path.join(data_path, fname))

                    clean_sol, noisy_sol, noisy_bound, noisy_bound_nc, \
                    lambda_lower_bounds, \
                    dual_obj_over_opti, dual_obj_over_opti_nc = pickle.load(result_file)

                    clean_sol_list[i_N, i_seed, i_p, i_k] = clean_sol
                    noisy_sol_list[i_N, i_seed, i_p, i_k] = noisy_sol
                    noisy_bound_list[i_N, i_seed, i_p, i_k] = noisy_bound
                    noisy_bound_nc_list[i_N, i_seed, i_p, i_k] = noisy_bound_nc


approx_ratio_noisy_sol = noisy_sol_list/clean_sol_list
approx_ratio_noisy_bound = noisy_bound_list/clean_sol_list
approx_ratio_noisy_bound_nc = noisy_bound_nc_list/clean_sol_list

# ---- Plotting the approximation ratios ----- #


for i_N, N in enumerate(N_list):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mean_approx_noisy_sol = np.mean(approx_ratio_noisy_sol[i_N, :, :, 1], axis = 0)
    std_approx_noisy_sol = np.std(approx_ratio_noisy_sol[i_N, :, :, 1], axis = 0)

    ax.plot(p_list, mean_approx_noisy_sol, marker = '^', color = CB_color_cycle[3], label = r'Primal')
    ax.fill_between(p_list,
                    (mean_approx_noisy_sol - std_approx_noisy_sol),
                    (mean_approx_noisy_sol + std_approx_noisy_sol),
                    color = CB_color_cycle[3], alpha = 0.1)

    mean_approx_noisy_bound_nc = np.mean(approx_ratio_noisy_bound_nc[i_N, :, :, 1], axis = 0)
    std_approx_noisy_bound_nc = np.std(approx_ratio_noisy_bound_nc[i_N, :, :, 1], axis = 0)

    ax.plot(p_list, mean_approx_noisy_bound_nc, marker = 'D', color = CB_color_cycle[5], label = r'Dual (NC)')
    ax.fill_between(p_list,
                    (mean_approx_noisy_bound_nc - std_approx_noisy_bound_nc),
                    (mean_approx_noisy_bound_nc + std_approx_noisy_bound_nc),
                    color = CB_color_cycle[5], alpha = 0.1)

    for i_k, k_dual in enumerate([1, N]):
        mean_approx_noisy_bound = np.mean(approx_ratio_noisy_bound[i_N, :, :, i_k], axis = 0)
        std_approx_noisy_bound = np.std(approx_ratio_noisy_bound[i_N, :, :, i_k], axis = 0)

        ax.plot(p_list, mean_approx_noisy_bound, marker = '.', color = CB_color_cycle[i_k], label = r'Dual (k = ' + str(k_dual) + ')')
        ax.fill_between(p_list,
                        (mean_approx_noisy_bound - std_approx_noisy_bound),
                        (mean_approx_noisy_bound + std_approx_noisy_bound),
                        color = CB_color_cycle[i_k], alpha = 0.1)


    ax.set_ylabel('Approx. ratio')
    ax.set_xlabel(r'$p$')
    ax.legend()
    plt.tight_layout()

    figname = "approx_ratios_" + str(N) + ".pdf"
    plt.savefig(os.path.join(data_path, figname), format = 'pdf')


# figname = "solutions.pdf"
# figname = "approx_ratios_with_nc.pdf"
# figname = "approx_ratios_with_nc_no_clean.pdf"
# figname = "approx_ratios_no_nc.pdf"


# fig = plt.figure()
# num_bins = 30
# ax = fig.add_subplot(411)
# ax.hist(approx_ratio_clean, label = 'p = 0', alpha = 0.75,
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
#             label = 'p = ' + str(p_noise_list[n_noise]),
#             alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins)
#
#     ax.hist(approx_ratio_noisy_bound[n_noise, :],
#             label = 'Bound, p = ' + str(p_noise_list[n_noise]),
#             alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins,
#              hatch='//', edgecolor='k', fill=True)
#
#     ax.set_xlim((0.4, 1))
#     ax.set_ylim((0, 6))
#     ax.axvline(x = mean_approx_noisy[n_noise], ls = '--', c = 'k', lw = 0.75)
#     ax.axvline(x = mean_approx_noisy_bound[n_noise], ls = '-.', c = 'k', lw = 0.75)
#     ax.legend(fontsize = 8)
#
# ax.set_xlabel(r'$\alpha$')
# plt.tight_layout()
# figname = "solutions_and_bounds.pdf"
# plt.savefig(os.path.join(data_path, figname), format = 'pdf')
