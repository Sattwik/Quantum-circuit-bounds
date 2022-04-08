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

data_path = "./../vqa_data/0218/20220318-004248/"
data_file_name = "maxcut1D_exact.pkl"

p_noise_lists = []
actual_sol_list = []
noisy_sol_list = []
noisy_bound_list = []
graphs_list = []

with open(os.path.join(data_path, data_file_name), "rb") as f_for_pkl:

    # m, n, d, p_noise_list,\
    # actual_sol, clean_sol, noisy_sol, noisy_bound = pickle.load(f_for_pkl)

    m, d, p_noise_list,\
        actual_sol, noisy_sol, noisy_bound,\
        graphs_list = pickle.load(f_for_pkl)

num_random_graphs = actual_sol.shape[0]
num_init_states = actual_sol.shape[1]
num_noise_probs = noisy_sol.shape[0]

approx_ratio_noisy = noisy_sol/actual_sol
approx_ratio_noisy_bound = noisy_bound/actual_sol

approx_ratio_noisy = np.reshape(approx_ratio_noisy, (num_noise_probs, num_random_graphs * num_init_states))
approx_ratio_noisy_bound = np.reshape(approx_ratio_noisy_bound, (num_noise_probs, num_random_graphs * num_init_states))

mean_approx_noisy = np.mean(approx_ratio_noisy, axis = 1)
std_approx_noisy = np.std(approx_ratio_noisy, axis = 1)

mean_approx_noisy_bound = np.mean(approx_ratio_noisy_bound, axis = 1)
std_approx_noisy_bound = np.std(approx_ratio_noisy_bound, axis = 1)

# ---- Plotting the approximation ratios ----- #

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(p_noise_list, mean_approx_noisy, marker = '^', color = CB_color_cycle[3], label = r'Noisy')
ax.fill_between(p_noise_list, (mean_approx_noisy - std_approx_noisy), (mean_approx_noisy + std_approx_noisy), color = CB_color_cycle[3], alpha = 0.1)

ax.plot(p_noise_list, mean_approx_noisy_bound, marker = '.', color = CB_color_cycle[0], label = r'Bound')
ax.fill_between(p_noise_list, (mean_approx_noisy_bound - std_approx_noisy_bound), (mean_approx_noisy_bound + std_approx_noisy_bound), color = CB_color_cycle[0], alpha = 0.1)

ax.axhline(y = 0.85, ls = '--', lw = 1, color = 'gray')

ax.set_ylabel('Approx. ratio')
ax.set_xlabel(r'$p$')
ax.legend()
plt.tight_layout()
# figname = "solutions.pdf"
# figname = "approx_ratios_with_nc.pdf"
# figname = "approx_ratios_with_nc_no_clean.pdf"
# figname = "approx_ratios_no_nc.pdf"
figname = "approx_ratios_no_nc_no_clean.pdf"
plt.savefig(os.path.join(data_path, figname), format = 'pdf')

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


# ---- Plotting the duality gap ----- #
duality_gap = noisy_sol - noisy_bound
duality_gap = np.reshape(duality_gap, (num_noise_probs, num_random_graphs * num_init_states))
mean_duality_gap = np.mean(duality_gap, axis = 1)
std_duality_gap = np.std(duality_gap, axis = 1)

# fig = plt.figure()
# num_bins = 30
# subplot_list = [311, 312, 313]
#
# for n_noise in range(num_noise_probs):
#     ax = fig.add_subplot(subplot_list[n_noise])
#     ax.hist(duality_gap[n_noise, :],
#             label = 'p = ' + str(p_noise_list[n_noise]),
#             alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins)
#
#     ax.set_ylim((0, 5))
#     ax.axvline(x = mean_duality_gap[n_noise], ls = '--', c = 'k', lw = 0.75)
#     ax.legend(fontsize = 8)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(p_noise_list, mean_duality_gap, marker = '.', color = CB_color_cycle[0], label = 'QAOA, individual noise')
# ax.fill_between(p_noise_list, (mean_duality_gap - std_duality_gap), (mean_duality_gap + std_duality_gap), color = CB_color_cycle[0], alpha = 0.1)
# ax.plot(p_noise_list, mean_duality_gap_nc, marker = 'D', color = CB_color_cycle[1], label = 'No channel')
# ax.fill_between(p_noise_list, (mean_duality_gap_nc - std_duality_gap_nc), (mean_duality_gap_nc + std_duality_gap_nc), color = CB_color_cycle[1], alpha = 0.1)
# ax.set_ylabel('Duality gap')
# ax.set_xlabel(r'$p$')
# ax.legend()
# plt.tight_layout()
# figname = "duality_gap.pdf"
# plt.savefig(os.path.join(data_path, figname), format = 'pdf')
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(p_noise_list, mean_duality_gap_global, marker = '.', color = CB_color_cycle[2])
# ax.fill_between(p_noise_list, (mean_duality_gap_global - std_duality_gap_global), (mean_duality_gap_global + std_duality_gap_global), color = CB_color_cycle[2], alpha = 0.1, label = 'QAOA, global noise')
# ax.set_ylabel('Duality gap')
# ax.set_xlabel(r'$p$')
# plt.tight_layout()
# figname = "duality_gap_global.pdf"
# plt.savefig(os.path.join(data_path, figname), format = 'pdf')

# ---- Plotting the solutions/bounds ----- #
# actual_sol = np.reshape(actual_sol, (num_random_graphs * num_init_states))
# mean_actual_sol = np.mean(actual_sol, axis = 0)
# std_actual_sol = np.std(actual_sol, axis = 0)
#
# clean_sol = np.reshape(clean_sol, (num_random_graphs * num_init_states))
# mean_clean_sol = np.mean(clean_sol, axis = 0)
# std_clean_sol = np.std(clean_sol, axis = 0)
#
# noisy_sol = np.reshape(noisy_sol, (num_noise_probs, num_random_graphs * num_init_states))
# mean_noisy_sol = np.mean(noisy_sol, axis = 1)
# std_noisy_sol = np.std(noisy_sol, axis = 1)
#
# noisy_bound = np.reshape(noisy_bound, (num_noise_probs, num_random_graphs * num_init_states))
# mean_noisy_bound = np.mean(noisy_bound, axis = 1)
# std_noisy_bound = np.std(noisy_bound, axis = 1)
#
# noisy_bound_nc = np.reshape(noisy_bound_nc, (num_noise_probs, num_random_graphs * num_init_states))
# mean_noisy_bound_nc = np.mean(noisy_bound_nc, axis = 1)
# std_noisy_bound_nc = np.std(noisy_bound_nc, axis = 1)
#
# noisy_sol_global = np.reshape(noisy_sol_global, (num_noise_probs, num_random_graphs * num_init_states))
# mean_noisy_sol_global = np.mean(noisy_sol_global, axis = 1)
# std_noisy_sol_global = np.std(noisy_sol_global, axis = 1)
#
# noisy_bound_global = np.reshape(noisy_bound_global, (num_noise_probs, num_random_graphs * num_init_states))
# mean_noisy_bound_global = np.mean(noisy_bound_global, axis = 1)
# std_noisy_bound_global = np.std(noisy_bound_global, axis = 1)
#
# # fig = plt.figure()
# # num_bins = 30
# # subplot_list = [311, 312, 313]
# #
# # for n_noise in range(num_noise_probs):
# #     ax = fig.add_subplot(subplot_list[n_noise])
# #     ax.hist(duality_gap[n_noise, :],
# #             label = 'p = ' + str(p_noise_list[n_noise]),
# #             alpha = 0.75, color = CB_color_cycle[n_noise + 1], bins = num_bins)
# #
# #     ax.set_ylim((0, 5))
# #     ax.axvline(x = mean_duality_gap[n_noise], ls = '--', c = 'k', lw = 0.75)
# #     ax.legend(fontsize = 8)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# # ax.plot(p_noise_list, mean_actual_sol, marker = 's', color = CB_color_cycle[5], label = 'MaxCut, exact solution')
# # ax.fill_between(p_noise_list, (mean_actual_sol - std_actual_sol), (mean_actual_sol + std_actual_sol), color = CB_color_cycle[5], alpha = 0.1)
#
# ax.plot(p_noise_list, mean_noisy_sol, marker = '^', color = CB_color_cycle[3], label = r'$v_{\text{noisy}}$')
# ax.fill_between(p_noise_list, (mean_noisy_sol - std_noisy_sol), (mean_noisy_sol + std_noisy_sol), color = CB_color_cycle[3], alpha = 0.1)
#
# ax.plot(p_noise_list, mean_noisy_bound, marker = '.', color = CB_color_cycle[0], label = r'$w_{\text{noisy}}$')
# ax.fill_between(p_noise_list, (mean_noisy_bound - std_noisy_bound), (mean_noisy_bound + std_noisy_bound), color = CB_color_cycle[0], alpha = 0.1)
#
# ax.plot([0, 1], [mean_clean_sol, mean_clean_sol], ls = "-.", lw = 1, color = CB_color_cycle[2], label = r'$v_{\text{clean}}$')
# ax.fill_between(p_noise_list, [(mean_clean_sol - std_clean_sol)] * len(p_noise_list), [(mean_clean_sol + std_clean_sol)] * len(p_noise_list), color = CB_color_cycle[2], alpha = 0.1)
#
# # ax.plot(p_noise_list, mean_noisy_bound_nc, marker = 'D', color = CB_color_cycle[1], label = r'$w_{\text{noisy}}^{\prime}$')
# # ax.fill_between(p_noise_list, (mean_noisy_bound_nc - std_noisy_bound_nc), (mean_noisy_bound_nc + std_noisy_bound_nc), color = CB_color_cycle[1], alpha = 0.1)
#
# ax.plot([0, 1], [np.mean(actual_sol), np.mean(actual_sol)], ls = "--", color = 'k', label = r'$v_{\text{solution}}$', lw = 1)
# ax.fill_between(p_noise_list, [(mean_actual_sol - std_actual_sol)] * len(p_noise_list), [(mean_actual_sol + std_actual_sol)] * len(p_noise_list), color = 'k', alpha = 0.05)
# ax.fill_between(p_noise_list, [(mean_actual_sol - std_actual_sol)] * len(p_noise_list), [(mean_actual_sol + std_actual_sol)] * len(p_noise_list), color = 'none', alpha = 0.05, hatch = '///', edgecolor = 'w')
#
# ax.set_ylabel('Solution')
# ax.set_xlabel(r'$p$')
# ax.legend()
# plt.tight_layout()
# # figname = "solutions.pdf"
# figname = "solutions_no_nc.pdf"
# plt.savefig(os.path.join(data_path, figname), format = 'pdf')
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(p_noise_list, mean_noisy_sol_global, marker = '^', color = CB_color_cycle[4], label = 'QAOA MaxCut, global noise, primal')
# ax.fill_between(p_noise_list, (mean_noisy_sol_global - std_noisy_sol_global), (mean_noisy_sol_global + std_noisy_sol_global), color = CB_color_cycle[4], alpha = 0.1)
#
# ax.plot(p_noise_list, mean_noisy_bound_nc, marker = 'D', color = CB_color_cycle[1], label = 'No channel, dual')
# ax.fill_between(p_noise_list, (mean_noisy_bound_nc - std_noisy_bound_nc), (mean_noisy_bound_nc + std_noisy_bound_nc), color = CB_color_cycle[1], alpha = 0.1)
#
# ax.plot(p_noise_list, mean_noisy_bound_global, marker = '.', color = CB_color_cycle[2], label = 'QAOA MaxCut, global noise, dual')
# ax.fill_between(p_noise_list, (mean_noisy_bound_global - std_noisy_bound_global), (mean_noisy_bound_global + std_noisy_bound_global), color = CB_color_cycle[2], alpha = 0.1)
#
# ax.set_ylabel('Solution')
# ax.set_xlabel(r'$p$')
# plt.tight_layout()
# ax.legend(loc = 'lower right')
# figname = "solutions_global.pdf"
# plt.savefig(os.path.join(data_path, figname), format = 'pdf')
