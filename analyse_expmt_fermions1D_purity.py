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

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amsmath}')
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


data_path = "./../vqa_data/0313/20230313-143850"

N_list = [20]
p_list = [0.05]
k_dual_list = [1, 5, 10, 15, 20]
d_list = np.arange(3, 43, 2)

num_k_duals = len(k_dual_list)
num_N = len(N_list)
num_p = len(p_list)
num_d = len(d_list)
num_seeds = 1

clean_sol_list = np.zeros((num_N, num_seeds, num_d, num_p, num_k_duals), dtype = float)
noisy_sol_list = np.zeros((num_N, num_seeds, num_d, num_p, num_k_duals), dtype = float)
noisy_bound_list = np.zeros((num_N, num_seeds, num_d, num_p, num_k_duals), dtype = float)
noisy_bound_purity_list = np.zeros((num_N, num_seeds, num_d, num_p, num_k_duals), dtype = float)
noisy_bound_nc_list = np.zeros((num_N, num_seeds, num_d, num_p, num_k_duals), dtype = float)

for i_N, N in enumerate(N_list):
    for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
        for i_d, d in enumerate(d_list):
            for i_p, p in enumerate(p_list):
                # for i_k, k_dual in enumerate([1, N]):
                for i_k, k_dual in enumerate(k_dual_list):
                    fname = "fermion1D-block-purity-N-" + str(N) + "-d-" + str(d) + "-seed-" + str(seed) + "-p-" + str(p) + "-kdual-" + str(k_dual) + ".pkl"
                    with open(os.path.join(data_path, fname), 'rb') as result_file:
                        print("Reading file: " + os.path.join(data_path, fname))

                        clean_sol, noisy_sol, noisy_bound, noisy_bound_nc, noisy_bound_purity, \
                        lambda_lower_bounds, \
                        dual_obj_over_opti, dual_obj_over_opti_nc = pickle.load(result_file)

                        clean_sol_list[i_N, i_seed, i_d, i_p, i_k] = clean_sol
                        noisy_sol_list[i_N, i_seed, i_d, i_p, i_k] = noisy_sol
                        noisy_bound_list[i_N, i_seed, i_d, i_p, i_k] = noisy_bound
                        noisy_bound_purity_list[i_N, i_seed, i_d, i_p, i_k] = noisy_bound_purity
                        noisy_bound_nc_list[i_N, i_seed, i_d, i_p, i_k] = noisy_bound_nc


approx_ratio_noisy_sol = noisy_sol_list/clean_sol_list
approx_ratio_noisy_bound = noisy_bound_list/clean_sol_list
approx_ratio_noisy_bound_purity = noisy_bound_purity_list/clean_sol_list
approx_ratio_noisy_bound_nc = noisy_bound_nc_list/clean_sol_list

# ---- Plotting the approximation ratios ----- #

# vs p

# for i_N, N in enumerate(N_list):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     mean_approx_noisy_sol = np.mean(approx_ratio_noisy_sol[i_N, :, :, 1], axis = 0)
#     std_approx_noisy_sol = np.std(approx_ratio_noisy_sol[i_N, :, :, 1], axis = 0)
#
#     ax.plot(p_list, mean_approx_noisy_sol, marker = '^', color = CB_color_cycle[3], label = r'Primal')
#     ax.fill_between(p_list,
#                     (mean_approx_noisy_sol - std_approx_noisy_sol),
#                     (mean_approx_noisy_sol + std_approx_noisy_sol),
#                     color = CB_color_cycle[3], alpha = 0.1)
#
#     mean_approx_noisy_bound_nc = np.mean(approx_ratio_noisy_bound_nc[i_N, :, :, 1], axis = 0)
#     std_approx_noisy_bound_nc = np.std(approx_ratio_noisy_bound_nc[i_N, :, :, 1], axis = 0)
#
#     ax.plot(p_list, mean_approx_noisy_bound_nc, marker = 'D', color = CB_color_cycle[5], label = r'Dual (NC)')
#     ax.fill_between(p_list,
#                     (mean_approx_noisy_bound_nc - std_approx_noisy_bound_nc),
#                     (mean_approx_noisy_bound_nc + std_approx_noisy_bound_nc),
#                     color = CB_color_cycle[5], alpha = 0.1)
#
#     for i_k, k_dual in enumerate([1, N]):
#         mean_approx_noisy_bound = np.mean(approx_ratio_noisy_bound[i_N, :, :, i_k], axis = 0)
#         std_approx_noisy_bound = np.std(approx_ratio_noisy_bound[i_N, :, :, i_k], axis = 0)
#
#         ax.plot(p_list, mean_approx_noisy_bound, marker = '.', color = CB_color_cycle[i_k], label = r'Dual (k = ' + str(k_dual) + ')')
#         ax.fill_between(p_list,
#                         (mean_approx_noisy_bound - std_approx_noisy_bound),
#                         (mean_approx_noisy_bound + std_approx_noisy_bound),
#                         color = CB_color_cycle[i_k], alpha = 0.1)
#
#
#     ax.set_ylabel('Approx. ratio')
#     ax.set_xlabel(r'$p$')
#     ax.legend()
#     plt.tight_layout()
#
#     figname = "approx_ratios_" + str(N) + ".pdf"
#     plt.savefig(os.path.join(data_path, figname), format = 'pdf')

# vs N



# for i_p, p in enumerate(p_list):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     mean_approx_noisy_sol = np.mean(approx_ratio_noisy_sol[:, :, i_p, 0], axis = 1)
#     std_approx_noisy_sol = np.std(approx_ratio_noisy_sol[:, :, i_p, 0], axis = 1)
#
#     ax.plot(N_list, mean_approx_noisy_sol, marker = '^', color = CB_color_cycle[3], label = r'Primal')
#     ax.fill_between(N_list,
#                     (mean_approx_noisy_sol - std_approx_noisy_sol),
#                     (mean_approx_noisy_sol + std_approx_noisy_sol),
#                     color = CB_color_cycle[3], alpha = 0.1)
#
#     mean_approx_noisy_bound_nc = np.mean(approx_ratio_noisy_bound_nc[:, :, i_p, 0], axis = 1)
#     std_approx_noisy_bound_nc = np.std(approx_ratio_noisy_bound_nc[:, :, i_p, 0], axis = 1)
#
#     ax.plot(N_list, mean_approx_noisy_bound_nc, marker = 'D', color = CB_color_cycle[5], label = r'Dual (NC)')
#     ax.fill_between(N_list,
#                     (mean_approx_noisy_bound_nc - std_approx_noisy_bound_nc),
#                     (mean_approx_noisy_bound_nc + std_approx_noisy_bound_nc),
#                     color = CB_color_cycle[5], alpha = 0.1)
#
#     for i_k, k_dual in enumerate([1]):
#         mean_approx_noisy_bound = np.mean(approx_ratio_noisy_bound[:, :, i_p, i_k], axis = 1)
#         std_approx_noisy_bound = np.std(approx_ratio_noisy_bound[:, :, i_p, i_k], axis = 1)
#
#         ax.plot(N_list, mean_approx_noisy_bound, marker = '.', color = CB_color_cycle[i_k], label = r'Dual (k = ' + str(k_dual) + ')')
#         ax.fill_between(N_list,
#                         (mean_approx_noisy_bound - std_approx_noisy_bound),
#                         (mean_approx_noisy_bound + std_approx_noisy_bound),
#                         color = CB_color_cycle[i_k], alpha = 0.1)
#
#     for i_k, k_dual in enumerate([1]):
#         mean_approx_noisy_bound_purity = np.mean(approx_ratio_noisy_bound_purity[:, :, i_p, i_k], axis = 1)
#         std_approx_noisy_bound_purity = np.std(approx_ratio_noisy_bound_purity[:, :, i_p, i_k], axis = 1)
#
#         ax.plot(N_list, mean_approx_noisy_bound_purity, marker = '.', color = CB_color_cycle[i_k + 1], label = r'purity. dual (k = ' + str(k_dual) + ')')
#         ax.fill_between(N_list,
#                         (mean_approx_noisy_bound_purity - std_approx_noisy_bound_purity),
#                         (mean_approx_noisy_bound_purity + std_approx_noisy_bound_purity),
#                         color = CB_color_cycle[i_k + 1], alpha = 0.1)
#
#
#     ax.set_ylabel('Approx. ratio')
#     ax.set_xlabel(r'$N$')
#     ax.legend()
#     ax.set_yscale('log')
#     plt.tight_layout()
#
#     figname = "approx_ratios_" + str(p) + ".pdf"
#     plt.savefig(os.path.join(data_path, figname), format = 'pdf')

# vs d

for i_N, N in enumerate(N_list):
    for i_p, p in enumerate(p_list):
            
        fig = plt.figure()
        ax = fig.add_subplot(111)

        mean_approx_noisy_sol = np.mean(approx_ratio_noisy_sol[i_N, :, :, i_p, 0], axis = 0)
        std_approx_noisy_sol = np.std(approx_ratio_noisy_sol[i_N, :, :, i_p, 0], axis = 0)

        ax.plot(d_list, mean_approx_noisy_sol, marker = '^', color = 'C0', label = r'Primal')
        ax.fill_between(d_list,
                        (mean_approx_noisy_sol - std_approx_noisy_sol),
                        (mean_approx_noisy_sol + std_approx_noisy_sol),
                        color = 'C0', alpha = 0.1)

        mean_approx_noisy_bound_nc = np.mean(approx_ratio_noisy_bound_nc[i_N, :, :, i_p, 0], axis = 0)
        std_approx_noisy_bound_nc = np.std(approx_ratio_noisy_bound_nc[i_N, :, :, i_p, 0], axis = 0)

        ax.plot(d_list, mean_approx_noisy_bound_nc, marker = 'D', color = 'C1', label = r'Dual (NC)')
        ax.fill_between(d_list,
                        (mean_approx_noisy_bound_nc - std_approx_noisy_bound_nc),
                        (mean_approx_noisy_bound_nc + std_approx_noisy_bound_nc),
                        color = 'C1', alpha = 0.1)

        for i_k, k_dual in enumerate(k_dual_list):
            # mean_approx_noisy_bound = np.mean(approx_ratio_noisy_bound[i_N, :, :, i_p, i_k], axis = 0)
            # std_approx_noisy_bound = np.std(approx_ratio_noisy_bound[i_N, :, :, i_p, i_k], axis = 0)

            # ax.plot(d_list, mean_approx_noisy_bound, marker = '.', color = 'C' + str(i_k + 2), ls = "--")
            # ax.fill_between(d_list,
            #                 (mean_approx_noisy_bound - std_approx_noisy_bound),
            #                 (mean_approx_noisy_bound + std_approx_noisy_bound),
            #                 color = 'C' + str(i_k + 2), alpha = 0.1)

            mean_approx_noisy_bound_purity = np.mean(approx_ratio_noisy_bound_purity[i_N, :, :, i_p, i_k], axis = 0)
            std_approx_noisy_bound_purity = np.std(approx_ratio_noisy_bound_purity[i_N, :, :, i_p, i_k], axis = 0)

            ax.plot(d_list, mean_approx_noisy_bound_purity, marker = '.', color = 'C' + str(i_k + 2), label = r'k = ' + str(k_dual), ls = ":")
            ax.fill_between(d_list,
                            (mean_approx_noisy_bound_purity - std_approx_noisy_bound_purity),
                            (mean_approx_noisy_bound_purity + std_approx_noisy_bound_purity),
                            color = 'C' + str(i_k + 2), alpha = 0.1)


        ax.set_ylabel('Approx. ratio')
        ax.set_xlabel(r'd')
        ax.legend()
        # plt.tight_layout()
        ax.set_yscale('log')

        figname = "approx_ratios_N_" + str(N) + "_p_" + str(p) + ".pdf"
        plt.savefig(os.path.join(data_path, figname), format = 'pdf')

        plt.show()


# i_p = 2
# mean_approx_noisy_bound = np.mean(approx_ratio_noisy_bound[:, :, i_p, i_k], axis = 1)
# mean_approx_noisy_sol = np.mean(approx_ratio_noisy_sol[:, :, i_p, 0], axis = 1)

# diff_sol = mean_approx_noisy_sol[0] - mean_approx_noisy_sol[-1]
# diff_bound = mean_approx_noisy_bound[0] - mean_approx_noisy_bound[-1]

# print("diff_sol  = ", diff_sol)
# print("diff_bound  = ", diff_bound)