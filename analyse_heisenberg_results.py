import pickle
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# from plotter import set_size
from matplotlib import rc
import matplotlib
from matplotlib.lines import Line2D

matplotlib.rcParams["image.cmap"] = "inferno"
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 7
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["font.family"] = "Times New Roman"
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

# data_path = "../vqa_data/0508/20230508-092949/"

# N_list = [16,]
# p_list = [0.03, 0.1, 0.3]
# d_list = [8, 10, 12, 20]
# D_list = [2, 8, 32, 64, 128]
# theta_list = [0.03, 0.1, 1.0]

# data_path = "../vqa_data/0508/20230508-101016/"
# N_list = [32]
# p_list = [0.03, 0.1, 0.3]
# d_list = [4, 6, 8, 10, 12, 20, 24]
# D_list = [2, 8, 16, 32, 64,]
# theta_list = [0.01, 0.1, 1.0]

# data_path = "../vqa_data/0508/20230508-143412/"
# N_list = [32]
# p_list = np.linspace(0.03, 0.3, 10)
# theta_list = [0.5]
# d_list = np.array(np.linspace(4, 24, 11), dtype = int)
# D_list = [16, 32, 64]

# data_path = "../vqa_data/0509/20230509-144732/"
# data_path = "./20230509-144732/"
# N_list = [32]
# p_list = np.linspace(0.03, 0.3, 5)
# p_list = np.append(p_list, 0.05)
# p_list = np.sort(p_list)
# theta_list = [0.159]
# d_list = np.array(np.linspace(4, 24, 11), dtype = int)
# D_list = [16, 32, 64]

# data_path = "../vqa_data/0510/20230510-143449/"
# N_list = [32]
# # p_list = [0.03, 0.1, 0.3]
# # p_list = np.linspace(0.03, 0.3, 5)
# p_list = [0.05]
# theta_list = [0.159]
# # [0.01, 0.1, 1.0]
# d_list = np.array(np.linspace(4, 24, 11), dtype = int)
# D_list = [16, 32, 64]

# data_path = "../vqa_data/0510/20230510-183410/"
# N_list = [32]
# # p_list = [0.03, 0.1, 0.3]
# # p_list = np.linspace(0.03, 0.3, 5)
# p_list = [0.03]
# theta_list = [0.159]
# # [0.01, 0.1, 1.0]
# d_list = np.concatenate((np.array(np.linspace(4, 24, 11), dtype = int), np.array(np.linspace(24, 240, 11), dtype = int)))
# D_list = [32]

data_path = "../vqa_data/0510/20230510-145039/"
N_list = [32]
# p_list = [0.03, 0.1, 0.3]
p_list = np.linspace(0.03, 0.3, 10)
theta_list = [0.01, 0.05, 0.1, 0.159]
d_list = np.array(np.linspace(4, 24, 11), dtype = int)
D_list = [16, 24, 32, 48, 64]

num_D = len(D_list)
num_N = len(N_list)
num_p = len(p_list)
num_theta = len(theta_list)
num_seeds = 1
num_d = len(d_list)

# clean_sol_list = np.zeros((num_N, num_seeds, num_p, num_k_duals), dtype = float)
# noisy_sol_list = np.zeros((num_N, num_seeds, num_p, num_k_duals), dtype = float)
# noisy_bound_list = np.zeros((num_N, num_seeds, num_p, num_k_duals), dtype = float)
# noisy_bound_proj_list = np.zeros((num_N, num_seeds, num_p, num_k_duals), dtype = float)
# noisy_bound_nc_list = np.zeros((num_N, num_seeds, num_p, num_k_duals), dtype = float)


dual_bound_list = np.zeros((num_N, num_seeds, num_p, num_D, num_theta, num_d), dtype = complex)
heis_bound_list = np.zeros((num_N, num_seeds, num_p, num_D, num_theta, num_d), dtype = complex)
heis_val_list = np.zeros((num_N, num_seeds, num_p, num_D, num_theta, num_d), dtype = complex)

for i_N, N in enumerate(N_list):
    for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
        for i_p, p in enumerate(p_list):
            # for i_k, k_dual in enumerate([1, N]):
            for i_D, D in enumerate(D_list):
                for i_d, d in enumerate(d_list):
                     for i_theta, theta in enumerate(theta_list):
                        fname = "heis1D-N-" + str(N) + "-d-" + str(d) + "-seed-" + \
                                str(seed) + "-theta-" + f'{theta:.4f}' + \
                                "-p-" + str(p) + "-D-" + str(D) + ".pkl"
                        with open(os.path.join(data_path, fname), 'rb') as result_file:
                            print("Reading file: " + os.path.join(data_path, fname))

                            heis_val, heis_bound, dual_bound = pickle.load(result_file)

                            dual_bound_list[i_N, i_seed, i_p, i_D, i_theta, i_d] = dual_bound
                            heis_bound_list[i_N, i_seed, i_p, i_D, i_theta, i_d] = heis_bound
                            heis_val_list[i_N, i_seed, i_p, i_D, i_theta, i_d] = heis_val

heis_bound_list = np.array(heis_bound_list)
dual_bound_list = np.array(dual_bound_list)
heis_val_list = np.array(heis_val_list)

clean_sol = -N
norm = 2 * N

scaled_heis_bound_list = (heis_bound_list - clean_sol)/norm
scaled_dual_bound_list = (dual_bound_list - clean_sol)/norm
scaled_heis_val_list = (heis_val_list - clean_sol)/norm

# bound vs depth plots

for i_N, N in enumerate(N_list):
    for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
        for i_p, p in enumerate(p_list):

            entropic_bound_list = []
            with open(os.path.join(data_path, "entropic_bound_noise_bounded_temp_" + str(p) + ".npy"), 'rb') as f:
                entropic_bound_list = np.load(f)
                
            for i_theta, theta in enumerate(theta_list):
                fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
                ax = fig.add_subplot(111)

                ax.axhline(y = 0, ls = '--', color = 'gray', lw = 0.75)

                ax.plot([1 + 2 * d for d in d_list], (entropic_bound_list - clean_sol)/norm, 
                            ls = "--", 
                            label = "Entropic", color = 'k', lw = 0.75, 
                            marker = '^', markersize = 3)

                legend_elements_1 = []

                for i_D, D in enumerate(D_list):
                    ax.plot([1 + 2 * d for d in d_list], (dual_bound_list[i_N, i_seed, i_p, i_D, i_theta, :] - clean_sol)/norm, 
                            label = "Dual, D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
                            marker = '.', markersize = 4)
                    # ax.plot([1 + 2 * d for d in d_list], (heis_bound_list[i_N, i_seed, i_p, i_D, i_theta, :] - clean_sol)/norm, 
                    #         ls = ":", 
                    #         label = "Heis.,  D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
                    #         marker = '+', markersize = 3)

                    legend_elements_1.append(Line2D([0], [0], color='C' + str(i_D), lw=0.75, label=r'$D = \ ' + str(D) + '$'))

                legend_elements_2 = [Line2D([0], [0], marker='.', color='w', label='Dual', markerfacecolor='k'),
                             Line2D([0], [0], marker='+', color='w', label='Heisenberg', markerfacecolor='k'),
                             Line2D([0], [0], marker='^', color='w', label='Entropic', markerfacecolor='k')]
                
                legend_elements_2 = [Line2D([0], [0], marker='.', color='w', label='Dual', markerfacecolor='k'),
                             Line2D([0], [0], marker='^', color='w', label='Entropic', markerfacecolor='k')]

                first_legend = ax.legend(handles = legend_elements_1, loc='lower right', frameon = True)
                ax.legend(handles = legend_elements_2, loc='lower left', bbox_to_anchor = (0.7, 0.12), frameon = True)
                ax.add_artist(first_legend)

                ax.set_ylim(bottom = 0.0, top = 0.8)
                ax.set_ylabel('Lower bounds')
                ax.set_xlabel('Circuit depth, ' + r'$d$')
                # ax.set_yscale('log')
                # ax.legend()
                ax.set_title("N = " + str(N) + r", $\theta$ = " + f'{theta:.2f}' + r", \ $p$ = " + str(p))
                plt.tight_layout()
                figname = str(i_theta) + str(i_p) + "_heis_test_N_" + str(N) + "_p_" + str(p) + "_theta_" + f'{theta:.2f}' + ".pdf"
                plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
                plt.close()

# bound vs p plots

# i_d = 4
# d = d_list[i_d]

# entropic_bound_vs_p = []

# for p in p_list:
#     with open(os.path.join(data_path, "entropic_bound_noise_bounded_temp_" + str(p) + ".npy"), 'rb') as f:
#         entropic_bound_list = np.load(f)
#     entropic_bound_vs_p.append(entropic_bound_list[i_d])
# entropic_bound_vs_p = np.array(entropic_bound_vs_p)

# for i_N, N in enumerate(N_list):
#     for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
#         for i_theta, theta in enumerate(theta_list):
#             for i_D, D in enumerate(D_list):
            
#                 fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
#                 ax = fig.add_subplot(111)

#                 ax.plot(p_list, (entropic_bound_vs_p - clean_sol)/norm, 
#                             ls = "--", 
#                             label = "Entropic", color = 'k', lw = 0.75, 
#                             marker = '^', markersize = 3)

#                 ax.plot(p_list, (dual_bound_list[i_N, i_seed, :, i_D, i_theta, i_d] - clean_sol)/norm, 
#                         label = "Dual, D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
#                         marker = '.', markersize = 4)
#                 ax.plot(p_list, (heis_bound_list[i_N, i_seed, :, i_D, i_theta, i_d] - clean_sol)/norm, 
#                         ls = ":", 
#                         label = "Heis.,  D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
#                         marker = '+', markersize = 3)

#                 legend_elements_2 = [Line2D([0], [0], marker='.', color='w', label='Dual', markerfacecolor='k'),
#                              Line2D([0], [0], marker='+', color='w', label='Heisenberg', markerfacecolor='k'),
#                              Line2D([0], [0], marker='^', color='w', label='Entropic', markerfacecolor='k')]

#                 # first_legend = ax.legend(handles = legend_elements_1, loc='lower left', frameon = True)
#                 ax.legend(handles = legend_elements_2, loc='lower left', bbox_to_anchor = (0.27, 0), frameon = True)
#                 # ax.add_artist(first_legend)

#                 ax.set_ylim(bottom = 0.0, top = 0.55)
#                 ax.set_ylabel('Lower bounds')
#                 ax.set_xlabel('Noise rate, ' + r'$p$')
#                 # ax.set_yscale('log')
#                 # ax.legend()
#                 ax.set_title("N = " + str(N) + r", $\theta$ = " + f'{theta:.2f}' + r", \ $D$ = " + str(D))
#                 plt.tight_layout()
#                 figname = str(i_theta + 1) + str(i_D) + "_heis_test_N_" + str(N) + "_D_" + str(D) + "_theta_" + f'{theta:.2f}' + ".pdf"
#                 plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
#                 plt.close()

# bound vs depth plot variants

# for i_N, N in enumerate(N_list):
#     for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
#         for i_p, p in enumerate(p_list):
#             for i_theta, theta in enumerate(theta_list):
#                 for i_D, D in enumerate(D_list):
#                     fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
#                     ax = fig.add_subplot(111)
                        
#                     ax.plot([1 + 2 * d for d in d_list], scaled_dual_bound_list[i_N, i_seed, i_p, i_D, i_theta, :], 
#                             label = "Dual, D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
#                             marker = '.', markersize = 4)
#                     ax.plot([1 + 2 * d for d in d_list], scaled_heis_bound_list[i_N, i_seed, i_p, i_D, i_theta, :], ls = "--", 
#                             label = "Heis. bound,  D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
#                             marker = 'D', markersize = 3)
#                     ax.plot([1 + 2 * d for d in d_list], scaled_heis_val_list[i_N, i_seed, i_p, i_D, i_theta, :], ls = "--", 
#                             label = "Heis. val,  D = " + str(D), color = 'k', lw = 0.75, 
#                             marker = '^', markersize = 3)
                    
#                     ax.set_ylim(bottom = 0.0, top = 0.51)
#                     # ax.set_yscale('log')
#                     ax.legend()
#                     ax.set_title("N = " + str(N) + r", $\theta$ = " + f'{theta:.2f}' + r", \ $p$ = " + str(p) + r' $D = $' + str(D))
#                     plt.tight_layout()
#                     figname = str(i_p) + str(i_theta) + str(i_D) + "_heis_test_N_" + str(N) + "_p_" + str(p) + "_theta_" + f'{theta:.2f}' + "_D_" + str(D) +".pdf"
#                     plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
#                     plt.close()

# for i_N, N in enumerate(N_list):
#     for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
#         for i_p, p in enumerate(p_list):
#             for i_theta, theta in enumerate(theta_list):
#                 fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
#                 ax = fig.add_subplot(111)
#                 for i_D, D in enumerate(D_list):
#                     ax.plot([1 + 2 * d for d in d_list], dual_bound_list[i_N, i_seed, i_p, i_D, i_theta, :], 
#                             label = "Dual, D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
#                             marker = '.', markersize = 4)
#                     ax.plot([1 + 2 * d for d in d_list], heis_bound_list[i_N, i_seed, i_p, i_D, i_theta, :], ls = "--", 
#                             label = "Heis.,  D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
#                             marker = 'D', markersize = 3)

#                 ax.legend()
#                 ax.set_title("N = " + str(N) + r", $\theta$ = " + f'{theta:.2f}' + r", \ $p$ = " + str(p))
#                 plt.tight_layout()
#                 figname = "unscaled_" + str(i_p) + str(i_theta) + "_heis_test_N_" + str(N) + "_p_" + str(p) + "_theta_" + f'{theta:.2f}' + ".pdf"
#                 plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
#                 plt.close()


# for i_N, N in enumerate(N_list):
#     for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
#         for i_p, p in enumerate(p_list):
#             for i_theta, theta in enumerate(theta_list):
#                 for i_D, D in enumerate(D_list):
#                     fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
#                     ax = fig.add_subplot(111)
                        
#                     ax.plot([1 + 2 * d for d in d_list], dual_bound_list[i_N, i_seed, i_p, i_D, i_theta, :], 
#                             label = "Dual, D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
#                             marker = '.', markersize = 4)
#                     ax.plot([1 + 2 * d for d in d_list], heis_bound_list[i_N, i_seed, i_p, i_D, i_theta, :], ls = "--", 
#                             label = "Heis. bound,  D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
#                             marker = 'D', markersize = 3)
#                     ax.plot([1 + 2 * d for d in d_list], heis_val_list[i_N, i_seed, i_p, i_D, i_theta, :], ls = "--", 
#                             label = "Heis. val,  D = " + str(D), color = 'k', lw = 0.75, 
#                             marker = '^', markersize = 3)

#                     ax.legend()
#                     ax.set_title("N = " + str(N) + r", $\theta$ = " + f'{theta:.2f}' + r", \ $p$ = " + str(p) + r' $D = $' + str(D))
#                     plt.tight_layout()
#                     figname = "unscaled_" + str(i_p) + str(i_theta) + str(i_D) + "_heis_test_N_" + str(N) + "_p_" + str(p) + "_theta_" + f'{theta:.2f}' + "_D_" + str(D) +".pdf"
#                     plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
#                     plt.close()