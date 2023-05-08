import pickle
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# from plotter import set_size
from matplotlib import rc
import matplotlib

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

data_path = "../vqa_data/0508/20230508-025157/"

N_list = [16,]
p_list = [0.03, 0.1, 0.3]
d_list = [8, 10, 12, 20]
D_list = [2, 8, 32, 64, 128]
theta_list = [0.03, 0.1, 1.0]

# data_path = "../vqa_data/0508/20230508-032044/"
# N_list = [32]
# p_list = [0.03, 0.1, 0.3]
# d_list = [4, 6, 8, 10, 12, 20, 24]
# D_list = [2, 8, 16, 32, 64, 128, 256]
# theta_list = [0.01, 0.1, 1.0]

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
            i_theta = i_p
            theta = p
            # for i_k, k_dual in enumerate([1, N]):
            for i_D, D in enumerate(D_list):
                for i_d, d in enumerate(d_list):
                    #  for i_theta, theta in enumerate(theta_list):
                    # for i_theta, theta in enumerate(p_list):

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

# ax.axhline(y = -num_sites, color = 'k', label = "GSE", ls = "--")
for i_N, N in enumerate(N_list):
    for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
        for i_p, p in enumerate(p_list):
            i_theta = i_p
            theta = p
            # for i_theta, theta in enumerate(theta_list):

            fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
            ax = fig.add_subplot(111)

            for i_D, D in enumerate(D_list):
                if i_D >= 2:
                    # ax.plot([1 + 2 * d for d in d_list], (dual_bound_list[i_N, i_seed, i_p, i_D, i_theta, :] - clean_sol)/norm, 
                    #         label = "Dual, D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
                    #         marker = '.', markersize = 4)
                    # ax.plot([1 + 2 * d for d in d_list], (heis_bound_list[i_N, i_seed, i_p, i_D, i_theta, :] - clean_sol)/norm, ls = "--", 
                    #         label = "Heis.,  D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
                    #         marker = 'D', markersize = 3)
                    
                    ax.plot([1 + 2 * d for d in d_list], dual_bound_list[i_N, i_seed, i_p, i_D, i_theta, :], 
                            label = "Dual, D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
                            marker = '.', markersize = 4)
                    ax.plot([1 + 2 * d for d in d_list], heis_bound_list[i_N, i_seed, i_p, i_D, i_theta, :], ls = "--", 
                            label = "Heis.,  D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
                            marker = 'D', markersize = 3)

            # ax.plot([1 + 2 * d for d in d_list], (outputs - clean_sol)/norm, label = "Output", 
            #         color = 'k', ls = ":")
            # ax.set_yscale('log')
            ax.legend()
            ax.set_title("N = " + str(N) + r", $\theta$ = " + f'{theta:.2f}' + r"$p$ = " + str(p))
            plt.tight_layout()
            figname = "heis_test_N_" + str(N) + "_p_" + str(p) + "_theta_" + f'{theta:.2f}' + ".pdf"
            plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
