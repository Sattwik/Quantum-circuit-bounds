import pickle
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# from plotter import set_size
from matplotlib import rc
import matplotlib

matplotlib.rcParams["image.cmap"] = "inferno"
matplotlib.rcParams["axes.titlesize"] = 8
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 7
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["xtick.labelsize"] = 6
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["font.family"] = "Times New Roman"
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


data_path = "../vqa_data/0508/20230508-123611/"
# data_path = "20230508-123611/"
N_list = [32]
p_list = np.linspace(0.03, 0.3, 10)
theta_list = np.linspace(0.01, 1.50, 11)
d_list = [6, 12, 20]
D_list = [16, 32, 64]

num_D = len(D_list)
num_N = len(N_list)
num_p = len(p_list)
num_theta = len(theta_list)
num_seeds = 1
num_d = len(d_list)

dual_bound_list = np.zeros((num_N, num_seeds, num_p, num_D, num_theta, num_d), dtype = complex)
heis_bound_list = np.zeros((num_N, num_seeds, num_p, num_D, num_theta, num_d), dtype = complex)
heis_val_list = np.zeros((num_N, num_seeds, num_p, num_D, num_theta, num_d), dtype = complex)

for i_N, N in enumerate(N_list):
    for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
        for i_p, p in enumerate(p_list):
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

clipped_heis_bound_list = np.clip(scaled_heis_bound_list, a_min = 0, a_max = None)
clipped_dual_bound_list = np.clip(scaled_dual_bound_list, a_min = 0, a_max = None)
clipped_heis_val_list = np.clip(scaled_heis_val_list, a_min = 0, a_max = None)

for i_N, N in enumerate(N_list):
    for i_seed, seed in enumerate(N + np.array(range(num_seeds))):
        for i_d, d in enumerate(d_list):
            for i_D, D in enumerate(D_list):
            
                fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
                ax = fig.add_subplot(111)
                # p vertical, theta horizontal
                # img = ax.imshow(np.real(scaled_dual_bound_list[i_N, i_seed, :, i_D, :, i_d]), 
                #                 origin = 'lower', interpolation='none', rasterized = True, aspect = 'auto')
                # img = ax.imshow(np.real(scaled_dual_bound_list[i_N, i_seed, :, i_D, :, i_d]), 
                #                 origin = 'lower', interpolation='none', rasterized = True, 
                #                 aspect = 'auto', norm='log', vmin = 1e-2, vmax = 0.55)
                # img = ax.imshow(np.real(scaled_dual_bound_list[i_N, i_seed, :, i_D, :, i_d]), 
                #                 origin = 'lower', interpolation='none', rasterized = True, 
                #                 aspect = 'auto', vmin = 5e-3, vmax = 0.55)
                img = ax.imshow(np.real(clipped_dual_bound_list[i_N, i_seed, :, i_D, :, i_d]), 
                                origin = 'lower', interpolation='none', rasterized = True, 
                                aspect = 'auto', vmin = 0, vmax = 0.55)
                ax.set_xticks(np.arange(len(theta_list)), labels=[f'{theta/np.pi:.2f}' for theta in theta_list])
                ax.set_yticks(np.arange(len(p_list)), labels=[f'{p:.2f}' for p in p_list])
                ax.set_ylabel(r'$p$')
                ax.set_xlabel(r'$\theta/ \pi$')
                # ax.set_title("Dual, N = " + str(N) + r", $d$ = " + str(d) + r",\ $D$ = " + str(D))
                ax.set_title("Bond dimension" r",\ $D$ = " + str(D))
                fig.colorbar(img)
                plt.tight_layout()
                figname = "0_dual_N_" + str(N) + "_d_" + str(d) + "_D_" + str(D) + ".pdf"
                plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
                plt.close()

                fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
                ax = fig.add_subplot(111)
                # img = ax.imshow(np.real(scaled_heis_bound_list[i_N, i_seed, :, i_D, :, i_d]), 
                #                 origin = 'lower', interpolation= 'none', rasterized = True, aspect = 'auto')
                # img = ax.imshow(np.real(scaled_heis_bound_list[i_N, i_seed, :, i_D, :, i_d]), 
                #                 origin = 'lower', interpolation= 'none', rasterized = True, 
                #                 aspect = 'auto', norm='log', vmin = 1e-2, vmax = 0.55)
                # img = ax.imshow(np.real(scaled_heis_bound_list[i_N, i_seed, :, i_D, :, i_d]), 
                #                 origin = 'lower', interpolation= 'none', rasterized = True, 
                #                 aspect = 'auto', vmin = 5e-3, vmax = 0.55)
                img = ax.imshow(np.real(clipped_heis_bound_list[i_N, i_seed, :, i_D, :, i_d]), 
                                origin = 'lower', interpolation= 'none', rasterized = True, 
                                aspect = 'auto', vmin = 0, vmax = 0.55)
                ax.set_xticks(np.arange(len(theta_list)), labels=[f'{theta/np.pi:.2f}' for theta in theta_list])
                ax.set_yticks(np.arange(len(p_list)), labels=[f'{p:.2f}' for p in p_list])
                ax.set_ylabel(r'$p$')
                ax.set_xlabel(r'$\theta / \pi$')
                ax.set_title("Heis. bound, N = " + str(N) + r", $d$ = " + str(d) + r",\ $D$ = " + str(D))
                fig.colorbar(img)
                plt.tight_layout()
                figname = "1_heisbound_N_" + str(N) + "_d_" + str(d) + "_D_" + str(D) + ".pdf"
                plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
                plt.close()

                fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
                ax = fig.add_subplot(111)
                # img = ax.imshow(np.real(scaled_heis_val_list[i_N, i_seed, :, i_D, :, i_d]), 
                #                 origin = 'lower', interpolation= 'none', rasterized = True, aspect = 'auto')
                # img = ax.imshow(np.real(scaled_heis_val_list[i_N, i_seed, :, i_D, :, i_d]), 
                #                 origin = 'lower', interpolation= 'none', rasterized = True, 
                #                 aspect = 'auto', norm='log', vmin = 1e-2, vmax = 0.55)
                # img = ax.imshow(np.real(scaled_heis_val_list[i_N, i_seed, :, i_D, :, i_d]), 
                #                 origin = 'lower', interpolation= 'none', rasterized = True, 
                #                 aspect = 'auto', vmin = 5e-3, vmax = 0.55)
                img = ax.imshow(np.real(clipped_heis_val_list[i_N, i_seed, :, i_D, :, i_d]), 
                                origin = 'lower', interpolation= 'none', rasterized = True, 
                                aspect = 'auto', vmin = 0, vmax = 0.55)
                ax.set_xticks(np.arange(len(theta_list)), labels=[f'{theta/np.pi:.2f}' for theta in theta_list])
                ax.set_yticks(np.arange(len(p_list)), labels=[f'{p:.2f}' for p in p_list])
                ax.set_ylabel(r'$p$')
                ax.set_xlabel(r'$\theta/ \pi$')
                ax.set_title("Heis. val, N = " + str(N) + r", $d$ = " + str(d) + r",\ $D$ = " + str(D))
                fig.colorbar(img)
                plt.tight_layout()
                figname = "2_heisval_N_" + str(N) + "_d_" + str(d) + "_D_" + str(D) + ".pdf"
                plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
                plt.close()
