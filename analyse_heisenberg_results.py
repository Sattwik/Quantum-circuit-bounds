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

data_path = "../vqa_data/heisenberg_tests/"

fname = "...?aSf/asefas"

num_sites = 4

p = 0.1
theta = 0.2

d_list = list(range(15))
D_list = [2, 6, 10, 16]


N = num_sites
clean_sol = -N
norm = 2 * N

fig = plt.figure(figsize=(3.5284350352843505, 2.469904524699045))
ax = fig.add_subplot(111)

# ax.axhline(y = -num_sites, color = 'k', label = "GSE", ls = "--")
for i_D, D in enumerate(D_list):
    if i_D >= 1:
        ax.plot([1 + 2 * d for d in d_list], (heis_bounds[:, i_D] - clean_sol)/norm, ls = "--", 
                label = "Heis.,  D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
                marker = 'D', markersize = 3)
        ax.plot([1 + 2 * d for d in d_list], (dual_bounds[:, i_D] - clean_sol)/norm, 
                label = "Dual, D = " + str(D), color = 'C' + str(i_D), lw = 0.75, 
                marker = '.', markersize = 4)

ax.plot([1 + 2 * d for d in d_list], (outputs - clean_sol)/norm, label = "Output", 
        color = 'k', ls = ":")
ax.set_yscale('log')
ax.legend()
ax.set_title("N = " + str(N) + r", $\theta$ = " + f'{theta:.2f}' + r"$p$ = " + str(p))
plt.tight_layout()
figname = "heis_test_N_" + str(num_sites) + "_p_" + str(p) + "_theta_" + f'{theta:.2f}' + ".pdf"
plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')
