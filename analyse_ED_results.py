import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib

from plotter import set_size


# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rcParams["image.cmap"] = "inferno"
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["xtick.labelsize"] = 8
# matplotlib.rcParams["xtick.labelsize"] = 4
matplotlib.rcParams["ytick.labelsize"] = 8
# matplotlib.rcParams["ytick.labelsize"] = 4
matplotlib.rcParams["font.family"] = "Times New Roman"
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

np.set_printoptions(linewidth=100)

data_path = "./../vqa_data/results_ed/"

N_list = [6, 8]
p_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

for N in N_list:
    for i_p, p in enumerate(p_list):

        width = 510/2
        # width = 100
        fig = plt.figure(figsize=set_size(width, fraction = 1, subplots = (1,1),  height_scale = 0.7))
        ax = fig.add_subplot(111)

        res_local_ansatz = []

        for depth in [2,3,4,5,6]:

            # local ansatz

            with open(data_path + 'local_ansatz/res_size' + str(N) + '_noise' + str(p) + '_depth_' + str(depth)) as f:
                print("Reading file: " + 'res_size' + str(N) + '_noise' + str(p) + '_depth_' + str(depth))
                contents = f.readlines()
                f.close()

            idx = contents[2].find('(')
            exact_result = float(contents[2][idx+1:idx+15])

            idx = contents[4].find('k')
            entropic_bound = float(contents[4][idx+2:idx+16])

            optimization_bound_local = -float(contents[-1])

            # full ansatz

            with open(data_path + 'general_ansatz/res_size' + str(N) + '_noise' + str(p) + '_depth' + str(depth)) as f:
                print("Reading file: " + 'res_size' + str(N) + '_noise' + str(p) + '_depth_' + str(depth))
                contents = f.readlines()
                f.close()

            optimization_bound_full = -float(contents[-1])

            n_iterations = 10*(len(contents) - 7)

            res_local_ansatz.append([2*depth + 2, exact_result, entropic_bound, optimization_bound_local, optimization_bound_full, n_iterations])

        res_local_ansatz = np.array(res_local_ansatz)

        clean_sol = -N
        norm = 2 * N

        ax.plot(res_local_ansatz[:, 0], (res_local_ansatz[:, 1] - clean_sol)/norm, color = 'k', lw = 0.75, marker = '^', markersize = 4, label = 'Output')
        ax.plot(res_local_ansatz[:, 0], (res_local_ansatz[:, 2] - clean_sol)/norm, marker = 'D', color = 'k', label = 'Entropic', markersize = 3, lw = 0.75, ls =  '--')
        ax.plot(res_local_ansatz[:, 0], (res_local_ansatz[:, 3] - clean_sol)/norm, color = 'C1', markersize = 4, lw= 0.75, marker = '.', label = 'Circuit dual (local)')
        ax.plot(res_local_ansatz[:, 0], (res_local_ansatz[:, 4] - clean_sol)/norm, color = 'C2', markersize = 4, lw= 0.75, marker = '.', label = 'Circuit dual (full)')

        ax.set_ylabel('Lower bounds')
        ax.set_xlabel('Circuit depth, ' + r'$d$')
        ax.set_yscale('log')
        # ax.set_ylim(top = 0.2)
        ax.legend()
        plt.tight_layout()
        ax.set_title('Noise rate, ' + r'$p = \ $' + str(p * 100) + '\%')
        # ax.set_yscale('log')

        figname = str(N) + str(i_p) + "_lower_bounds_N_" + str(N) + "_p_" + str(p) + ".pdf"
        plt.savefig(os.path.join(data_path, figname), format = 'pdf', bbox_inches = 'tight')

# res_general_ansatz = []

# for depth in [2, 3, 4, 5, 6]:

#     # general ansatz

#     with open('./general_ansatz/res_size' + str(N) + '_noise' + str(p) + '_depth' + str(depth)) as f:
#         contents = f.readlines()
#         f.close()

#     idx = contents[2].find('(')
#     exact_result = float(contents[2][idx+1:idx+15])

#     idx = contents[4].find('k')
#     entropic_bound = float(contents[4][idx+2:idx+16])

#     optimization_bound = -float(contents[-1])


#     n_iterations = 10*(len(contents) - 7)

#     res_general_ansatz.append([2*depth + 2, exact_result, entropic_bound, optimization_bound, n_iterations])

# res_general_ansatz = np.array(res_general_ansatz)

# plt.plot(res_general_ansatz[:, 0], res_general_ansatz[:, 1], 'oC1')
# plt.plot(res_general_ansatz[:, 0], res_general_ansatz[:, 2], 'oC3')
# plt.plot(res_general_ansatz[:, 0], res_general_ansatz[:, 3], 'oC4')

        


# res_local_circuit_ansatz = []

# for depth in [2, 3, 4, 5, 6]:

#     # general ansatz

#     with open('./local_circuit_ansatz/res_size' + str(N) + '_noise' + str(p) + '_depth_' + str(depth)) as f:
#         contents = f.readlines()
#         f.close()

#     idx = contents[2].find('(')
#     exact_result = float(contents[2][idx+1:idx+15])

#     idx = contents[4].find('k')
#     entropic_bound = float(contents[4][idx+2:idx+16])

#     optimization_bound = -float(contents[-1])


#     n_iterations = 10*(len(contents) - 7)

#     res_local_circuit_ansatz.append([2*depth + 2, exact_result, entropic_bound, optimization_bound, n_iterations])

# res_local_circuit_ansatz = np.array(res_local_circuit_ansatz)

# plt.plot(res_local_circuit_ansatz[:, 0], res_local_circuit_ansatz[:, 1], '*C1')
# plt.plot(res_local_circuit_ansatz[:, 0], res_local_circuit_ansatz[:, 2], '*C3')
# plt.plot(res_local_circuit_ansatz[:, 0], res_local_circuit_ansatz[:, 3], '*C4')
# plt.show()



# for n in range(len([2, 3, 4, 5, 6])):

#     # print('local circuit')
#     # print(res_local_circuit_ansatz[n])

#     print('local')
#     print(res_local_ansatz[n])

#     # print('general')
#     # print(res_general_ansatz[n])
    
#     input()


    
