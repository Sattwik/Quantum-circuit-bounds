import pickle
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import qutip

from vqa import graphs
from vqa import problems
from vqa import algorithms

m = 2
n = 3
lattice = graphs.define_lattice(m = m, n = n)

p = 3

num_random_graphs = 5
num_init_states = 5
num_monte_carlo = 5
num_noise_probs = 3
p_noise_list = [0.001, 0.01, 0.1]

# Save file
today = date.today()
mmdd =  today.strftime("%m%d%y")[:4]

data_folder_path = os.path.join('./../vqa_data', mmdd)

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

now = datetime.now().strftime("%Y%m%d-%H%M%S")
result_save_path = os.path.join(data_folder_path, now)
file_name = "qaoa_maxcut_exact.pkl"

actual_maxcut_sol = np.zeros((num_random_graphs, num_init_states))
clean_QAOA_maxcut_sol = np.zeros((num_random_graphs, num_init_states))
noisy_QAOA_maxcut_sol = np.zeros((num_noise_probs, num_random_graphs, num_init_states))

for n_graph in range(num_random_graphs):

    graph = graphs.create_random_connectivity(lattice)

    maxcut_obj = problems.MaxCut(graph, lattice)

    mc_probs = np.zeros(graph.number_of_nodes())
    p_noise = 0
    qaoa_obj = algorithms.QAOA(maxcut_obj, p, p_noise, mc_probs)

    for n_init in range(num_init_states):

        gamma_init = np.random.uniform(low = 0, high = 2 * np.pi, size = p)
        beta_init = np.random.uniform(low = 0, high = np.pi, size = p)
        gamma_beta_init = np.concatenate((gamma_init, beta_init))

        obj_over_opti, opt_result = algorithms.optimize_QAOA(gamma_beta_init, qaoa_obj)

        actual_maxcut_sol[n_graph, n_init] = np.min(maxcut_obj.H)
        clean_QAOA_maxcut_sol[n_graph, n_init] = obj_over_opti[-1]

        noisy_sol = 0

        for n_noise, p_noise in enumerate(p_noise_list):

            for n_mc in range(num_monte_carlo):

                mc_probs = np.random.uniform(low = 0, high = 1, size = (p, graph.number_of_nodes()))
                qaoa_obj_noisy = algorithms.QAOA(maxcut_obj, p, p_noise, mc_probs, noisy = True)

                obj_over_opti_noisy, opt_result_noisy = \
                        algorithms.optimize_QAOA(gamma_beta_init, qaoa_obj_noisy)

                noisy_sol += obj_over_opti_noisy[-1]/num_monte_carlo

            noisy_QAOA_maxcut_sol[n_noise, n_graph, n_init] = noisy_sol

data_list = [m, n, p, p_noise_list,
             actual_maxcut_sol, clean_QAOA_maxcut_sol, noisy_QAOA_maxcut_sol]

with open(os.path.join(result_save_path, file_name), "wb") as f_for_pkl:
    pickle.dump(data_list, f_for_pkl)
