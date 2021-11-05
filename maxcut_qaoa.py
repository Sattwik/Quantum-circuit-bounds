import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import qutip

from vqa import graphs
from vqa import problems
from vqa import algorithms

lattice = graphs.define_lattice(m = 2, n = 3)
p = 3

num_random_graphs = 10
num_init_states = 10

actual_maxcut_sol = np.zeros((num_random_graphs, num_init_states))
clean_QAOA_maxcut_sol = np.zeros((num_random_graphs, num_init_states))

for n_graph in range(num_random_graphs):

    graph = graphs.create_random_connectivity(lattice)

    maxcut_obj = problems.MaxCut(graph, lattice)

    qaoa_obj = algorithms.QAOA(maxcut_obj, p)

    for n_init in range(num_init_states):

        gamma_init = np.random.uniform(low = 0, high = 2 * np.pi, size = p)
        beta_init = np.random.uniform(low = 0, high = np.pi, size = p)
        gamma_beta_init = np.concatenate((gamma_init, beta_init))

        obj_init = algorithms.objective_external_QAOA(gamma_beta_init, qaoa_obj)
        grad_init = algorithms.gradient_external_QAOA(gamma_beta_init, qaoa_obj)

        obj_over_opti, opt_result = algorithms.optimize_QAOA(gamma_beta_init, qaoa_obj)

        actual_maxcut_sol[n_graph, n_init] = np.min(maxcut_obj.H)
        clean_QAOA_maxcut_sol[n_graph, n_init] = obj_over_opti[-1]
