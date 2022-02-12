from typing import List, Tuple, Callable, Dict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def define_lattice(dims: List):

    return nx.grid_graph(dims)

def create_random_connectivity(nx_lattice, p = 0.7):

    graph = nx_lattice.copy()

    for edge in nx_lattice.edges:

        x = np.random.binomial(1, p, size=None)

        # print(edge)
        # print(x)

        if x == 0:
            graph.remove_edge(*edge)

    return graph

def draw(graph):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    nx.draw(graph, with_labels=True)
    plt.show()
