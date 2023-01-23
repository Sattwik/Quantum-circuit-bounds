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
        if x == 0:
            graph.remove_edge(*edge)
    return graph

def draw(graph):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    nx.draw(graph, with_labels=True, node_color = "yellow")
    plt.show()

def dist_p_subgraph_from_edge(graph, p, edge):
    neighbors1 = set(nx.dfs_preorder_nodes(graph, edge[0], depth_limit = p))
    neighbors2 = set(nx.dfs_preorder_nodes(graph, edge[1], depth_limit = p))
    neighbors = neighbors1.union(neighbors2)
    return graph.subgraph(neighbors)

def all_dist_p_subgraphs(graph, p):
    subgraphs = []
    for edge in graph.edges:
        subgraphs.append(dist_p_subgraph_from_edge(graph, p, edge))
    return (graph.edges, subgraphs)

# for testing module
if __name__ == "__main__":
    lattice = define_lattice((3,3))
    graph = create_random_connectivity(lattice)
    draw(graph)

    edges, subgraphs = all_dist_p_subgraphs(graph, 1)

    for sg, edge in zip(subgraphs, graph.edges):
        print(edge)
        draw(sg)
