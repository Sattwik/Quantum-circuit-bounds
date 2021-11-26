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
import tensornetwork as tn

from vqa import graphs
from vqa import problems
from vqa import algorithms
from vqa import dual

m = 2
n = 2
lattice = graphs.define_lattice(m = m, n = n)

graph = graphs.create_random_connectivity(lattice)

maxcut_obj = problems.MaxCut(graph, lattice)

p = 1

# TEST1: gamma = 0, beta = 0 in the circuit layer should not change the init state
# PASSED
# gamma = np.array([0])
# beta = np.array([0])
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma, beta = beta, p_noise = 0)
#
# state = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)
#
# print(np.linalg.norm(state.tensor - dual_obj.rho_init_tensor.tensor))

# TEST2: gamma != 0, beta != 0 in the circuit layer should match with
# direct computation
# FAILED: error in problem unitary in dual

gamma = np.array([np.pi/4])
beta = np.array([0])

dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma, beta = beta, p_noise = 0)
state = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)

state_dm = np.reshape(state.tensor, (dual_obj.dim, dual_obj.dim))

psi_after_step = maxcut_obj.init_state()
psi_after_step = maxcut_obj.Up(gamma[0]) * psi_after_step
psi_after_step = maxcut_obj.Um(beta[0]) * psi_after_step

rho_after_step = psi_after_step * psi_after_step.dag()

rho = rho_after_step.full()

print(np.linalg.norm(state_dm - rho))

# TEST3: gamma != 0, see if the superoperator unitary works with just two
# sites

gamma = np.pi/4

Hp = -0.5 * (qutip.tensor(qutip.qeye(2), qutip.qeye(2)) - qutip.tensor(qutip.sigmaz(), qutip.sigmaz()))

U = np.diag([1, np.exp(1j * gamma), np.exp(1j * gamma), 1])

U_direct = (-1j * gamma * Hp).expm()

state_init = qutip.rand_ket(4, dims = [[2, 2], [1, 1]])
rho_init = (state_init * state_init.dag()).full()
rho_init_tensor = tn.Node(rho_init)
U_node = tn.Node(dual_obj.reshapeU_2site(U))

new_edge_j = U_node[2] ^ rho_init_tensor[0]
new_edge_k = U_node[3] ^ rho_init_tensor[1]

rho_tensor = U_node @ rho_init_tensor

state_direct = U_direct * state_init
rho_direct = (state_direct * state_direct.dag()).full()

print(np.linalg.norm(rho_direct - rho_tensor.tensor))
