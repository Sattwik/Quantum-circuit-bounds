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

# gamma = np.array([np.pi/4])
# beta = np.array([np.pi/19])
#
# dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma, beta = beta, p_noise = 0)
# state = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)
#
# state_dm = dual_obj.tensor_2_mat(state.tensor)
#
# psi_after_step = maxcut_obj.init_state()
# psi_after_step = maxcut_obj.Up(gamma[0]) * psi_after_step
# psi_after_step = maxcut_obj.Um(beta[0]) * psi_after_step
#
# rho_after_step = psi_after_step * psi_after_step.dag()
#
# rho = rho_after_step.full()
#
# print(np.linalg.norm(state_dm - rho))

# TEST3: gamma != 0, see if the superoperator unitary works with just two
# sites

gamma0 = np.pi/2
beta0 = np.pi/19

gamma = np.array([gamma0])
beta = np.array([beta0])

lattice = graphs.define_lattice(m = 1, n = 2)
maxcut_obj = problems.MaxCut(lattice, lattice)
p = 1

dual_obj = dual.MaxCutDual(prob_obj = maxcut_obj, p = p, gamma = gamma, beta = beta, p_noise = 0)
state = dual_obj.circuit_layer(layer_num = 0, var_tensor = dual_obj.rho_init_tensor)

state_dm = dual_obj.tensor_2_mat(state.tensor)

psi_after_step = maxcut_obj.init_state()
psi_after_step = maxcut_obj.Up(gamma[0]) * psi_after_step
psi_after_step = maxcut_obj.Um(beta[0]) * psi_after_step

rho_after_step = psi_after_step * psi_after_step.dag()

rho_init_direct = (maxcut_obj.init_state() * maxcut_obj.init_state().dag()).full()

rho = rho_after_step.full()

Hp = -0.5*(qutip.tensor(qutip.qeye(2), qutip.qeye(2)) - qutip.tensor(qutip.sigmaz(), qutip.sigmaz()))
U = (-1j * Hp * gamma0).expm().full()
U_dag = U.conj().T

# (ja jb) (ia ib) -> (ja jb ia ib)
U_tensor = U.flatten()
# (ja jb ia ib) -> (ja) (jb) (ia) (ib)
U_tensor = U_tensor.reshape((2,2,2,2))
# (ja) (jb) (ia) (ib) -> (ja) (ia) (jb) (ib)
U_tensor = U_tensor.transpose([0, 2, 1, 3])

# (iap ibp) (kap kbp) -> (iap ibp kap kbp)
U_dag_tensor = U_dag.flatten()
# (iap ibp kap kbp) -> (iap) (ibp) (kap) (kbp)
U_dag_tensor = U_dag_tensor.reshape((2,2,2,2))
# (iap) (ibp) (kap) (kbp) -> (iap) (kap) (ibp) (kbp)
U_dag_tensor = U_dag_tensor.transpose([0, 2, 1, 3])

# rho_tdot = dual_obj.rho_init_tensor.tensor
# rho_tdot = np.tensordot(U_tensor, rho_tdot, axes = ([1, 3], [0, 2]))
# rho_tdot = rho_tdot.transpose([])
# rho_tdot = np.tensordot(U_dag_tensor, rho_tdot, axes = ([0, 2], [1, 3]))
# rho_tdot = dual_obj.tensor_2_mat(rho_tdot)

print(np.linalg.norm(state_dm - rho))
