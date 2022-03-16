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
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.test_util import check_grads
import scipy

from vqa_bounds import maxcut1D, graphs, circuit_utils, dual_utils

m = 6
lattice = graphs.define_lattice((m,))

graph = graphs.create_random_connectivity(lattice)

d = 3

sys_obj = maxcut1D.MaxCut1D(graph, lattice, d, p = 0)

gamma_init = np.ones(d)
beta_init = np.ones(d//2)

circuit_params_init = np.concatenate((gamma_init, beta_init))

ub_array = np.concatenate((np.ones(d) * 2 * np.pi, np.ones(d//2) * np.pi))
bnds = scipy.optimize.Bounds(lb = 0, ub = ub_array)

obj_over_opti, opt_result = circuit_utils.optimize_circuit(circuit_params_init, bnds, sys_obj)

# sys_obj.update_opt_circ_params(np.concatenate((gamma, beta)))
#
# state = sys_obj.circuit_layer(layer_num = 0, var_tensor = sys_obj.rho_init_tensor)
# state = sys_obj.circuit_layer(layer_num = 1, var_tensor = state)
#
# print(np.linalg.norm(state.tensor - sys_obj.rho_init_tensor.tensor))
