import abc
from typing import List, Tuple, Callable, Dict
from functools import partial
import copy
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.config import config
import tensornetwork as tn
tn.set_default_backend("jax")
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
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

from vqa_bounds import mpo

num_sites = 4

#-------------------------------------------#
#---------- error dynamics test ------------#
#-------------------------------------------#

# d = 15
# p = 0.01
# theta = np.pi/7
# seed = 69

# circ = mpo.SumZ_RXX(num_sites, d, p, theta, seed)

# primal1 = circ.primal_noisy()
# print(primal1)

# init_mpo_tensors = circ.init_mpo(D = 2 ** num_sites)

# primal2 = mpo.trace_two_MPOs(circ.psi_init_tensors, init_mpo_tensors)

# print(np.abs(primal1 - primal2))


# # error_list = circ.error_dynamics(D = 5)
# # plt.plot(list(range(circ.depth))[::-1], error_list)
# # plt.show()

# heis_bound, dual_bound = circ.bounds(D = 2 ** num_sites)

p = 0.1
theta = 0.2

d_list = list(range(15))
D_list = [2, 6, 10, 16]

heis_bounds = np.zeros((len(d_list), len(D_list)), dtype = complex)
dual_bounds = np.zeros((len(d_list), len(D_list)), dtype = complex)
heis_vals = np.zeros((len(d_list), len(D_list)), dtype = complex)
outputs = np.zeros((len(d_list),), dtype = complex)

for i_d, d in enumerate(d_list):

    print('d = ', d)
    seed = np.random.randint(low = 1, high = 100)
    circ = mpo.SumZ_RXX(num_sites, d, p, theta, seed)
    outputs[i_d] = complex(circ.primal_noisy())

    for i_D, D in enumerate(D_list):
        print('D = ', d)
        hval, hb, db = circ.bounds(D = D)

        # TODO: check if imaginary parts zero
        heis_vals[i_d, i_D] = complex(hval)
        heis_bounds[i_d, i_D] = complex(hb)
        dual_bounds[i_d, i_D] = complex(db)

data_path = "../vqa_data/heisenberg_tests/"

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

#-------------------------------------------#
#------------- circuit test ----------------#
#-------------------------------------------#

# d = 3
# p = 0.0
# theta = np.pi/7
# seed = 69

# circ = mpo.SumZ_RXX(num_sites, d, p, theta, seed)

# print(circ.primal_noisy())

# primal1 = circ.primal_noisy()

# init_mpo_tensors = circ.init_mpo(D = 10)

# primal2 = mpo.trace_two_MPOs(circ.psi_init_tensors, init_mpo_tensors)

# print(np.abs(primal1 - primal2))

#-------------------------------------------#
#--------------- Haar test -----------------#
#-------------------------------------------#

# def plot_bloch_sphere(bloch_vectors):
#     """ Helper function to plot vectors on a sphere."""
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

#     ax.grid(False)
#     ax.set_axis_off()
#     ax.view_init(30, 45)
#     ax.dist = 7

#     # Draw the axes (source: https://github.com/matplotlib/matplotlib/issues/13575)
#     x, y, z = np.array([[-1.5,0,0], [0,-1.5,0], [0,0,-1.5]])
#     u, v, w = np.array([[3,0,0], [0,3,0], [0,0,3]])
#     ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05, color="black", linewidth=0.5)

#     ax.text(0, 0, 1.7, r"|0⟩", color="black", fontsize=16)
#     ax.text(0, 0, -1.9, r"|1⟩", color="black", fontsize=16)
#     ax.text(1.9, 0, 0, r"|+⟩", color="black", fontsize=16)
#     ax.text(-1.7, 0, 0, r"|–⟩", color="black", fontsize=16)
#     ax.text(0, 1.7, 0, r"|i+⟩", color="black", fontsize=16)
#     ax.text(0,-1.9, 0, r"|i–⟩", color="black", fontsize=16)

#     ax.scatter(
#         bloch_vectors[:,0], bloch_vectors[:,1], bloch_vectors[:, 2], c='#e29d9e', alpha=0.3
#     )

# # Used the mixed state simulator so we could have the density matrix for this part!
# def convert_to_bloch_vector(rho):
#     X = np.array([[0, 1], [1, 0]])
#     Y = np.array([[0, -1j], [1j, 0]])
#     Z = np.array([[1, 0], [0, -1]])
    
#     """Convert a density matrix to a Bloch vector."""
#     ax = np.trace(np.dot(rho, X)).real
#     ay = np.trace(np.dot(rho, Y)).real
#     az = np.trace(np.dot(rho, Z)).real
#     return [ax, ay, az]

# key = jax.random.PRNGKey(69)

# num_samples = 2023

# psi0 = jnp.array([1,0], dtype = complex)
# rho0 = jnp.outer(psi0, psi0)
# bloch_vectors = []

# for i in range(num_samples):
#     U_Haar, key = mpo.HaarSQ(key)
#     rho = jnp.matmul(U_Haar, jnp.matmul(rho0, U_Haar.conj().T))
#     bloch_vectors.append(convert_to_bloch_vector(rho))

# plot_bloch_sphere(jnp.array(bloch_vectors))



#-------------------------------------------#
#----- mpo to (mpo + mpo_dag)/2 test -------#
#-------------------------------------------#

# tensors = [jax.random.normal(shape = (i + 1, 2, i + 2, 2), key = jax.random.PRNGKey(i), dtype = complex) for i in range(num_sites)]
# lastshape = tensors[-1].shape[0:2] + (1,) + (tensors[-1].shape[3],)
# tensors[-1] = jax.random.normal(shape = lastshape, key = jax.random.PRNGKey(num_sites + 1), dtype = complex)

# T_full = mpo.full_contract(tensors)

# T_herm_full = (T_full + T_full.conj().T)/2

# herm_tensors = mpo.hermitize_mpo(tensors)

# T_herm_contract = mpo.full_contract(herm_tensors)

# print(jnp.linalg.norm(T_herm_full - T_herm_contract))


#-------------------------------------------#
#----------- mpo conjugate test ------------#
#-------------------------------------------#

# tensors = [jax.random.normal(shape = (i + 1, 2, i + 2, 2), key = jax.random.PRNGKey(i), dtype = complex) for i in range(num_sites)]
# lastshape = tensors[-1].shape[0:2] + (1,) + (tensors[-1].shape[3],)
# tensors[-1] = jax.random.normal(shape = lastshape, key = jax.random.PRNGKey(num_sites + 1), dtype = complex)

# conj_tensors = mpo.conjugate_mpo(tensors)

# T_full = mpo.full_contract(tensors)
# T_dag_full = mpo.full_contract(conj_tensors)

# print(jnp.linalg.norm(T_full.conj().T - T_dag_full))


#-------------------------------------------#
#------------ vec to mpo test --------------#
#-------------------------------------------#

# shape = mpo.mpo_tensors_shape_from_bond_dim(N = num_sites, D = 6) 
# t_vec_lengths = [s[0] * s[1] * s[2] * s[3] for s in shape]
# vec_total_length = np.sum(t_vec_lengths)
# vec = jnp.arange(vec_total_length)

# mpo_tensors = mpo.vec_to_herm_mpo(vec, tuple(shape))


#-------------------------------------------#
#----------- vec to tensor test ------------#
#-------------------------------------------#

# t_shape = (3,2,7,2)
# v_shape = mpo.tensor_shape_to_vec_shape(t_shape)
# vec = jnp.arange(v_shape[0])

# tensor = mpo.vec_to_herm_tensor(vec, t_shape)

# seed = 420
# key = jax.random.PRNGKey(seed)
# circ = mpo.SumZ_RXX(N = num_sites, d = 4, p = 0.0, key = key)
# # circ.theta = jnp.zeros(circ.theta.shape)

# primal1 = circ.primal_noisy()

# init_mpo_tensors = circ.init_mpo(D = 4)

# primal2 = mpo.trace_two_MPOs(circ.psi_init_tensors, init_mpo_tensors)

# print(np.abs(primal1 - primal2))

#-------------------------------------------#
#-------------- mpo init test --------------#
#-------------------------------------------#

# seed = 420
# key = jax.random.PRNGKey(seed)
# circ = mpo.SumZ_RXX(N = num_sites, d = 4, p = 0.0, key = key)
# # circ.theta = jnp.zeros(circ.theta.shape)

# primal1 = circ.primal_noisy()

# init_mpo_tensors = circ.init_mpo(D = 4)

# primal2 = mpo.trace_two_MPOs(circ.psi_init_tensors, init_mpo_tensors)

# print(np.abs(primal1 - primal2))



#-------------------------------------------#
#--------------- energy test ---------------#
#-------------------------------------------#

# tensors = [jax.random.normal(shape = (i + 1, 2, i + 2, 2), key = jax.random.PRNGKey(i), dtype = complex) for i in range(num_sites)]
# lastshape = tensors[-1].shape[0:2] + (1,) + (tensors[-1].shape[3],)
# tensors[-1] = jax.random.normal(shape = lastshape, key = jax.random.PRNGKey(num_sites + 1), dtype = complex)

# H_tensors = [jax.random.normal(shape = (i + 1, 2, i + 2, 2), key = jax.random.PRNGKey(i + 69), dtype = complex) for i in range(num_sites)]
# lastshape = H_tensors[-1].shape[0:2] + (1,) + (H_tensors[-1].shape[3],)
# H_tensors[-1] = jax.random.normal(shape = lastshape, key = jax.random.PRNGKey(num_sites + 1 + 69), dtype = complex)

# res = mpo.trace_H_MPO(H_tensors, tensors)
# T_full = mpo.full_contract(tensors)
# H_full = mpo.full_contract(H_tensors)
# res2 = jnp.trace(jnp.matmul(H_full, T_full))
# print(jnp.abs(res - res2))

#-------------------------------------------#
#----------- circuit primal test -----------#
#-------------------------------------------#

# seed = 69
# key = jax.random.PRNGKey(seed)
# circ = mpo.SumZ_RXX(N = num_sites, d = 8, p = 0.0, key = key)

# print(circ.primal_noisy())

#-------------------------------------------#
#----------- two qubit gate test -----------#
#-------------------------------------------#

# tensors = [jax.random.normal(shape = (i + 1, 2, i + 2, 2), key = jax.random.PRNGKey(i), dtype = complex) for i in range(num_sites)]
# lastshape = tensors[-1].shape[0:2] + (1,) + (tensors[-1].shape[3],)
# tensors[-1] = jax.random.normal(shape = lastshape, key = jax.random.PRNGKey(num_sites + 1), dtype = complex)

# full_tensor_pre = mpo.full_contract(tensors)

# sites_to_gate = [0,1]

# _, gate_tensor = mpo.CNOT()
# identity_left = jnp.identity(2 ** (sites_to_gate[0]), dtype = complex) 
# identity_right = jnp.identity(2 ** (num_sites - sites_to_gate[0] - 2), dtype = complex) 
# gate_full = jnp.kron(identity_left, jnp.kron(mpo.full_contract(gate_tensor), identity_right))

# print(jnp.linalg.norm(jnp.matmul(mpo.full_contract(gate_tensor), mpo.full_contract(gate_tensor).conj().T) - jnp.identity(4)))

# new_tensors = mpo.twoq_gate(gate_tensor, [tensors[i] for i in sites_to_gate])

# for i, site in enumerate(sites_to_gate):
#     tensors[site] = new_tensors[i]

# full1 = mpo.full_contract(tensors)

# full2 = jnp.matmul(gate_full, jnp.matmul(full_tensor_pre, gate_full.conj().T))

# print(jnp.linalg.norm(full1 - full2))

#-------------------------------------------#
#----------- one qubit gate test -----------#
#-------------------------------------------#

# tensors = [jax.random.normal(shape = (i + 1, 2, i + 2, 2), key = jax.random.PRNGKey(i), dtype = complex) for i in range(num_sites)]
# lastshape = tensors[-1].shape[0:2] + (1,) + (tensors[-1].shape[3],)
# tensors[-1] = jax.random.normal(shape = lastshape, key = jax.random.PRNGKey(num_sites + 1), dtype = complex)

# full_tensor_pre = mpo.full_contract(tensors)

# site_to_gate = 1

# gate_tensor = mpo.RX(np.pi/2)
# identity_left = jnp.identity(2 ** (site_to_gate), dtype = complex) 
# identity_right = jnp.identity(2 ** (num_sites - site_to_gate - 1), dtype = complex) 
# gate_full = jnp.kron(identity_left, jnp.kron(gate_tensor, identity_right))

# print(jnp.matmul(gate_tensor, gate_tensor.conj().T))

# new_tensor = mpo.singleq_gate(gate_tensor, tensors[site_to_gate])
# tensors[site_to_gate] = new_tensor

# full1 = mpo.full_contract(tensors)

# full2 = jnp.matmul(gate_full, jnp.matmul(full_tensor_pre, gate_full.conj().T))

# print(jnp.linalg.norm(full1 - full2))



#-------------------------------------------#
#------------- common gates test -----------#
#-------------------------------------------#

# theta = np.pi/5
# c = jnp.cos(theta)
# s = jnp.sin(theta)

# U = jnp.array([[c, 0.0, 0.0, -1j*s],
#                [0.0, c, -1j*s, 0.0],
#                [0.0, -1j*s, c, 0.0],
#                [-1j*s, 0.0, 0.0, c]])

# # CNOT = jnp.array([[1, 0, 0, 0],
# #                   [0, 1, 0, 0],
# #                   [0, 0, 0, 1],
# #                   [0, 0, 1, 0]], dtype = complex)

# tensors, s = mpo.gate_to_MPO(U, num_sites = 2, D = 4)
# T_full = mpo.full_contract(tensors)

# print(jnp.linalg.norm(U - T_full))

#-------------------------------------------#
#------------- trace(MPO^2) test -----------#
#-------------------------------------------#


# tensors = [jax.random.normal(shape = (i + 1, 2, i + 2, 2), key = jax.random.PRNGKey(i), dtype = complex) for i in range(num_sites)]
# lastshape = tensors[-1].shape[0:2] + (1,) + (tensors[-1].shape[3],)
# tensors[-1] = jax.random.normal(shape = lastshape, key = jax.random.PRNGKey(num_sites + 1), dtype = complex)
#
# res = mpo.trace_MPO_squared(tensors)
# T_full = mpo.full_contract(tensors)
# res2 = jnp.trace(jnp.matmul(T_full, T_full))
# print(jnp.abs(res - res2))

#-------------------------------------------#
#------------- MPO subtract test -----------#
#-------------------------------------------#

# tensors2 = [jax.random.normal(shape = (i + 3, 2, i + 4, 2), key = jax.random.PRNGKey(i * 5), dtype = complex) for i in range(num_sites)]
# lastshape = tensors2[-1].shape[0:2] + (1,) + (tensors2[-1].shape[3],)
# firstshape = (1,) + tensors2[0].shape[1:]
# tensors2[-1] = jax.random.normal(shape = lastshape, key = jax.random.PRNGKey(num_sites * 5), dtype = complex)
# tensors2[0] = jax.random.normal(shape = firstshape, key = jax.random.PRNGKey(num_sites * 9), dtype = complex)
#
# stensors = mpo.subtract_MPO(tensors, tensors2)
#
# full1 = mpo.full_contract(tensors)
# full2 = mpo.full_contract(tensors2)
#
# print(jnp.linalg.norm(full1 + full2 - mpo.full_contract(stensors)))

#-------------------------------------------#
#---------------- canon test ---------------#
#-------------------------------------------#

# ctensors = mpo.left_canonicalize(tensors)
# # ctensors = mpo.right_canonicalize(tensors)
# norms = mpo.check_canon(ctensors, canon = "left")
#
# T_full = mpo.full_contract(tensors)
# CT_full = mpo.full_contract(ctensors)
#
# print(np.linalg.norm(T_full - CT_full))



# jnp.tensordot(u, u.conj(), axes = ((0, 1, 3), (0, 3, 1)))


# M_full = mpo.full_contract(tensors)
# H_full = H.full_ham()
#
# print(jnp.linalg.norm(M_full - H_full))

# tensor = jax.random.normal(shape = (19, 16, 21, 16), key = jax.random.PRNGKey(69))

# u, s, vh = mpo.left_split_lurd_tensor(tensor, D = 30)
#
# node = tn.Node(tensor, axis_names = ["left", "up", "right", "down"])
#
# left_node, S_node, right_node, trunc_S_vals = \
# tn.split_node_full_svd(
# node,
# left_edges = [node["left"], node["up"], node["down"]],
# right_edges = [node["right"]],
# max_singular_values = 30,
# left_name = "shdf",
# left_edge_name = "right",
# right_edge_name = "Sr")
#
# left_node.reorder_edges([left_node["left"], left_node["up"],
#                         S_node["right"], left_node["down"]])
#
# print(jnp.linalg.norm(u - left_node.tensor))
# print(jnp.linalg.norm(vh - right_node.tensor))
# print(jnp.linalg.norm(jnp.diag(s) - S_node.tensor))

# u, s, vh = mpo.right_split_lurd_tensor(tensor, D = 10)
#
# node = tn.Node(tensor, axis_names = ["left", "up", "right", "down"])
#
# left_node, S_node, right_node, trunc_S_vals = \
# tn.split_node_full_svd(
# node,
# left_edges = [node["left"]],
# right_edges = [node["up"], node["down"], node["right"]],
# max_singular_values = 10,
# left_name = "shdf",
# left_edge_name = "Sl",
# right_edge_name = "left")
#
# right_node.reorder_edges([S_node["left"], right_node["up"],
#                          right_node["right"], right_node["down"]])
#
# print(jnp.linalg.norm(u - left_node.tensor))
# print(jnp.linalg.norm(vh - right_node.tensor))
# print(jnp.linalg.norm(jnp.diag(s) - S_node.tensor))

# D = None
#
# dims_max = (2 * 2) ** np.concatenate((np.arange(1, num_sites//2 + 1),
#                                  np.arange(num_sites//2 - 1, 0, -1)))
# num_bonds = num_sites - 1
#
# if D is not None:
#     compressed_dims = jnp.where(dims_max <= D, dims_max, D)
# else:
#     compressed_dims = [None] * num_bonds
#
# i = 0
# u, s, vh = mpo.split_lurd_tensor(tensors[i], D = compressed_dims[i])
# svh = jnp.matmul(jnp.diag(s), vh)
# new_right = jnp.tensordot(svh, tensors[i + 1], axes=((-1,), (0,)))
#

# arrs = mpo.left_canonicalize(arrs)
# norms = mpo.check_canon(arrs)
#
# print(norms)

# i = 0
# compressed_dims = [None] * M.num_bonds
# left_node, S_node, right_node, trunc_S_vals = tn.split_node_full_svd(M.nodes[i], left_edges = [M.nodes[i]["left"], M.nodes[i]["up"], M.nodes[i]["down"]], right_edges = [M.nodes[i]["right"]], max_singular_values = compressed_dims[i], left_name = M.name + str(i), left_edge_name = "right", right_edge_name = "Sr")
#
# left_node.reorder_edges([left_node["left"], left_node["up"], S_node["right"], left_node["down"]])
#
# res = tn.contract_between(S_node, right_node, name = "SV",
# output_edge_order = [S_node["right"], right_node["right"]],
# axis_names = ["Sl", "right"])
#
# new_right = tn.contract_between(res, M.nodes[i + 1], name = M.name + str(i + 1),
# output_edge_order = [res["Sl"], M.nodes[i + 1]["up"], M.nodes[i + 1]["right"], M.nodes[i + 1]["down"]],
# axis_names = ["left", "up", "right", "down"])
#
# M.nodes[i] = left_node
# M.nodes[i + 1] = new_right
#
# check_node = tn.replicate_nodes([M.nodes[i],])[0]
# check_node_conj = tn.replicate_nodes([M.nodes[i]], conjugate = True)[0]
#
# check_node["left"] ^ check_node_conj["left"]
# check_node["up"] ^ check_node_conj["down"]
# check_node["down"] ^ check_node_conj["up"]
#
# check = tn.contract_between(check_node, check_node_conj,
# output_edge_order = [check_node["right"], check_node_conj["right"]])