import abc
from typing import List, Tuple, Callable, Dict
from functools import partial
import copy

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.config import config
import tensornetwork as tn
tn.set_default_backend("jax")
config.update("jax_enable_x64", True)

from vqa_bounds import mpo

num_sites = 4

#-------------------------------------------#
#------------ vec to mpo test --------------#
#-------------------------------------------#

shape = mpo.mpo_tensors_shape_from_bond_dim(N = num_sites, D = 6) 
t_vec_lengths = [s[0] * s[1] * s[2] * s[3] for s in shape]
vec_total_length = np.sum(t_vec_lengths)
vec = jnp.arange(vec_total_length)

mpo_tensors = mpo.vec_to_herm_mpo(vec, tuple(shape))


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