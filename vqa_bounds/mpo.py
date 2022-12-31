"""
V1:
    sigma_d = -H.
    lambda_d = 0.
    OBC.
"""

import abc
from typing import List, Tuple, Callable, Dict
from functools import partial
import copy

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
# from tensornetwork.backends.decorators import jit
# from tensornetwork.backends import backend_factory
from jax.config import config
import tensornetwork as tn
tn.set_default_backend("jax")
config.update("jax_enable_x64", True)

# NEED:
# 2. Compress

class Hamiltonian():
    def __init__(self, num_sites: int):
        self.num_sites = num_sites
        self.site_tuple_list = [(i , i + 1) for i in range(num_sites - 1)]
        self.terms = {}

    def full_ham(self):
        full_mat = 0

        for site_tuple, term in self.terms.items():
            dim_left = 2 ** site_tuple[0]
            dim_right = 2 ** (self.num_sites - site_tuple[-1] - 1)

            identity_left = jnp.identity(dim_left)
            identity_right = jnp.identity(dim_right)

            full_mat += jnp.kron(identity_left, jnp.kron(term, identity_right))

        return full_mat

class MPO():
    def __init__(self, num_sites: int, name: str = None, arrs: List = None, canon: str = None):
        """
        MPO is a list of (tn.Node)s with bonds connected using tn.^.

        Assumes that input arrays have axes shaped as (l, u, r, d)

        self.canon =
            None: not canonicalized
            left
            right
            Id: is identity MPO
        """
        # self.backend =
        self.d = 2
        self.num_sites = num_sites
        self.L = num_sites
        self.num_bonds = num_sites - 1 # not counting the dim 1 bonds at the ends
        # self.dims_max = np.where(dims <= self.D, dims, self.D)
        self.name = name
        # self.arrs = arrs

        self.nodes = []

        if name is None:
            name = "?"

        if arrs is not None:
            for i, arr in enumerate(arrs):
                self.nodes.append(tn.Node(arr, name = name + str(i),
                axis_names = ["left", "up", "right", "down"]))
                self.canon = canon

        else:
            id = jnp.eye(2, dtype = complex)
            id = id.reshape((1, 2, 1, 2))
            self.nodes = [tn.Node(id, name = "Id" + str(i),
            axis_names = ["left", "up", "right", "down"])] * self.num_sites
            self.canon = "Id"

        self.dims = [node.shape[2] for node in self.nodes[:-1]]

        assert(len(self.dims) == self.num_bonds)

        for i in range(num_sites - 1):
            nodeL = self.nodes[i]
            nodeR = self.nodes[i + 1]
            bond = nodeL["right"] ^ nodeR["left"]

@jit
def full_contract(arrs: List[jnp.array]):
    """
    WARNING: memory expensive!!
    """

    mpo = MPO(len(arrs), arrs = arrs, canon = None)

    full_op = tn.contractors.auto(mpo.nodes,
    output_edge_order = [mpo.nodes[0]["left"]] + \
    [node["up"] for node in mpo.nodes] + \
    [node["down"] for node in mpo.nodes] + \
    [mpo.nodes[-1]["right"]])

    full_op = jnp.squeeze(full_op.tensor)
    full_op = jnp.reshape(full_op, (2**mpo.num_sites, 2**mpo.num_sites))

    return full_op

# @partial(jit, backend=backend_factory.get_backend("jax"))
# @partial(jit, static_argnums = 0)
def left_canonicalize_body_fun(i, args):
    arrs, compressed_dims = args

    nodeL = tn.Node(arrs[i], axis_names = ["left", "up", "right", "down"])
    nodeR = tn.Node(arrs[i + 1], axis_names = ["left", "up", "right", "down"])

    nodeL["right"] ^ nodeR["left"]

    left_node, S_node, right_node, trunc_S_vals = \
    tn.split_node_full_svd(
    nodeL,
    left_edges = [nodeL["left"], nodeL["up"], nodeL["down"]],
    right_edges = [nodeL["right"]],
    max_singular_values = compressed_dims.at[i].get(),
    left_name = "whatevs",
    left_edge_name = "right",
    right_edge_name = "Sr")

    left_node.reorder_edges([left_node["left"], left_node["up"],
                            S_node["right"], left_node["down"]])

    res = tn.contract_between(S_node, right_node, name = "SV",
    output_edge_order = [S_node["right"], right_node["right"]],
    axis_names = ["Sl", "right"])

    new_right = tn.contract_between(res, nodeR,
    name = "whatevs",
    output_edge_order = [res["Sl"], nodeR["up"],
    nodeR["right"], nodeR["down"]],
    axis_names = ["left", "up", "right", "down"])

    arrs[i] = left_node.tensor
    arrs[i + 1] = new_right.tensor

    return (arrs, compressed_dims)

@partial(jit, static_argnums = (1,))
def left_canonicalize(arrs: List[jnp.array], D:int = None):
    """
    Left canonicalize (leave last site uncanonicalized)
    and compress if D is specified.
    """

    num_sites = len(arrs)
    dims_max = (2 * 2) ** np.concatenate((np.arange(1, num_sites//2 + 1),
                                     np.arange(num_sites//2 - 1, 0, -1)))
    num_bonds = num_sites - 1

    if D is not None:
        compressed_dims = np.where(dims_max <= D, dims_max, D)
    else:
        compressed_dims = [None] * num_bonds

    # first iteration
    i = 0
    nodeL = tn.Node(arrs[i], axis_names = ["left", "up", "right", "down"])
    nodeR = tn.Node(arrs[i + 1], axis_names = ["left", "up", "right", "down"])

    nodeL["right"] ^ nodeR["left"]

    left_node, S_node, right_node, trunc_S_vals = \
    tn.split_node_full_svd(
    nodeL,
    left_edges = [nodeL["left"], nodeL["up"], nodeL["down"]],
    right_edges = [nodeL["right"]],
    max_singular_values = compressed_dims[i],
    left_name = str(i),
    left_edge_name = "right",
    right_edge_name = "Sr")

    left_node.reorder_edges([left_node["left"], left_node["up"],
                            S_node["right"], left_node["down"]])


    res = tn.contract_between(S_node, right_node, name = "SV",
    output_edge_order = [S_node["right"], right_node["right"]],
    axis_names = ["Sl", "right"])

    new_right = tn.contract_between(res, nodeR,
    name = str(i + 1),
    output_edge_order = [res["Sl"], nodeR["up"],
    nodeR["right"], nodeR["down"]],
    axis_names = ["left", "up", "right", "down"])

    arrs[i] = left_node.tensor
    arrs[i + 1] = new_right.tensor

    # middle iterations
    init_args = (jnp.array(arrs[1:-2]), jnp.array(compressed_dims[1:-1]))
    arrs[1:-2], _, _, _ = jax.lax.fori_loop(0, num_sites-3,
    left_canonicalize_body_fun, init_args)

    # last iteration
    i = num_sites - 2
    nodeL = tn.Node(arrs[i], axis_names = ["left", "up", "right", "down"])
    nodeR = tn.Node(arrs[i + 1], axis_names = ["left", "up", "right", "down"])

    nodeL["right"] ^ nodeR["left"]

    left_node, S_node, right_node, trunc_S_vals = \
    tn.split_node_full_svd(
    nodeL,
    left_edges = [nodeL["left"], nodeL["up"], nodeL["down"]],
    right_edges = [nodeL["right"]],
    max_singular_values = compressed_dims[i],
    left_name = str(i),
    left_edge_name = "right",
    right_edge_name = "Sr")

    left_node.reorder_edges([left_node["left"], left_node["up"],
                            S_node["right"], left_node["down"]])

    res = tn.contract_between(S_node, right_node, name = "SV",
    output_edge_order = [S_node["right"], right_node["right"]],
    axis_names = ["Sl", "right"])

    new_right = tn.contract_between(res, nodeR,
    name = str(i + 1),
    output_edge_order = [res["Sl"], nodeR["up"],
    nodeR["right"], nodeR["down"]],
    axis_names = ["left", "up", "right", "down"])

    arrs[i] = left_node.tensor
    arrs[i + 1] = new_right.tensor

    return arrs

    # for i in range(self.num_sites - 1):
    #     left_node, S_node, right_node, trunc_S_vals = \
    #     tn.split_node_full_svd(
    #     self.nodes[i],
    #     left_edges = [self.nodes[i]["left"], self.nodes[i]["up"], self.nodes[i]["down"]],
    #     right_edges = [self.nodes[i]["right"]],
    #     max_singular_values = compressed_dims[i],
    #     left_name = self.name + str(i),
    #     left_edge_name = "right",
    #     right_edge_name = "Sr")
    #
    #     left_node.reorder_edges([left_node["left"], left_node["up"],
    #                             S_node["right"], left_node["down"]])
    #
    #     res = tn.contract_between(S_node, right_node, name = "SV",
    #     output_edge_order = [S_node["right"], right_node["right"]],
    #     axis_names = ["Sl", "right"])
    #
    #     new_right = tn.contract_between(res, self.nodes[i + 1],
    #     name = self.name + str(i + 1),
    #     output_edge_order = [res["Sl"], self.nodes[i + 1]["up"],
    #     self.nodes[i + 1]["right"], self.nodes[i + 1]["down"]],
    #     axis_names = ["left", "up", "right", "down"])
    #
    #     self.nodes[i] = left_node
    #     self.nodes[i + 1] = new_right

    # self.canon = "left"

def check_canon_left_body_fun(i, args):
    arrs, norm_list = args

    check_node = tn.Node(arrs[i], axis_names = ["left", "up", "right", "down"])
    check_node_conj = tn.replicate_nodes([check_node,], conjugate = True)[0]

    check_node["left"] ^ check_node_conj["left"]
    check_node["up"] ^ check_node_conj["down"]
    check_node["down"] ^ check_node_conj["up"]

    check = tn.contract_between(check_node, check_node_conj,
    output_edge_order = [check_node["right"], check_node_conj["right"]])

    check_tensor = check.tensor

    norm_list.at[i].set(jnp.linalg.norm(check_tensor - jnp.identity(check_tensor.shape[0])))

    return arrs, norm_list

@partial(jit, static_argnums = (1,))
def check_canon(arrs: List[jnp.array], canon = "left"):

    num_sites = len(arrs)
    norm_list = jnp.zeros(num_sites - 1)

    if canon == "left":
        init_args = (arrs, norm_list)
        arrs, norm_list = jax.lax.fori_loop(0, num_sites-1, check_canon_left_body_fun, init_args)

        # for i in range(num_sites - 1):
        #     check_node = tn.Node(arrs[i], axis_names = ["left", "up", "right", "down"])
        #     check_node_conj = tn.replicate_nodes([check_node,], conjugate = True)[0]
        #
        #     check_node["left"] ^ check_node_conj["left"]
        #     check_node["up"] ^ check_node_conj["down"]
        #     check_node["down"] ^ check_node_conj["up"]
        #
        #     check = tn.contract_between(check_node, check_node_conj,
        #     output_edge_order = [check_node["right"], check_node_conj["right"]])
        #
        #     check_tensor = check.tensor
        #
        #     norm_list.append(jnp.linalg.norm(check_tensor - jnp.identity(check_tensor.shape[0])))

    return norm_list

def subtract_MPO(a: MPO, b: MPO):
    """
    Direct sum.
    """

def trace_MPO_squared(H: MPO):
    """
    Connect up/down edges with a replica (tn.replicate_nodes) and
    use tn.contract_auto or tn.contract_between.
    """

def HamSumZ(num_sites: int):
    """
    Returns Hamiltonian and MPO representations of \sum Z.
    """
    H = Hamiltonian(num_sites)

    I = jnp.array([[1, 0],[0, 1]], dtype = complex)
    Z = jnp.array([[1, 0],[0, -1]], dtype = complex)

    for site_tuple in H.site_tuple_list:
        H.terms[site_tuple] = jnp.kron(I, Z)

    H.terms[(0,1)] += jnp.kron(Z, I)

    arrs = []

    op = jnp.block([Z, I]) # shape = (bs) (b' s')) # b is bond, s is physical
    op = jnp.reshape(op, (1, 2, 2, 2)) # shape = (b) (s) (b') (s')
    arrs.append(op)

    op = jnp.block([[I, jnp.zeros((2,2))], [Z, I]]) # shape = (bs) (b' s'))
    op = jnp.reshape(op, (2, 2, 2, 2)) # shape = (b) (s) (b') (s')
    arrs += [op] * (num_sites - 2)

    op = jnp.row_stack((I, Z))
    op = jnp.reshape(op, (2, 2, 1, 2)) # shape = (b) (s) (b') (s')
    arrs.append(op)

    return H, arrs

# managing bond dims?
    # optimization will need fixed sizes for MPOs
# subtract/sum with negation
    # probably easiest to implement subtraction directly
# dual circuit
# vec to MPO
    # vec to operators first
    # sum with conj tr to get Herm
# tr(.^2)
    # optimal contraction order?
# local Ham to MPO
    # ??
