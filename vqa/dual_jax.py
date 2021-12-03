from typing import List, Tuple, Callable, Dict

import tensornetwork as tn
import numpy as np
import scipy
import qutip
import jax.numpy as jnp
from jax import jit, grad, vmap

from vqa import graphs
from vqa import problems
from vqa import algorithms

"""
Moving the action of the class MaxCutDual() to pure functions in order to be
able to use JAX cleanly.
"""

#------------ Problem parameters ------------#

class MaxCutDualParams():

    def __init__(self, prob_obj: problems.Problem,
                 p: int, gamma: np.array, beta: np.array, p_noise: float):

        dual_obj = dual.MaxCutDual(prob_obj = prob_obj, p = p, gamma = gamma, beta = beta, p_noise = p_noise)

#------------ Tensor reshaping methods ------------#

def mat_2_tensor(A: jnp.array, dim_list: Tuple, num_sites_in_lattice: int):

    """
    Converts an array A that is indexed as (i1 i2 ... iN)(i1p i2p ... iNp)
    into a tensor indexed as (i1) (i1p) (i2) (i2p) ... (iN) (iNp). () denote
    composite indices.
    """

    # (i1 i2 ... iN)(i1p i2p ... iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
    T = jnp.flatten(A)

    # (i1 i2 ... iN i1p i2p ... iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
    T = jnp.reshape(T, dim_list)

    # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1) (i1p) ... (iN) (iNp)
    # [0, N, 1, N + 1, ..., N - 1, 2N - 1]
    i1 = jnp.arange(num_sites_in_lattice)
    i2 = num_sites_in_lattice + i1
    i  = jnp.zeros(2 * num_sites_in_lattice, dtype = int)
    i = i.at[::2].set(i1)
    i = i.at[1::2].set(i2)

    T = jnp.transpose(T, tuple(i))

    return T

def tensor_2_mat(T: jnp.array, dim: int, num_sites_in_lattice: int):

    """
    Converts a tensor T that is indexed as  (i1) (i1p) (i2) (i2p) ... (iN) (iNp)
    into a matrix indexed as (i1 i2 ... iN)(i1p i2p ... iNp). () denote
    composite indices.
    """

    # (i1) (i1p) (i2) (i2p) ... (iN) (iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
    # [0, 2, 4, ..., 2N - 2, 1, 3, ..., 2N - 1]
    i1 = jnp.arange(0, 2 * num_sites_in_lattice - 1, 2)
    i2 = jnp.arange(1, 2 * num_sites_in_lattice, 2)
    i = jnp.concatenate((i1, i2))

    A = jnp.transpose(T, tuple(i))

    # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
    A = jnp.flatten(A)

    # (i1 i2 ... iN i1p i2p ... iNp) -> (i1 i2 ... iN)(i1p i2p ... iNp)
    A = jnp.reshape(A, (dim, dim))

    return A

#------------ Dual variable reshape methods ------------#

def assemble_vars_into_tensors(vars_vec: jnp.array, p: int, dim: int):

    utri_indices = jnp.triu_indices(dim, 1)
    ltri_indices = (utri_indices[1], utri_indices[0])
    num_tri_elements = utri_indices[0].shape[0]
    num_diag_elements = dim

    vars_diag_list, vars_real_list, vars_imag_list, Lambdas \
            = unvectorize_vars(vars_vec, p, num_diag_elements, num_tri_elements)

    Sigmas = []

    for i in range(p):

        tri_real = jnp.zeros((dim, dim), dtype = complex)
        tri_imag = jnp.zeros((dim, dim), dtype = complex)

        tri_real[utri_indices] = vars_real_list[i]
        tri_real[ltri_indices] = vars_real_list[i]
        tri_imag[utri_indices] = 1j * vars_imag_list[i]
        tri_imag[ltri_indices] = -1j * vars_imag_list[i]

        vars_full = jnp.diag(vars_diag_list[i]) + tri_real + tri_imag

        Sigmas.append(tn.Node(mat_2_tensor(vars_full)))

    return Sigmas, Lambdas

def unvectorize_vars(vars_vec: jnp.array, p: int,
                     num_diag_elements: int, num_tri_elements: int):

    Lambdas = vars_vec[:p]

    vars_vec_split = jnp.split(vars_vec[p:], p)
    # split the variables into p equal arrays

    vars_diag_list = []
    vars_real_list = []
    vars_imag_list = []

    for i in range(p):

        # for each circuit step the variables are arranged as:
        # [vars_diag, vars_real, vars_imag]

        vars_diag = vars_vec_split[i][:num_diag_elements]
        vars_real = vars_vec_split[i][num_diag_elements:num_diag_elements + num_tri_elements]
        vars_imag = vars_vec_split[i][num_diag_elements + num_tri_elements:]

        vars_diag_list.append(vars_diag)
        vars_real_list.append(vars_real)
        vars_imag_list.append(vars_imag)

    return vars_diag_list, vars_real_list, vars_imag_list, Lambdas

#------------ Main dual function calc. methods ------------#

def objective(vars_vec: jnp.array, ?):

    # Unvectorize the vars_vec and construct Sigmas and Lambdas
    Sigmas, Lambdas = assemble_vars_into_tensors(vars_vec)

    # Compute the objective function using the list of tensors
    obj = cost(?)

    return obj

def cost(?):

    # the entropy term
    cost = jnp.dot(Lambdas, entropy_bounds)

    # the effective Hamiltonian term
    for i in range(p):

        # i corresponds to the layer of the circuit, 0 being the earliest
        # i.e. i = 0 corresponds to t = 1 in notes

        # construct the effective Hamiltonian for the ith step/layer
        Hi = construct_H(i = i, p = p, Sigmas = Sigmas,
                        rho_init_tensor = rho_init_tensor,
                        gamma = gamma, beta = beta, p_noise = p_noise,
                        num_sites_in_lattice = num_sites_in_lattice,
                        graph_edges = graph_edges,
                        graph_site_nums = graph_site_nums,
                        H_problem = H_problem)

        Ei = jnp.linalg.eigvals(Hi)

        cost += -Lambdas[i] * jnp.log2(jnp.sum(jnp.exp(-Ei/Lambdas[i])))

    # the initial state term
    epsilon_1_rho = noisy_circuit_layer(i = 0, Sigmas = Sigmas,
                    rho_init_tensor = rho_init_tensor,
                    gamma = gamma, beta = beta, p_noise = p_noise,
                    num_sites_in_lattice = num_sites_in_lattice,
                    graph_edges = graph_edges,
                    graph_site_nums = graph_site_nums).tensor

    sigma1 = tensor_2_mat(Sigmas[0].tensor)
    epsilon_1_rho = tensor_2_mat(epsilon_1_rho)
    cost += -jnp.trace(sigma1 @ epsilon_1_rho)

    return cost

def construct_H(i: int, p: int, Sigmas: List, rho_init_tensor: tn.Node,
                gamma: jnp.array, beta: jnp.array, p_noise: float,
                num_sites_in_lattice: int,
                graph_edges: List, graph_site_nums: List,
                H_problem: jnp.array):

    """
    Constructs, from the Sigma dual variables, the effective Hamiltonian
    corresponding to a layer.

    Params:
        i: layer number (>= 0)
    Returns:
        Hi: np.array of shape (self.dim, self.dim)
    """

    if i == p - 1:
        # TODO: check that the tensor product/site num ordering is
        # consistent
        Hi = tensor_2_mat(Sigmas[i].tensor) + H_problem

        return Hi

    else:
        Hi = Sigmas[i].tensor - noisy_circuit_layer(i = i + 1, Sigmas = Sigmas,
                        rho_init_tensor = rho_init_tensor,
                        gamma = gamma, beta = beta, p_noise = p_noise,
                        num_sites_in_lattice = num_sites_in_lattice,
                        graph_edges = graph_edges,
                        graph_site_nums = graph_site_nums).tensor

        return self.tensor_2_mat(Hi)

def noisy_circuit_layer(i: int, Sigmas: List, rho_init_tensor: tn.Node,
                        gamma: jnp.array, beta: jnp.array, p_noise: float,
                        num_sites_in_lattice: int,
                        graph_edges: List, graph_site_nums: List):

    """
    Applies the unitary corresponding to a circuit layer followed by noise
    to the dual variable Sigma. The noise model is depolarizing noise on
    each qubit.

    Params:
        i: layer number. Used to specify the tensor to act on and the gate
        params to use. Note the action when i = 0 is on the init state not
        the Sigma variable.
    Returns:
        res = tn.Node of shape self.dim_list that contains the tensor
        after the action of the layer.
    """

    if i == 0:
        res_tensor = circuit_layer(var_tensor = rho_init_tensor,
                          gamma = gamma[i], beta = beta[i],
                          num_sites_in_lattice = num_sites_in_lattice,
                          graph_edges = graph_edges,
                          graph_site_nums = graph_site_nums)
    else:
        res_tensor = circuit_layer(var_tensor = Sigmas[i],
                          gamma = gamma[i], beta = beta[i],
                          num_sites_in_lattice = num_sites_in_lattice,
                          graph_edges = graph_edges,
                          graph_site_nums = graph_site_nums)

    res_tensor = noise_layer(var_tensor = res_tensor,
                num_sites_in_lattice = num_sites_in_lattice, p_noise = p_noise)

    return res_tensor

def noise_layer(var_tensor: tn.Node, num_sites_in_lattice: int, p_noise: float):

    """
    Applies depolarizing noise on the var_tensor at all sites.
    """

    res_tensor = var_tensor

    X = jnp.array([[0, 1],[1, 0]])
    Y = jnp.array([[0, -1j],[1j, 0]])
    Z = jnp.array([[1, 0],[0, -1]])

    for site_num in range(num_sites_in_lattice):

        # --- applying I --- #
        res_array = (1 - 3 * p_noise/4) * res_tensor.tensor

        # --- applying X --- #
        tmp_tensor = res_tensor

        X_node = tn.Node(np.sqrt(p_noise/4) * X, axis_names = ["ja","ia"])
        X_prime_node = tn.Node(np.sqrt(p_noise/4) * X, axis_names = ["iap","kap"])

        ia  = 2 * site_num
        iap = 2 * site_num + 1

        edge_a      = X_node["ia"] ^ tmp_tensor[ia]
        edge_a_p    = X_prime_node["iap"] ^ tmp_tensor[iap]

        # perform the contraction
        new_edge_order = tmp_tensor.edges[:ia] + [X_node["ja"]] +\
                         tmp_tensor.edges[ia + 1:]
        tmp_tensor = tn.contract_between(X_node, tmp_tensor, output_edge_order = new_edge_order)

        new_edge_order = tmp_tensor.edges[:iap] + [X_prime_node["kap"]] +\
                         tmp_tensor.edges[iap + 1:]
        tmp_tensor = tn.contract_between(X_prime_node, tmp_tensor, output_edge_order = new_edge_order)

        res_array += tmp_tensor.tensor

        # --- applying Y --- #
        tmp_tensor = res_tensor

        Y_node = tn.Node(np.sqrt(p_noise/4) * Y, axis_names = ["ja","ia"])
        Y_prime_node = tn.Node(np.sqrt(p_noise/4) * Y, axis_names = ["iap","kap"])

        ia  = 2 * site_num
        iap = 2 * site_num + 1

        edge_a      = Y_node["ia"] ^ tmp_tensor[ia]
        edge_a_p    = Y_prime_node["iap"] ^ tmp_tensor[iap]

        # perform the contraction
        new_edge_order = tmp_tensor.edges[:ia] + [Y_node["ja"]] +\
                         tmp_tensor.edges[ia + 1:]
        tmp_tensor = tn.contract_between(Y_node, tmp_tensor, output_edge_order = new_edge_order)

        new_edge_order = tmp_tensor.edges[:iap] + [Y_prime_node["kap"]] +\
                         tmp_tensor.edges[iap + 1:]
        tmp_tensor = tn.contract_between(Y_prime_node, tmp_tensor, output_edge_order = new_edge_order)

        res_array += tmp_tensor.tensor

        # --- applying Z --- #
        tmp_tensor = res_tensor

        Z_node = tn.Node(np.sqrt(p_noise/4) * Z, axis_names = ["ja","ia"])
        Z_prime_node = tn.Node(np.sqrt(p_noise/4) * Z, axis_names = ["iap","kap"])

        ia  = 2 * site_num
        iap = 2 * site_num + 1

        edge_a      = Z_node["ia"] ^ tmp_tensor[ia]
        edge_a_p    = Z_prime_node["iap"] ^ tmp_tensor[iap]

        # perform the contraction
        new_edge_order = tmp_tensor.edges[:ia] + [Z_node["ja"]] +\
                         tmp_tensor.edges[ia + 1:]
        tmp_tensor = tn.contract_between(Z_node, tmp_tensor, output_edge_order = new_edge_order)

        new_edge_order = tmp_tensor.edges[:iap] + [Z_prime_node["kap"]] +\
                         tmp_tensor.edges[iap + 1:]
        tmp_tensor = tn.contract_between(Z_prime_node, tmp_tensor, output_edge_order = new_edge_order)

        res_array += tmp_tensor.tensor

        # update tensor
        res_tensor = tn.Node(res_array)

    return res_tensor

def circuit_layer(var_tensor: tn.Node,
                  gamma: float, beta: float,
                  num_sites_in_lattice: int,
                  graph_edges: List, graph_site_nums: List):

    res_tensor = var_tensor

    #----- Applying the problem unitary -----#
    # U = exp(-i gamma/2 w * (I - Z_j Z_k))
    U = jnp.diag([1, jnp.exp(1j * gamma), jnp.exp(1j * gamma), 1])
    U_dag = jnp.conj(jnp.transpose(U))

    # (ja jb) (ia ib) -> (ja jb ia ib)
    U_tensor = jnp.flatten(U)
    # (ja jb ia ib) -> (ja) (jb) (ia) (ib)
    U_tensor = jnp.reshape(U_tensor, (2,2,2,2))
    # (ja) (jb) (ia) (ib) -> (ja) (ia) (jb) (ib)
    U_tensor = jnp.transpose(U_tensor, [0, 2, 1, 3])

    # (iap ibp) (kap kbp) -> (iap ibp kap kbp)
    U_dag_tensor = jnp.flatten(U_dag)
    # (iap ibp kap kbp) -> (iap) (ibp) (kap) (kbp)
    U_dag_tensor = jnp.reshape(U_dag_tensor, (2,2,2,2))
    # (iap) (ibp) (kap) (kbp) -> (iap) (kap) (ibp) (kbp)
    U_dag_tensor = jnp.transpose(U_dag_tensor, [0, 2, 1, 3])

    for edge in graph_edges:

        site_num_a = jnp.min(graph_site_nums[edge[0]], graph_site_nums[edge[1]])
        site_num_b = jnp.max(graph_site_nums[edge[0]], graph_site_nums[edge[1]])

        ia = 2 * site_num_a
        ib = 2 * site_num_b

        iap = 2 * site_num_a + 1
        ibp = 2 * site_num_b + 1

        U_node      = tn.Node(U_tensor, axis_names = ["ja", "ia", "jb", "ib"], name = "U_node")
        U_dag_node  = tn.Node(U_dag_tensor, axis_names = ["iap", "kap", "ibp", "kbp"], name = "U_dag_node")

        # assumption that tn.Node() orders the axes in the same order as
        # the input np.array
        edge_a = U_node["ia"] ^ res_tensor[ia]
        edge_b = U_node["ib"] ^ res_tensor[ib]

        new_edge_order = res_tensor.edges[:ia] + [U_node["ja"]] +\
                         res_tensor.edges[ia + 1: ib] + [U_node["jb"]] +\
                         res_tensor.edges[ib + 1:]

        res_tensor = tn.contract_between(U_node, res_tensor, output_edge_order = new_edge_order)

        edge_a_p = U_dag_node["iap"] ^ res_tensor[iap]
        edge_b_p = U_dag_node["ibp"] ^ res_tensor[ibp]

        new_edge_order = res_tensor.edges[:iap] + [U_dag_node["kap"]] +\
                         res_tensor.edges[iap + 1: ibp] + [U_dag_node["kbp"]] +\
                         res_tensor.edges[ibp + 1:]

        res_tensor = tn.contract_between(U_dag_node, res_tensor, output_edge_order = new_edge_order)

    #----- Applying the mixing unitary -----#
    X = jnp.array([[0,1], [1,0]])
    I = jnp.array([[1,0], [0,1]])

    Ux = jnp.cos(beta) * I - 1j * jnp.sin(beta) * X
    Ux_dag = jnp.conj(jnp.transpose(Ux))

    for site_num in range(num_sites_in_lattice):

        Ux_node = tn.Node(Ux, axis_names = ["ja","ia"])
        Ux_dag_node = tn.Node(Ux_dag, axis_names = ["iap","kap"])

        ia  = 2 * site_num
        iap = 2 * site_num + 1

        edge_a      = Ux_node["ia"] ^ res_tensor[ia]
        edge_a_p    = Ux_dag_node["iap"] ^ res_tensor[iap]

        # perform the contraction

        new_edge_order = res_tensor.edges[:ia] + [Ux_node["ja"]] +\
                         res_tensor.edges[ia + 1:]
        res_tensor = tn.contract_between(Ux_node, res_tensor, output_edge_order = new_edge_order)

        new_edge_order = res_tensor.edges[:iap] + [Ux_dag_node["kap"]] +\
                         res_tensor.edges[iap + 1:]
        res_tensor = tn.contract_between(Ux_dag_node, res_tensor, output_edge_order = new_edge_order)

    return res_tensor

#------------ Other helper methods ------------#

def gen_entropy_bounds(p_noise: float, num_sites_in_lattice: int):

    q = 1 - p_noise
    q_powers = [q**i for i in range(p)]

    entropy_bounds = num_sites_in_lattice * p_noise * \
                     np.array([np.sum(q_powers[:i+1]) for i in range(p)])

    return entropy_bounds
