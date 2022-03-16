import abc
from typing import List, Tuple, Callable, Dict
from functools import partial
import time

import numpy as np
import scipy
import networkx as nx
import qutip
import tensornetwork as tn
tn.set_default_backend("jax")
import jax.numpy as jnp
from jax import jit, grad, vmap, value_and_grad
from jax.example_libraries import optimizers

from vqa_bounds import graphs, meta_system

class MaxCut1D(meta_system.System):

    def __init__(self, graph, lattice, d: int, p: float):

        self.graph = graph
        self.lattice = lattice

        # preparing the Z operators
        self.site_z_ops = {}
        self.site_nums = {}
        op_num = 0

        self.num_sites_in_lattice = self.lattice.number_of_nodes() # assuming even
        # layer numbering starts from 1
        self.site_index_list_odd_layer = list(zip(range(1, self.num_sites_in_lattice, 2), range(2, self.num_sites_in_lattice, 2)))
        self.site_index_list_even_layer = list(zip(range(0, self.num_sites_in_lattice, 2), range(1, self.num_sites_in_lattice, 2)))

        self.Z_qutip = qutip.sigmaz()
        self.X_qutip = qutip.sigmax()
        self.I_qutip = qutip.qeye(2)

        self.I_tot = qutip.tensor([self.I_qutip] * self.num_sites_in_lattice)

        for site in self.lattice:

            self.site_z_ops[site] = qutip.tensor([self.I_qutip] * op_num + [self.Z_qutip] + \
                                            [self.I_qutip] * (self.num_sites_in_lattice - op_num - 1))

            self.site_nums[site] = op_num
            op_num += 1

        # the problem Hamiltonian
        self.H = 0

        for edge in self.graph.edges:

            Zj = self.site_z_ops[edge[0]]
            Zk = self.site_z_ops[edge[1]]
            wjk = -1

            local_op = wjk/2 * (self.I_tot - Zj * Zk)

            self.H += local_op

        # for dual
        # self.local_dim = 2
        self.d = d
        # self.p = p
        # self.psi_init = self.init_state()
        # self.H_problem = jnp.array(self.H.full())
        #
        # self.dim_list = tuple([self.local_dim] * 2 * self.num_sites_in_lattice)
        # self.dim = self.local_dim ** self.num_sites_in_lattice
        #
        # self.utri_indices = jnp.triu_indices(self.dim, 1)
        # self.ltri_indices = (self.utri_indices[1], self.utri_indices[0])
        # self.num_tri_elements = self.utri_indices[0].shape[0]
        # self.num_diag_elements = self.dim
        #
        # # constructing initial density operator
        # self.rho_init = (self.psi_init * self.psi_init.dag()).full()
        # self.rho_init_tensor = tn.Node(self.mat_2_tensor(self.rho_init), name = "rho_init")
        #
        # self.X = jnp.array(qutip.sigmax().full())
        # self.Y = jnp.array(qutip.sigmay().full())
        # self.Z = jnp.array(qutip.sigmaz().full())
        # self.I = jnp.array(qutip.qeye(2).full())
        #
        # self.init_entropy_bounds()
        #
        # self.len_vars = (1 + self.num_diag_elements + 2 * self.num_tri_elements) * self.d

    def update_opt_circ_params(self, opt_circuit_params: np.array):

        self.opt_gamma = opt_circuit_params[:self.d]
        self.opt_beta = opt_circuit_params[self.d:]

    def init_state(self):

        psi0 = qutip.basis([2] * self.num_sites_in_lattice)
        psi0 = qutip.hadamard_transform(N = self.num_sites_in_lattice) * psi0

        return psi0

    def circuit_layer(self, state, layer_num, circ_params):

        gamma = circ_params[:self.d]
        beta = circ_params[self.d:]

        H_2site = qutip.tensor([self.Z_qutip] * 2) - qutip.tensor([self.I_qutip] * 2)
        U_2site = (-1j * gamma[layer_num - 1] * H_2site).expm()

        # print("layer num = ", layer_num)

        if layer_num % 2 == 0:
            site_index_list = self.site_index_list_even_layer
        else:
            site_index_list = self.site_index_list_odd_layer

        for site_tuple in site_index_list:

            # print("site_tuple = ", site_tuple)

            if site_tuple in self.graph.edges:

                # print("edge = ", site_tuple)

                U_2site_full = qutip.tensor(
                                [self.I_qutip] * site_tuple[0] +
                                [U_2site] +
                                [self.I_qutip] * (self.num_sites_in_lattice - site_tuple[0] - 2))
                state = U_2site_full * state

        if layer_num % 2 == 0:

            U_1site = (-1j * beta[(layer_num - 2)//2] * self.X_qutip).expm()

            for site in self.graph:

                site_num = self.site_nums[site]

                state = qutip.tensor([self.I_qutip] * site_num + [U_1site] + \
                        [self.I_qutip] * (self.num_sites_in_lattice - site_num - 1)) * state

        return state

    def circ_obj(self, circ_params: np.array):

        """
        Note: for uniform weight MaxCut the objective has periodicity of 2 pi in
        every gamma and pi in every beta.
        """

        psi_after_step = self.init_state()

        for layer_num in range(1, self.d + 1):

            psi_after_step = self.circuit_layer(psi_after_step, layer_num, circ_params)

        return qutip.expect(self.H, psi_after_step)

    def circ_grad(self, circ_params: np.array):

        gamma = circ_params[:self.d]
        beta = circ_params[self.d:]

        delta = 1e-4

        gradient = np.zeros(len(circ_params))

        for n in range(len(circ_params)):

            circ_params_plus = np.copy(circ_params)
            circ_params_plus[n] += delta

            circ_params_minus = np.copy(circ_params)
            circ_params_minus[n] -= delta

            objective_plus = self.circ_obj(circ_params_plus)
            objective_minus = self.circ_obj(circ_params_minus)

            gradient[n] = (objective_plus - objective_minus)/(2 * delta)

        return gradient

    # def mat_2_tensor(self, A: jnp.array):
    #
    #     """
    #     Converts an array A that is indexed as (i1 i2 ... iN)(i1p i2p ... iNp)
    #     into a tensor indexed as (i1) (i1p) (i2) (i2p) ... (iN) (iNp). () denote
    #     composite indices.
    #     """
    #
    #     # (i1 i2 ... iN)(i1p i2p ... iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
    #     T = jnp.ravel(A)
    #
    #     # (i1 i2 ... iN i1p i2p ... iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
    #     T = jnp.reshape(T, tuple([self.local_dim] * self.num_sites_in_lattice * 2))
    #
    #     # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1) (i1p) ... (iN) (iNp)
    #     # [0, N, 1, N + 1, ..., N - 1, 2N - 1]
    #     i1 = np.arange(self.num_sites_in_lattice)
    #     i2 = self.num_sites_in_lattice + i1
    #     i  = np.zeros(2 * self.num_sites_in_lattice, dtype = int)
    #     i[::2] = i1
    #     i[1::2] = i2
    #     # i = i.at[::2].set(i1)
    #     # i = i.at[1::2].set(i2)
    #
    #     T = jnp.transpose(T, tuple(i))
    #
    #     return T
    #
    # def tensor_2_mat(self, T: jnp.array):
    #
    #     """
    #     Converts a tensor T that is indexed as  (i1) (i1p) (i2) (i2p) ... (iN) (iNp)
    #     into a matrix indexed as (i1 i2 ... iN)(i1p i2p ... iNp). () denote
    #     composite indices.
    #     """
    #
    #     # (i1) (i1p) (i2) (i2p) ... (iN) (iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
    #     # [0, 2, 4, ..., 2N - 2, 1, 3, ..., 2N - 1]
    #     i1 = np.arange(0, 2 * self.num_sites_in_lattice - 1, 2)
    #     i2 = np.arange(1, 2 * self.num_sites_in_lattice, 2)
    #     i = np.concatenate((i1, i2))
    #
    #     A = jnp.transpose(T, tuple(i))
    #
    #     # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
    #     A = jnp.ravel(A)
    #
    #     # (i1 i2 ... iN i1p i2p ... iNp) -> (i1 i2 ... iN)(i1p i2p ... iNp)
    #     A = jnp.reshape(A, (self.dim, self.dim))
    #
    #     return A
    #
    # def init_entropy_bounds(self):
    #
    #     q = 1 - self.p
    #     q_powers = jnp.array([q**i for i in range(self.d)])
    #
    #     self.entropy_bounds = self.num_sites_in_lattice * self.p * jnp.log(2) * \
    #                           jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(self.d)])
    #
    # def dual_obj(self, dual_vars: jnp.array):
    #
    #     # Unvectorize the vars_vec and construct Sigmas and Lambdas
    #     self.assemble_vars_into_tensors(dual_vars)
    #
    #     # Compute the objective function using the list of tensors
    #     obj = self.cost()
    #
    #     return -jnp.real(obj)
    #
    # def dual_grad(self, dual_vars: jnp.array):
    #
    #     return grad(self.dual_obj)(dual_vars)
    #
    # def assemble_vars_into_tensors(self, vars_vec: jnp.array):
    #
    #     vars_diag_list, vars_real_list, vars_imag_list = self.unvectorize_vars(vars_vec)
    #     self.Sigmas = []
    #
    #     for i in range(self.d):
    #
    #         tri_real = jnp.zeros((self.dim, self.dim), dtype = complex)
    #         tri_imag = jnp.zeros((self.dim, self.dim), dtype = complex)
    #
    #         tri_real = tri_real.at[self.utri_indices].set(vars_real_list[i])
    #         tri_real = tri_real.at[self.ltri_indices].set(vars_real_list[i])
    #         tri_imag = tri_imag.at[self.utri_indices].set(1j * vars_imag_list[i])
    #         tri_imag = tri_imag.at[self.ltri_indices].set(-1j * vars_imag_list[i])
    #
    #         vars_full = jnp.diag(vars_diag_list[i]) + tri_real + tri_imag
    #
    #         self.Sigmas.append(tn.Node(self.mat_2_tensor(vars_full)))
    #
    # def unvectorize_vars(self, vars_vec: jnp.array):
    #
    #     a_vars = vars_vec.at[:self.d].get()
    #     self.Lambdas = jnp.log(1 + jnp.exp(a_vars))
    #
    #     vars_vec_split = jnp.split(vars_vec.at[self.d:].get(), self.d)
    #     # split the variables into p equal arrays
    #
    #     vars_diag_list = []
    #     vars_real_list = []
    #     vars_imag_list = []
    #
    #     for i in range(self.d):
    #
    #         # for each circuit step the variables are arranged as:
    #         # [vars_diag, vars_real, vars_imag]
    #
    #         vars_diag = vars_vec_split[i].at[:self.num_diag_elements].get()
    #         vars_real = vars_vec_split[i].at[self.num_diag_elements:self.num_diag_elements + self.num_tri_elements].get()
    #         vars_imag = vars_vec_split[i].at[self.num_diag_elements + self.num_tri_elements:].get()
    #
    #         vars_diag_list.append(vars_diag)
    #         vars_real_list.append(vars_real)
    #         vars_imag_list.append(vars_imag)
    #
    #     return vars_diag_list, vars_real_list, vars_imag_list
    #
    # def cost(self):
    #
    #     # the entropy term
    #     cost = jnp.dot(self.Lambdas, self.entropy_bounds)
    #
    #     # the effective Hamiltonian term
    #     for i in range(self.d):
    #
    #         # i corresponds to the layer of the circuit, 0 being the earliest
    #         # i.e. i = 0 corresponds to t = 1 in notes
    #
    #         # construct the effective Hamiltonian for the ith step/layer
    #         Hi = self.construct_H(i)
    #         Ei = jnp.linalg.eigvals(Hi)
    #
    #         cost += -self.Lambdas[i] * jnp.log(jnp.sum(jnp.exp(-Ei/self.Lambdas[i])))
    #
    #     # the initial state term
    #     # TODO: will this be faster if expressed in terms of contractions?
    #     sigma1 = self.tensor_2_mat(self.Sigmas[0].tensor)
    #     epsilon_1_rho = self.rho_init_tensor
    #     epsilon_1_rho = self.circuit_layer(layer_num = 0,
    #                                        var_tensor = epsilon_1_rho,
    #                                        mode = 'primal')
    #     epsilon_1_rho = self.noise_layer(epsilon_1_rho)
    #     epsilon_1_rho = self.tensor_2_mat(epsilon_1_rho.tensor)
    #     cost += -jnp.trace(jnp.matmul(sigma1, epsilon_1_rho))
    #
    #     # rho_init_mat = self.tensor_2_mat(self.rho_init_tensor.tensor)
    #     # epsilon_1_dag_sigma1 = self.tensor_2_mat(self.noisy_circuit_layer(i = 0).tensor)
    #     # cost += -jnp.trace(jnp.matmul(rho_init_mat, epsilon_1_dag_sigma1))
    #
    #     return cost
    #
    # def construct_H(self, i: int):
    #
    #     """
    #     Constructs, from the Sigma dual variables, the effective Hamiltonian
    #     corresponding to a layer.
    #
    #     Params:
    #         i: layer number (>= 0)
    #     Returns:
    #         Hi: np.array of shape (self.dim, self.dim)
    #     """
    #
    #     if i == self.d - 1:
    #         # TODO: check that the tensor product/site num ordering is
    #         # consistent
    #         Hi = self.tensor_2_mat(self.Sigmas[i].tensor) + self.H_problem
    #
    #         return Hi
    #
    #     else:
    #         Hi = self.Sigmas[i].tensor - self.noisy_circuit_layer(i + 1).tensor
    #
    #         return self.tensor_2_mat(Hi)
    #
    # def noise_layer(self, var_tensor: tn.Node):
    #
    #     """
    #     Applies depolarizing noise on the var_tensor at all sites.
    #     """
    #
    #     res_tensor = var_tensor
    #
    #     for site_num in range(self.num_sites_in_lattice):
    #
    #         # --- applying I --- #
    #         res_array = (1 - 3 * self.p/4) * res_tensor.tensor
    #
    #         # --- applying X --- #
    #         tmp_tensor = res_tensor
    #
    #         X_node = tn.Node(np.sqrt(self.p/4) * self.X, axis_names = ["ja","ia"])
    #         X_prime_node = tn.Node(np.sqrt(self.p/4) * self.X, axis_names = ["iap","kap"])
    #
    #         ia  = 2 * site_num
    #         iap = 2 * site_num + 1
    #
    #         edge_a      = X_node["ia"] ^ tmp_tensor[ia]
    #         edge_a_p    = X_prime_node["iap"] ^ tmp_tensor[iap]
    #
    #         # perform the contraction
    #         new_edge_order = tmp_tensor.edges[:ia] + [X_node["ja"]] +\
    #                          tmp_tensor.edges[ia + 1:]
    #         tmp_tensor = tn.contract_between(X_node, tmp_tensor, output_edge_order = new_edge_order)
    #
    #         new_edge_order = tmp_tensor.edges[:iap] + [X_prime_node["kap"]] +\
    #                          tmp_tensor.edges[iap + 1:]
    #         tmp_tensor = tn.contract_between(X_prime_node, tmp_tensor, output_edge_order = new_edge_order)
    #
    #         res_array += tmp_tensor.tensor
    #
    #         # --- applying Y --- #
    #         tmp_tensor = res_tensor
    #
    #         Y_node = tn.Node(np.sqrt(self.p/4) * self.Y, axis_names = ["ja","ia"])
    #         Y_prime_node = tn.Node(np.sqrt(self.p/4) * self.Y, axis_names = ["iap","kap"])
    #
    #         ia  = 2 * site_num
    #         iap = 2 * site_num + 1
    #
    #         edge_a      = Y_node["ia"] ^ tmp_tensor[ia]
    #         edge_a_p    = Y_prime_node["iap"] ^ tmp_tensor[iap]
    #
    #         # perform the contraction
    #         new_edge_order = tmp_tensor.edges[:ia] + [Y_node["ja"]] +\
    #                          tmp_tensor.edges[ia + 1:]
    #         tmp_tensor = tn.contract_between(Y_node, tmp_tensor, output_edge_order = new_edge_order)
    #
    #         new_edge_order = tmp_tensor.edges[:iap] + [Y_prime_node["kap"]] +\
    #                          tmp_tensor.edges[iap + 1:]
    #         tmp_tensor = tn.contract_between(Y_prime_node, tmp_tensor, output_edge_order = new_edge_order)
    #
    #         res_array += tmp_tensor.tensor
    #
    #         # --- applying Z --- #
    #         tmp_tensor = res_tensor
    #
    #         Z_node = tn.Node(np.sqrt(self.p/4) * self.Z, axis_names = ["ja","ia"])
    #         Z_prime_node = tn.Node(np.sqrt(self.p/4) * self.Z, axis_names = ["iap","kap"])
    #
    #         ia  = 2 * site_num
    #         iap = 2 * site_num + 1
    #
    #         edge_a      = Z_node["ia"] ^ tmp_tensor[ia]
    #         edge_a_p    = Z_prime_node["iap"] ^ tmp_tensor[iap]
    #
    #         # perform the contraction
    #         new_edge_order = tmp_tensor.edges[:ia] + [Z_node["ja"]] +\
    #                          tmp_tensor.edges[ia + 1:]
    #         tmp_tensor = tn.contract_between(Z_node, tmp_tensor, output_edge_order = new_edge_order)
    #
    #         new_edge_order = tmp_tensor.edges[:iap] + [Z_prime_node["kap"]] +\
    #                          tmp_tensor.edges[iap + 1:]
    #         tmp_tensor = tn.contract_between(Z_prime_node, tmp_tensor, output_edge_order = new_edge_order)
    #
    #         res_array += tmp_tensor.tensor
    #
    #         # update tensor
    #         res_tensor = tn.Node(res_array)
    #
    #     return res_tensor
    #
    # def problem_layer(self, layer_num: int, var_tensor: tn.Node, mode: str = "dual"):
    #
    #     res_tensor = var_tensor
    #
    #     #----- Applying the problem unitary -----#
    #     gamma = self.opt_gamma[layer_num]
    #
    #     # U = exp(-i gamma/2 w * (I - Z_j Z_k))
    #     U = jnp.diag(jnp.array([1, jnp.exp(1j * gamma), jnp.exp(1j * gamma), 1]))
    #     U_dag = jnp.transpose(jnp.conj(U))
    #
    #     # taking the adjoint of U and U_dag to impose dual channel on Sigmas
    #     if mode == 'dual':
    #         U = jnp.conj(jnp.transpose(U))
    #         U_dag = jnp.conj(jnp.transpose(U_dag))
    #
    #     # (ja jb) (ia ib) -> (ja jb ia ib)
    #     U_tensor = jnp.ravel(U)
    #     # (ja jb ia ib) -> (ja) (jb) (ia) (ib)
    #     U_tensor = jnp.reshape(U_tensor, (2,2,2,2))
    #     # (ja) (jb) (ia) (ib) -> (ja) (ia) (jb) (ib)
    #     U_tensor = jnp.transpose(U_tensor, [0, 2, 1, 3])
    #
    #     # (iap ibp) (kap kbp) -> (iap ibp kap kbp)
    #     U_dag_tensor = jnp.ravel(U_dag)
    #     # (iap ibp kap kbp) -> (iap) (ibp) (kap) (kbp)
    #     U_dag_tensor = jnp.reshape(U_dag_tensor, (2,2,2,2))
    #     # (iap) (ibp) (kap) (kbp) -> (iap) (kap) (ibp) (kbp)
    #     U_dag_tensor = jnp.transpose(U_dag_tensor, [0, 2, 1, 3])
    #
    #     for edge in self.graph.edges:
    #
    #         site_num_a = min([self.site_nums[edge[0]], self.site_nums[edge[1]]])
    #         site_num_b = max([self.site_nums[edge[0]], self.site_nums[edge[1]]])
    #
    #         ia = 2 * site_num_a
    #         ib = 2 * site_num_b
    #
    #         iap = 2 * site_num_a + 1
    #         ibp = 2 * site_num_b + 1
    #
    #         U_node      = tn.Node(U_tensor, axis_names = ["ja", "ia", "jb", "ib"], name = "U_node")
    #         U_dag_node  = tn.Node(U_dag_tensor, axis_names = ["iap", "kap", "ibp", "kbp"], name = "U_dag_node")
    #
    #         # assumption that tn.Node() orders the axes in the same order as
    #         # the input np.array
    #
    #         edge_a = U_node["ia"] ^ res_tensor[ia]
    #         edge_b = U_node["ib"] ^ res_tensor[ib]
    #
    #         new_edge_order = res_tensor.edges[:ia] + [U_node["ja"]] +\
    #                          res_tensor.edges[ia + 1: ib] + [U_node["jb"]] +\
    #                          res_tensor.edges[ib + 1:]
    #
    #         res_tensor = tn.contract_between(U_node, res_tensor,
    #                                     output_edge_order = new_edge_order)
    #
    #         edge_a_p = U_dag_node["iap"] ^ res_tensor[iap]
    #         edge_b_p = U_dag_node["ibp"] ^ res_tensor[ibp]
    #
    #         new_edge_order = res_tensor.edges[:iap] + [U_dag_node["kap"]] +\
    #                          res_tensor.edges[iap + 1: ibp] + [U_dag_node["kbp"]] +\
    #                          res_tensor.edges[ibp + 1:]
    #
    #         res_tensor = tn.contract_between(U_dag_node, res_tensor,
    #                                     output_edge_order = new_edge_order)
    #
    #     return res_tensor
    #
    # def mixing_layer(self, layer_num: int, var_tensor: tn.Node, mode: str = "dual"):
    #
    #     res_tensor = var_tensor
    #
    #     #----- Applying the mixing unitary -----#
    #     beta = self.opt_beta[layer_num]
    #
    #     Ux = jnp.cos(beta) * self.I - 1j * jnp.sin(beta) * self.X
    #     Ux_dag = jnp.transpose(jnp.conj(Ux))
    #
    #     # taking the adjoint of U and U_dag to impose dual channel on Sigmas
    #     if mode == 'dual':
    #         Ux = jnp.conj(jnp.transpose(Ux))
    #         Ux_dag = jnp.conj(jnp.transpose(Ux_dag))
    #
    #     for site_num in range(self.num_sites_in_lattice):
    #
    #         Ux_node = tn.Node(Ux, axis_names = ["ja","ia"])
    #         Ux_dag_node = tn.Node(Ux_dag, axis_names = ["iap","kap"])
    #
    #         ia  = 2 * site_num
    #         iap = 2 * site_num + 1
    #
    #         edge_a      = Ux_node["ia"] ^ res_tensor[ia]
    #         edge_a_p    = Ux_dag_node["iap"] ^ res_tensor[iap]
    #
    #         # perform the contraction
    #         new_edge_order = res_tensor.edges[:ia] + [Ux_node["ja"]] +\
    #                          res_tensor.edges[ia + 1:]
    #         res_tensor = tn.contract_between(Ux_node, res_tensor,
    #                                 output_edge_order = new_edge_order)
    #
    #         new_edge_order = res_tensor.edges[:iap] + [Ux_dag_node["kap"]] +\
    #                          res_tensor.edges[iap + 1:]
    #         res_tensor = tn.contract_between(Ux_dag_node, res_tensor,
    #                                 output_edge_order = new_edge_order)
    #
    #     return res_tensor
    #
    # def circuit_layer(self, layer_num: int, var_tensor: tn.Node, mode: str = "dual"):
    #
    #     res_tensor = var_tensor
    #
    #     if mode == "dual":
    #         res_tensor = self.mixing_layer(layer_num, res_tensor, "dual")
    #         res_tensor = self.problem_layer(layer_num, res_tensor, "dual")
    #
    #     else:
    #         res_tensor = self.problem_layer(layer_num, res_tensor, "primal")
    #         res_tensor = self.mixing_layer(layer_num, res_tensor, "primal")
    #
    #     return res_tensor
    #
    # def noisy_circuit_layer(self, i: int):
    #
    #     """
    #     Applies the unitary corresponding to a circuit layer followed by noise
    #     to the dual variable Sigma. The noise model is depolarizing noise on
    #     each qubit.
    #
    #     Params:
    #         i: layer number. Used to specify the tensor to act on and the gate
    #         params to use. Note the action when i = 0 is on the init state not
    #         the Sigma variable.
    #     Returns:
    #         res = tn.Node of shape self.dim_list that contains the tensor
    #         after the action of the layer.
    #     """
    #
    #     # if i == 0:
    #     #     res_tensor = self.circuit_layer(layer_num = i,
    #     #                                     var_tensor = self.rho_init_tensor)
    #     # else:
    #     #     res_tensor = self.circuit_layer(layer_num = i,
    #     #                                     var_tensor = self.Sigmas[i])
    #
    #     res_tensor = self.Sigmas[i]
    #     res_tensor = self.noise_layer(res_tensor)
    #     res_tensor = self.circuit_layer(layer_num = i,
    #                                     var_tensor = res_tensor,
    #                                     mode = 'dual')
    #
    #     return res_tensor
    #
    # def primal_noisy(self):
    #
    #     var_tensor = self.rho_init_tensor
    #
    #     for i in range(self.d):
    #
    #         var_tensor = self.circuit_layer(layer_num = i,
    #                                         var_tensor = var_tensor,
    #                                         mode = 'primal')
    #         var_tensor = self.noise_layer(var_tensor)
    #
    #     var_mat = self.tensor_2_mat(var_tensor.tensor)
    #
    #     return var_mat, jnp.trace(jnp.matmul(var_mat, self.H_problem))
