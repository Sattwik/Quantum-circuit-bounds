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

class SumSigma1D(meta_system.System):

    def __init__(self, lattice,  d: int, p: float, circ_backend = "qutip"):

        self.circ_backend = circ_backend

        self.lattice = lattice
        self.graph = lattice

        # preparing the Z operators
        self.site_z_ops = {}
        self.site_x_ops = {}
        self.site_y_ops = {}

        self.site_nums = {}
        op_num = 0

        self.num_sites_in_lattice = self.lattice.number_of_nodes() # assuming even
        # layer numbering starts from 1
        self.site_tuple_list_even_layer = list(zip(range(0, self.N, 2), range(1, self.N, 2)))

        self.Z_qutip = qutip.sigmaz()
        self.X_qutip = qutip.sigmax()
        self.Y_qutip = qutip.sigmay()
        self.I_qutip = qutip.qeye(2)

        self.I_tot = qutip.tensor([self.I_qutip] * self.num_sites_in_lattice)

        # the problem Hamiltonian
        self.H = 0

        for site in self.lattice:

            self.site_z_ops[site] = qutip.tensor([self.I_qutip] * op_num + [self.Z_qutip] + \
                                            [self.I_qutip] * (self.num_sites_in_lattice - op_num - 1))
            self.site_x_ops[site] = qutip.tensor([self.I_qutip] * op_num + [self.X_qutip] + \
                                            [self.I_qutip] * (self.num_sites_in_lattice - op_num - 1))
            self.site_y_ops[site] = qutip.tensor([self.I_qutip] * op_num + [self.Y_qutip] + \
                                            [self.I_qutip] * (self.num_sites_in_lattice - op_num - 1))

            self.site_nums[site] = op_num
            op_num += 1

            self.H += self.site_z_ops[site]



        for s in self.graph.edges:

            Zj = self.site_z_ops[edge[0]]
            Zk = self.site_z_ops[edge[1]]
            wjk = -1

            local_op = wjk/2 * (self.I_tot - Zj * Zk)

            self.H += local_op

        # for dual
        self.local_dim = 2
        self.d = d
        self.p = p
        self.psi_init = self.init_state()
        self.psi_init_jax = jnp.array(self.psi_init)
        self.H_problem = jnp.array(self.H.full())

        self.dim_list = tuple([self.local_dim] * 2 * self.num_sites_in_lattice)
        self.dim = self.local_dim ** self.num_sites_in_lattice

        self.utri_indices = jnp.triu_indices(self.dim, 1)
        self.ltri_indices = (self.utri_indices[1], self.utri_indices[0])
        self.num_tri_elements = self.utri_indices[0].shape[0]
        self.num_diag_elements = self.dim

        self.total_num_vars = (1 + self.num_diag_elements + 2 * self.num_tri_elements) * self.d

        self.I_jax = jnp.array(self.I_qutip.full())
        self.X_jax = jnp.array(self.X_qutip.full())
        self.Y_jax = jnp.array(self.Y_qutip.full())
        self.Z_jax = jnp.array(self.Z_qutip.full())

        # # constructing initial density operator
        self.rho_init = jnp.array((self.psi_init * self.psi_init.dag()).full())

        self.init_entropy_bounds()

    def update_opt_circ_params(self, opt_circuit_params: np.array):

        self.opt_gamma = opt_circuit_params[:self.d]
        self.opt_beta = opt_circuit_params[self.d:]

    def init_state(self):

        psi0 = qutip.basis([2] * self.num_sites_in_lattice)
        psi0 = qutip.hadamard_transform(N = self.num_sites_in_lattice) * psi0

        return psi0

    def circuit_layer_jax(self, state: jnp.array, layer_num: int, cicr_params: np.array):

        gamma = circ_params[:self.d]
        beta = circ_params[self.d:]

        U_2site = jnp.diag(jnp.array([1, jnp.exp(1j * gamma[layer_num - 1]), jnp.exp(1j * gamma[layer_num - 1]), 1]))

        if layer_num % 2 == 0:
            site_tuple_list = self.site_tuple_list_even_layer
        else:
            site_tuple_list = self.site_tuple_list_odd_layer

        for site_tuple in site_tuple_list:

            if site_tuple in self.graph.edges:

                dim_left = self.local_dim ** site_tuple[0]
                dim_right = self.local_dim ** (self.num_sites_in_lattice - site_tuple[1] - 1)

                identity_left = jnp.identity(dim_left)
                identity_right = jnp.identity(dim_right)

                state = jnp.kron(identity_left, jnp.kron(U_2site, identity_right)) * state

        if layer_num % 2 == 0:

            U_1site = jnp.cos(beta[(layer_num - 2)//2]) * self.I_jax - 1j * jnp.sin(beta[(layer_num - 2)//2]) * self.X_jax

            for site in self.graph:

                site_num = self.site_nums[site]

                dim_left = self.local_dim ** site_num
                dim_right = self.local_dim ** (self.num_sites_in_lattice - site_num - 1)

                identity_left = jnp.identity(dim_left)
                identity_right = jnp.identity(dim_right)

                state = jnp.kron(identity_left, jnp.kron(U_1site, identity_right)) * state

        return state

    def circuit_layer(self, state, layer_num, circ_params):

        gamma = circ_params[:self.d]
        beta = circ_params[self.d:]

        H_2site = qutip.tensor([self.Z_qutip] * 2) - qutip.tensor([self.I_qutip] * 2)
        U_2site = (-1j * (gamma[layer_num - 1]/2) * H_2site).expm()

        if layer_num % 2 == 0:
            site_tuple_list = self.site_tuple_list_even_layer
        else:
            site_tuple_list = self.site_tuple_list_odd_layer

        for site_tuple in site_tuple_list:

            if site_tuple in self.graph.edges:

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

    def init_entropy_bounds(self):

        q = 1 - self.p
        q_powers = jnp.array([q**i for i in range(self.d)])

        self.entropy_bounds = self.num_sites_in_lattice * self.p * jnp.log(2) * \
                              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(self.d)])

    def mat_2_tensor(self, A: jnp.array):

        """
        Converts an array A that is indexed as (i1 i2 ... iN)(i1p i2p ... iNp)
        into a tensor indexed as (i1) (i1p) (i2) (i2p) ... (iN) (iNp). () denote
        composite indices.
        """

        # (i1 i2 ... iN)(i1p i2p ... iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
        T = jnp.ravel(A)

        # (i1 i2 ... iN i1p i2p ... iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
        T = jnp.reshape(T, tuple([self.local_dim] * self.num_sites_in_lattice * 2))

        # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1) (i1p) ... (iN) (iNp)
        # [0, N, 1, N + 1, ..., N - 1, 2N - 1]
        i1 = np.arange(self.num_sites_in_lattice)
        i2 = self.num_sites_in_lattice + i1
        i  = np.zeros(2 * self.num_sites_in_lattice, dtype = int)
        i[::2] = i1
        i[1::2] = i2
        # i = i.at[::2].set(i1)
        # i = i.at[1::2].set(i2)

        T = jnp.transpose(T, tuple(i))

        return T

    def tensor_2_mat(self, T: jnp.array):

        """
        Converts a tensor T that is indexed as  (i1) (i1p) (i2) (i2p) ... (iN) (iNp)
        into a matrix indexed as (i1 i2 ... iN)(i1p i2p ... iNp). () denote
        composite indices.
        """

        # (i1) (i1p) (i2) (i2p) ... (iN) (iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
        # [0, 2, 4, ..., 2N - 2, 1, 3, ..., 2N - 1]
        i1 = np.arange(0, 2 * self.num_sites_in_lattice - 1, 2)
        i2 = np.arange(1, 2 * self.num_sites_in_lattice, 2)
        i = np.concatenate((i1, i2))

        A = jnp.transpose(T, tuple(i))

        # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
        A = jnp.ravel(A)

        # (i1 i2 ... iN i1p i2p ... iNp) -> (i1 i2 ... iN)(i1p i2p ... iNp)
        A = jnp.reshape(A, (self.dim, self.dim))

        return A

    def dual_obj(self, dual_vars: jnp.array):

        # Unvectorize the vars_vec and construct Sigmas and Lambdas
        self.assemble_vars_into_tensors(dual_vars)

        # Compute the objective function using the list of tensors
        obj = self.cost()

        return -jnp.real(obj)

    def dual_grad(self, dual_vars: jnp.array):

        return grad(self.dual_obj)(dual_vars)

    def assemble_vars_into_tensors(self, vars_vec: jnp.array):

        vars_diag_list, vars_real_list, vars_imag_list = self.unvectorize_vars(vars_vec)
        self.Sigmas = []

        for i in range(self.d):

            tri_real = jnp.zeros((self.dim, self.dim), dtype = complex)
            tri_imag = jnp.zeros((self.dim, self.dim), dtype = complex)

            tri_real = tri_real.at[self.utri_indices].set(vars_real_list[i])
            tri_real = tri_real.at[self.ltri_indices].set(vars_real_list[i])
            tri_imag = tri_imag.at[self.utri_indices].set(1j * vars_imag_list[i])
            tri_imag = tri_imag.at[self.ltri_indices].set(-1j * vars_imag_list[i])

            vars_full = jnp.diag(vars_diag_list[i]) + tri_real + tri_imag

            self.Sigmas.append(tn.Node(self.mat_2_tensor(vars_full)))

    def unvectorize_vars(self, vars_vec: jnp.array):

        a_vars = vars_vec.at[:self.d].get()
        self.Lambdas = jnp.log(1 + jnp.exp(a_vars))

        vars_vec_split = jnp.split(vars_vec.at[self.d:].get(), self.d)
        # split the variables into p equal arrays

        vars_diag_list = []
        vars_real_list = []
        vars_imag_list = []

        for i in range(self.d):

            # for each circuit step the variables are arranged as:
            # [vars_diag, vars_real, vars_imag]

            vars_diag = vars_vec_split[i].at[:self.num_diag_elements].get()
            vars_real = vars_vec_split[i].at[self.num_diag_elements:self.num_diag_elements + self.num_tri_elements].get()
            vars_imag = vars_vec_split[i].at[self.num_diag_elements + self.num_tri_elements:].get()

            vars_diag_list.append(vars_diag)
            vars_real_list.append(vars_real)
            vars_imag_list.append(vars_imag)

        return vars_diag_list, vars_real_list, vars_imag_list

    def cost(self):

        # the entropy term
        cost = jnp.dot(self.Lambdas, self.entropy_bounds)

        # the effective Hamiltonian term
        for i in range(self.d):

            # i corresponds to the layer of the circuit, 0 being the earliest
            # i.e. i = 0 corresponds to t = 1 in notes

            # construct the effective Hamiltonian for the ith step/layer
            Hi = self.construct_H(i)
            Ei = jnp.linalg.eigvalsh(Hi)

            cost += -self.Lambdas[i] * jnp.log(jnp.sum(jnp.exp(-Ei/self.Lambdas[i])))

        # the initial state term
        epsilon1_dag_sigma1 = self.tensor_2_mat(self.noisy_dual_layer(0).tensor)

        cost += -jnp.trace(jnp.matmul(self.rho_init, epsilon1_dag_sigma1))

        return cost

    def construct_H(self, i: int):

        """
        Constructs, from the Sigma dual variables, the effective Hamiltonian
        corresponding to a layer.

        Params:
            i: layer number (>= 0)
        Returns:
            Hi: np.array of shape (self.dim, self.dim)
        """

        if i == self.d - 1:
            Hi = self.tensor_2_mat(self.Sigmas[i].tensor) + self.H_problem

            return Hi

        else:
            Hi = self.Sigmas[i].tensor - self.noisy_dual_layer(i + 1).tensor

            return self.tensor_2_mat(Hi)

    def noisy_dual_layer(self, i: int):

        """
        Applies the noise followed by unitary corresponding to a circuit layer
        to the dual variable Sigma. The noise model is depolarizing noise on
        each qubit.

        Params:
            i: layer number. Used to specify the tensors to act on and the gate
            params to use.
        Returns:
            res_tensors = tensors after the dual layer
        """

        res_tensor = self.Sigmas[i]
        res_tensor = self.noise_layer(res_tensor)
        res_tensor = self.dual_unitary_layer(layer_num = i + 1,
                                        var_tensor = res_tensor)
                                        # i + 1 here because unitary_layer
                                        # counts layers differently

        return res_tensor

    def noise_layer(self, var_tensor: tn.Node):

        """
        Applies depolarizing noise on the var_tensor at all sites.
        """

        res_tensor = var_tensor

        for site_num in range(self.num_sites_in_lattice):

            # --- applying I --- #
            res_array = (1 - 3 * self.p/4) * res_tensor.tensor

            # --- applying X --- #
            tmp_tensor = res_tensor

            X_node = tn.Node(np.sqrt(self.p/4) * self.X_jax, axis_names = ["ja","ia"])
            X_prime_node = tn.Node(np.sqrt(self.p/4) * self.X_jax, axis_names = ["iap","kap"])

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

            Y_node = tn.Node(np.sqrt(self.p/4) * self.Y_jax, axis_names = ["ja","ia"])
            Y_prime_node = tn.Node(np.sqrt(self.p/4) * self.Y_jax, axis_names = ["iap","kap"])

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

            Z_node = tn.Node(np.sqrt(self.p/4) * self.Z_jax, axis_names = ["ja","ia"])
            Z_prime_node = tn.Node(np.sqrt(self.p/4) * self.Z_jax, axis_names = ["iap","kap"])

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

    def dual_unitary_layer(self, layer_num: int, var_tensor: tn.Node):

        res_tensor = var_tensor

        if layer_num % 2 == 0:
            res_tensor = self.mixing_unitaries(layer_num, res_tensor)

        res_tensor = self.problem_unitaries(layer_num, res_tensor)

        return res_tensor

    def problem_unitaries(self, layer_num: int, var_tensor: tn.Node):

        #----- Applying the problem unitary -----#
        gamma = self.opt_gamma[layer_num - 1]

        # U = exp(-i gamma/2 w * (I - Z_j Z_k))
        U = jnp.diag(jnp.array([1, jnp.exp(1j * gamma), jnp.exp(1j * gamma), 1]))
        U_dag = jnp.transpose(jnp.conj(U))

        # adjoint because dual
        U = jnp.conj(jnp.transpose(U))
        U_dag = jnp.conj(jnp.transpose(U_dag))

        # (ja jb) (ia ib) -> (ja jb ia ib)
        U_tensor = jnp.ravel(U)
        # (ja jb ia ib) -> (ja) (jb) (ia) (ib)
        U_tensor = jnp.reshape(U_tensor, (2,2,2,2))
        # (ja) (jb) (ia) (ib) -> (ja) (ia) (jb) (ib)
        U_tensor = jnp.transpose(U_tensor, [0, 2, 1, 3])

        # (iap ibp) (kap kbp) -> (iap ibp kap kbp)
        U_dag_tensor = jnp.ravel(U_dag)
        # (iap ibp kap kbp) -> (iap) (ibp) (kap) (kbp)
        U_dag_tensor = jnp.reshape(U_dag_tensor, (2,2,2,2))
        # (iap) (ibp) (kap) (kbp) -> (iap) (kap) (ibp) (kbp)
        U_dag_tensor = jnp.transpose(U_dag_tensor, [0, 2, 1, 3])

        res_tensor = var_tensor

        if layer_num%2 == 0:
            site_tuple_list = self.site_tuple_list_even_layer
        else:
            site_tuple_list = self.site_tuple_list_odd_layer

        for site_tuple in site_tuple_list:

            if site_tuple in self.graph.edges:

                site_num_a = min(site_tuple[0], site_tuple[1])
                site_num_b = max(site_tuple[0], site_tuple[1])

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

                res_tensor = tn.contract_between(U_node, res_tensor,
                                            output_edge_order = new_edge_order)

                edge_a_p = U_dag_node["iap"] ^ res_tensor[iap]
                edge_b_p = U_dag_node["ibp"] ^ res_tensor[ibp]

                new_edge_order = res_tensor.edges[:iap] + [U_dag_node["kap"]] +\
                                 res_tensor.edges[iap + 1: ibp] + [U_dag_node["kbp"]] +\
                                 res_tensor.edges[ibp + 1:]

                res_tensor = tn.contract_between(U_dag_node, res_tensor,
                                            output_edge_order = new_edge_order)

        return res_tensor

    def mixing_unitaries(self, layer_num: int, var_tensor: tn.Node):

        assert layer_num % 2 == 0

        #----- Applying the mixing unitary -----#
        beta = self.opt_beta[(layer_num - 2)//2]

        Ux = jnp.cos(beta) * self.I_jax - 1j * jnp.sin(beta) * self.X_jax
        Ux_dag = jnp.transpose(jnp.conj(Ux))

        # adjoint because dual
        Ux = jnp.conj(jnp.transpose(Ux))
        Ux_dag = jnp.conj(jnp.transpose(Ux_dag))

        res_tensor = var_tensor

        for site_num in range(self.num_sites_in_lattice):

            Ux_node = tn.Node(Ux, axis_names = ["ja","ia"])
            Ux_dag_node = tn.Node(Ux_dag, axis_names = ["iap","kap"])

            ia  = 2 * site_num
            iap = 2 * site_num + 1

            edge_a      = Ux_node["ia"] ^ res_tensor[ia]
            edge_a_p    = Ux_dag_node["iap"] ^ res_tensor[iap]

            # perform the contraction
            new_edge_order = res_tensor.edges[:ia] + [Ux_node["ja"]] +\
                             res_tensor.edges[ia + 1:]
            res_tensor = tn.contract_between(Ux_node, res_tensor,
                                    output_edge_order = new_edge_order)

            new_edge_order = res_tensor.edges[:iap] + [Ux_dag_node["kap"]] +\
                             res_tensor.edges[iap + 1:]
            res_tensor = tn.contract_between(Ux_dag_node, res_tensor,
                                    output_edge_order = new_edge_order)

        return res_tensor

    def primal_noisy(self):

        rho_init_qutip = self.psi_init * self.psi_init.dag()

        gamma = self.opt_gamma
        beta = self.opt_beta

        rho_after_step = rho_init_qutip

        for layer_num in range(1, self.d + 1):

            H_2site = qutip.tensor([self.Z_qutip] * 2) - qutip.tensor([self.I_qutip] * 2)
            U_2site = (-1j * (gamma[layer_num - 1]/2) * H_2site).expm()

            # problem unitaries
            if layer_num % 2 == 0:
                site_index_list = self.site_tuple_list_even_layer
            else:
                site_index_list = self.site_tuple_list_odd_layer

            for site_tuple in site_index_list:

                if site_tuple in self.graph.edges:

                    U_2site_full = qutip.tensor(
                                    [self.I_qutip] * site_tuple[0] +
                                    [U_2site] +
                                    [self.I_qutip] * (self.num_sites_in_lattice - site_tuple[0] - 2))
                    rho_after_step = U_2site_full * rho_after_step * U_2site_full.dag()

            # mixing unitaries
            if layer_num % 2 == 0:

                U_1site = (-1j * beta[(layer_num - 2)//2] * self.X_qutip).expm()

                for site in self.graph:

                    site_num = self.site_nums[site]

                    U_1site_full = qutip.tensor([self.I_qutip] * site_num + [U_1site] + \
                            [self.I_qutip] * (self.num_sites_in_lattice - site_num - 1))

                    rho_after_step = U_1site_full * rho_after_step * U_1site_full.dag()

            # noise
            for site in self.lattice:

                rho_after_step = (1 - 3 * self.p/4) * rho_after_step \
                + (self.p/4) * (self.site_x_ops[site] * rho_after_step * self.site_x_ops[site] + \
                                self.site_y_ops[site] * rho_after_step * self.site_y_ops[site] + \
                                self.site_z_ops[site] * rho_after_step * self.site_z_ops[site])

        return qutip.expect(self.H, rho_after_step)

class MaxCut1DNoChannel():
    def __init__(self, graph, lattice, d: int, p: float, circ_backend = "qutip"):

        self.graph = graph
        self.lattice = lattice

        self.circ_backend = circ_backend

        # preparing the Z operators
        self.site_z_ops = {}
        self.site_x_ops = {}
        self.site_y_ops = {}

        self.site_nums = {}
        op_num = 0

        self.num_sites_in_lattice = self.lattice.number_of_nodes() # assuming even
        # layer numbering starts from 1
        self.site_tuple_list_odd_layer = list(zip(range(1, self.num_sites_in_lattice, 2), range(2, self.num_sites_in_lattice, 2)))
        self.site_tuple_list_even_layer = list(zip(range(0, self.num_sites_in_lattice, 2), range(1, self.num_sites_in_lattice, 2)))

        self.Z_qutip = qutip.sigmaz()
        self.X_qutip = qutip.sigmax()
        self.Y_qutip = qutip.sigmay()
        self.I_qutip = qutip.qeye(2)

        self.I_tot = qutip.tensor([self.I_qutip] * self.num_sites_in_lattice)

        for site in self.lattice:

            self.site_z_ops[site] = qutip.tensor([self.I_qutip] * op_num + [self.Z_qutip] + \
                                            [self.I_qutip] * (self.num_sites_in_lattice - op_num - 1))
            self.site_x_ops[site] = qutip.tensor([self.I_qutip] * op_num + [self.X_qutip] + \
                                            [self.I_qutip] * (self.num_sites_in_lattice - op_num - 1))
            self.site_y_ops[site] = qutip.tensor([self.I_qutip] * op_num + [self.Y_qutip] + \
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
        self.local_dim = 2
        self.d = d
        self.p = p
        self.psi_init = self.init_state()
        self.psi_init_jax = jnp.array(self.psi_init)
        self.H_problem = jnp.array(self.H.full())

        self.dim_list = tuple([self.local_dim] * 2 * self.num_sites_in_lattice)
        self.dim = self.local_dim ** self.num_sites_in_lattice

        self.utri_indices = jnp.triu_indices(self.dim, 1)
        self.ltri_indices = (self.utri_indices[1], self.utri_indices[0])
        self.num_tri_elements = self.utri_indices[0].shape[0]
        self.num_diag_elements = self.dim

        self.total_num_vars = (1 + self.num_diag_elements + 2 * self.num_tri_elements) * self.d

        self.I_jax = jnp.array(self.I_qutip.full())
        self.X_jax = jnp.array(self.X_qutip.full())
        self.Y_jax = jnp.array(self.Y_qutip.full())
        self.Z_jax = jnp.array(self.Z_qutip.full())

        # # constructing initial density operator
        self.rho_init = jnp.array((self.psi_init * self.psi_init.dag()).full())

        self.init_entropy_bounds()

    def update_opt_circ_params(self, opt_circuit_params: np.array):

        self.opt_gamma = opt_circuit_params[:self.d]
        self.opt_beta = opt_circuit_params[self.d:]

    def init_state(self):

        psi0 = qutip.basis([2] * self.num_sites_in_lattice)
        psi0 = qutip.hadamard_transform(N = self.num_sites_in_lattice) * psi0

        return psi0

    def init_entropy_bounds(self):

        q = 1 - self.p
        q_powers = jnp.array([q**i for i in range(self.d)])

        self.entropy_bounds = self.num_sites_in_lattice * self.p * jnp.log(2) * \
                              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(self.d)])

    def mat_2_tensor(self, A: jnp.array):

        """
        Converts an array A that is indexed as (i1 i2 ... iN)(i1p i2p ... iNp)
        into a tensor indexed as (i1) (i1p) (i2) (i2p) ... (iN) (iNp). () denote
        composite indices.
        """

        # (i1 i2 ... iN)(i1p i2p ... iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
        T = jnp.ravel(A)

        # (i1 i2 ... iN i1p i2p ... iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
        T = jnp.reshape(T, tuple([self.local_dim] * self.num_sites_in_lattice * 2))

        # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1) (i1p) ... (iN) (iNp)
        # [0, N, 1, N + 1, ..., N - 1, 2N - 1]
        i1 = np.arange(self.num_sites_in_lattice)
        i2 = self.num_sites_in_lattice + i1
        i  = np.zeros(2 * self.num_sites_in_lattice, dtype = int)
        i[::2] = i1
        i[1::2] = i2
        # i = i.at[::2].set(i1)
        # i = i.at[1::2].set(i2)

        T = jnp.transpose(T, tuple(i))

        return T

    def tensor_2_mat(self, T: jnp.array):

        """
        Converts a tensor T that is indexed as  (i1) (i1p) (i2) (i2p) ... (iN) (iNp)
        into a matrix indexed as (i1 i2 ... iN)(i1p i2p ... iNp). () denote
        composite indices.
        """

        # (i1) (i1p) (i2) (i2p) ... (iN) (iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
        # [0, 2, 4, ..., 2N - 2, 1, 3, ..., 2N - 1]
        i1 = np.arange(0, 2 * self.num_sites_in_lattice - 1, 2)
        i2 = np.arange(1, 2 * self.num_sites_in_lattice, 2)
        i = np.concatenate((i1, i2))

        A = jnp.transpose(T, tuple(i))

        # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
        A = jnp.ravel(A)

        # (i1 i2 ... iN i1p i2p ... iNp) -> (i1 i2 ... iN)(i1p i2p ... iNp)
        A = jnp.reshape(A, (self.dim, self.dim))

        return A

    def dual_obj(self, dual_nc_vars: jnp.array):

        lmbda = jnp.log(1 + jnp.exp(dual_nc_vars[0]))

        cost = lmbda * self.entropy_bounds[-1]
        Hi = self.H_problem
        Ei = jnp.linalg.eigvalsh(Hi)

        cost += -lmbda * jnp.log(jnp.sum(jnp.exp(-Ei/lmbda)))

        return -jnp.real(cost)

    def dual_grad(self, dual_nc_vars: jnp.array):

        return grad(self.dual_obj)(dual_nc_vars)
