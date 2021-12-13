from typing import List, Tuple, Callable, Dict
from functools import partial
import abc

import tensornetwork as tn
tn.set_default_backend("jax")
import numpy as np
import scipy
import qutip
import jax.numpy as jnp
from jax import jit, grad, vmap, value_and_grad
from jax.example_libraries import optimizers

from vqa import graphs
from vqa import problems
from vqa import algorithms
from vqa import dual

class MaxCutDualJAX():

    """
    A class to calculate the dual function and its gradient for the MaxCut QAOA
    problem.

    Members:

        self.Sigmas: list of tensors each with 2N legs of dimension local_dim
                (local_dim = 2 by default). If each sigma is self-adjoint, the
                dual objective function is real.

        self.Lambdas: list of positive real numbers.

    Params:
        prob_obj = Problem object (NB, none of the unitary methods
        from this object are utilised here; the unitary/noise operations are
        implemented from scratch for this class.)

        d = number of layers

        gamma = params of problem layer

        beta = params of mixing layer

        p: depolarizing noise probability on each qubit
    """

    def __init__(self, prob_obj: problems.Problem,
                 d: int, gamma: jnp.array, beta: jnp.array, p: float):

        self.prob_obj = prob_obj
        self.num_sites_in_lattice = self.prob_obj.num_sites_in_lattice
        self.local_dim = 2
        self.d = d
        self.gamma = gamma
        self.beta = beta
        self.p = p
        self.psi_init = prob_obj.init_state()
        self.H_problem = jnp.array(self.prob_obj.H.full())

        self.dim_list = tuple([self.local_dim] * 2 * self.num_sites_in_lattice)
        self.dim = self.local_dim ** self.num_sites_in_lattice

        self.utri_indices = jnp.triu_indices(self.dim, 1)
        self.ltri_indices = (self.utri_indices[1], self.utri_indices[0])
        self.num_tri_elements = self.utri_indices[0].shape[0]
        self.num_diag_elements = self.dim

        # constructing initial density operator
        self.rho_init = (self.psi_init * self.psi_init.dag()).full()
        self.rho_init_tensor = tn.Node(self.mat_2_tensor(self.rho_init), name = "rho_init")

        self.X = jnp.array(qutip.sigmax().full())
        self.Y = jnp.array(qutip.sigmay().full())
        self.Z = jnp.array(qutip.sigmaz().full())
        self.I = jnp.array(qutip.qeye(2).full())

        self.init_entropy_bounds()

        self.len_vars = (1 + self.num_diag_elements + 2 * self.num_tri_elements) * self.d

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

    def init_entropy_bounds(self):

        q = 1 - self.p
        q_powers = jnp.array([q**i for i in range(self.d)])

        self.entropy_bounds = self.num_sites_in_lattice * self.p * jnp.log(2) * \
                              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(self.d)])

    def objective(self, vars_vec: jnp.array):

        # Unvectorize the vars_vec and construct Sigmas and Lambdas
        self.assemble_vars_into_tensors(vars_vec)

        # Compute the objective function using the list of tensors
        obj = self.cost()

        return -jnp.real(obj)

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
            Ei = jnp.linalg.eigvals(Hi)

            cost += -self.Lambdas[i] * jnp.log(jnp.sum(jnp.exp(-Ei/self.Lambdas[i])))

        # the initial state term
        # TODO: will this be faster if expressed in terms of contractions?
        sigma1 = self.tensor_2_mat(self.Sigmas[0].tensor)
        epsilon_1_rho = self.rho_init_tensor
        epsilon_1_rho = self.circuit_layer(layer_num = 0,
                                           var_tensor = epsilon_1_rho,
                                           mode = 'primal')
        epsilon_1_rho = self.noise_layer(epsilon_1_rho)
        epsilon_1_rho = self.tensor_2_mat(epsilon_1_rho.tensor)
        cost += -jnp.trace(jnp.matmul(sigma1, epsilon_1_rho))

        # rho_init_mat = self.tensor_2_mat(self.rho_init_tensor.tensor)
        # epsilon_1_dag_sigma1 = self.tensor_2_mat(self.noisy_circuit_layer(i = 0).tensor)
        # cost += -jnp.trace(jnp.matmul(rho_init_mat, epsilon_1_dag_sigma1))

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
            # TODO: check that the tensor product/site num ordering is
            # consistent
            Hi = self.tensor_2_mat(self.Sigmas[i].tensor) + self.H_problem

            return Hi

        else:
            Hi = self.Sigmas[i].tensor - self.noisy_circuit_layer(i + 1).tensor

            return self.tensor_2_mat(Hi)

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

            X_node = tn.Node(np.sqrt(self.p/4) * self.X, axis_names = ["ja","ia"])
            X_prime_node = tn.Node(np.sqrt(self.p/4) * self.X, axis_names = ["iap","kap"])

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

            Y_node = tn.Node(np.sqrt(self.p/4) * self.Y, axis_names = ["ja","ia"])
            Y_prime_node = tn.Node(np.sqrt(self.p/4) * self.Y, axis_names = ["iap","kap"])

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

            Z_node = tn.Node(np.sqrt(self.p/4) * self.Z, axis_names = ["ja","ia"])
            Z_prime_node = tn.Node(np.sqrt(self.p/4) * self.Z, axis_names = ["iap","kap"])

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

    def problem_layer(self, layer_num: int, var_tensor: tn.Node, mode: str = "dual"):

        res_tensor = var_tensor

        #----- Applying the problem unitary -----#
        gamma = self.gamma[layer_num]

        # U = exp(-i gamma/2 w * (I - Z_j Z_k))
        U = jnp.diag(jnp.array([1, jnp.exp(1j * gamma), jnp.exp(1j * gamma), 1]))
        U_dag = jnp.transpose(jnp.conj(U))

        # taking the adjoint of U and U_dag to impose dual channel on Sigmas
        if mode == 'dual':
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

        for edge in self.prob_obj.graph.edges:

            site_num_a = min([self.prob_obj.site_nums[edge[0]], self.prob_obj.site_nums[edge[1]]])
            site_num_b = max([self.prob_obj.site_nums[edge[0]], self.prob_obj.site_nums[edge[1]]])

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

    def mixing_layer(self, layer_num: int, var_tensor: tn.Node, mode: str = "dual"):

        res_tensor = var_tensor

        #----- Applying the mixing unitary -----#
        beta = self.beta[layer_num]

        Ux = jnp.cos(beta) * self.I - 1j * jnp.sin(beta) * self.X
        Ux_dag = jnp.transpose(jnp.conj(Ux))

        # taking the adjoint of U and U_dag to impose dual channel on Sigmas
        if mode == 'dual':
            Ux = jnp.conj(jnp.transpose(Ux))
            Ux_dag = jnp.conj(jnp.transpose(Ux_dag))

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

    def circuit_layer(self, layer_num: int, var_tensor: tn.Node, mode: str = "dual"):

        res_tensor = var_tensor

        if mode == "dual":
            res_tensor = self.mixing_layer(layer_num, res_tensor, "dual")
            res_tensor = self.problem_layer(layer_num, res_tensor, "dual")

        else:
            res_tensor = self.problem_layer(layer_num, res_tensor, "primal")
            res_tensor = self.mixing_layer(layer_num, res_tensor, "primal")

        return res_tensor

    def noisy_circuit_layer(self, i: int):

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

        # if i == 0:
        #     res_tensor = self.circuit_layer(layer_num = i,
        #                                     var_tensor = self.rho_init_tensor)
        # else:
        #     res_tensor = self.circuit_layer(layer_num = i,
        #                                     var_tensor = self.Sigmas[i])

        res_tensor = self.Sigmas[i]
        res_tensor = self.noise_layer(res_tensor)
        res_tensor = self.circuit_layer(layer_num = i,
                                        var_tensor = res_tensor,
                                        mode = 'dual')

        return res_tensor

    def primal_noisy(self):

        var_tensor = self.rho_init_tensor

        for i in range(self.d):

            var_tensor = self.circuit_layer(layer_num = i,
                                            var_tensor = var_tensor,
                                            mode = 'primal')
            var_tensor = self.noise_layer(var_tensor)

        var_mat = self.tensor_2_mat(var_tensor.tensor)

        return var_mat, jnp.trace(jnp.matmul(var_mat, self.H_problem))

class MaxCutDualJAXGlobal():

    """
    A class to calculate the dual function and its gradient for a modified
    version of the MaxCut QAOA problem. The channel is assumed to be just a
    global depol. channel (no unitaries).

    Members:
        self.Sigmas: list of tensors each with 2N legs of dimension local_dim
                (local_dim = 2 by default). If each sigma is self-adjoint, the
                dual objective function is real.

        self.Lambdas: list of positive real numbers.

    Params:
        prob_obj = Problem object (NB, none of the unitary methods
        from this object are utilised here; the unitary/noise operations are
        implemented from scratch for this class.)

        d = number of layers

        p = global depolarizing noise probability
    """

    def __init__(self, prob_obj: problems.Problem, d: int, p: float):

        self.prob_obj = prob_obj
        self.num_sites_in_lattice = self.prob_obj.num_sites_in_lattice
        self.local_dim = 2
        self.d = d

        self.p = p
        self.psi_init = prob_obj.init_state()
        self.H_problem = jnp.array(self.prob_obj.H.full())

        self.dim_list = tuple([self.local_dim] * 2 * self.num_sites_in_lattice)
        self.dim = self.local_dim ** self.num_sites_in_lattice

        self.utri_indices = jnp.triu_indices(self.dim, 1)
        self.ltri_indices = (self.utri_indices[1], self.utri_indices[0])
        self.num_tri_elements = self.utri_indices[0].shape[0]
        self.num_diag_elements = self.dim

        # constructing initial density operator
        self.rho_init = (self.psi_init * self.psi_init.dag()).full()
        self.rho_init_tensor = tn.Node(self.mat_2_tensor(self.rho_init), name = "rho_init")

        self.X = jnp.array(qutip.sigmax().full())
        self.Y = jnp.array(qutip.sigmay().full())
        self.Z = jnp.array(qutip.sigmaz().full())
        self.I = jnp.array(qutip.qeye(2).full())

        self.init_entropy_bounds()

        self.len_vars = (1 + self.num_diag_elements + 2 * self.num_tri_elements) * self.d

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

    def init_entropy_bounds(self):

        """
        These bounds are in base e.
        """

        q = 1 - self.p
        q_powers = jnp.array([q**i for i in range(self.d)])

        self.entropy_bounds = self.num_sites_in_lattice * self.p * jnp.log(2) * \
                              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(self.d)])

    def objective(self, vars_vec: jnp.array):

        # Unvectorize the vars_vec and construct Sigmas and Lambdas
        self.assemble_vars_into_tensors(vars_vec)

        # Compute the objective function using the list of tensors
        obj = self.cost()

        return -jnp.real(obj)

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
            Ei = jnp.linalg.eigvals(Hi)

            cost += -self.Lambdas[i] * jnp.log(jnp.sum(jnp.exp(-Ei/self.Lambdas[i])))

        # the initial state term
        # TODO: will this be faster if expressed in terms of contractions?
        sigma1 = self.tensor_2_mat(self.Sigmas[0].tensor)
        epsilon_1_rho = self.tensor_2_mat(self.noisy_circuit_layer(i = 0).tensor)
        cost += -jnp.trace(jnp.matmul(sigma1, epsilon_1_rho))

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
            # TODO: check that the tensor product/site num ordering is
            # consistent
            Hi = self.tensor_2_mat(self.Sigmas[i].tensor) + self.H_problem

            return Hi

        else:
            Hi = self.Sigmas[i].tensor - self.noisy_circuit_layer(i + 1).tensor

            return self.tensor_2_mat(Hi)

    def noise_layer(self, var_tensor: tn.Node):

        """
        Applies global depolarizing noise on the var_tensor.
        """

        res_tensor = var_tensor
        res_mat = self.tensor_2_mat(res_tensor.tensor)
        identity = self.mat_2_tensor(jnp.identity(self.dim, dtype = complex))

        res_array = (1 - self.p) * res_tensor.tensor +\
                 (self.p) * jnp.trace(res_mat) * identity/self.dim

        res_tensor = tn.Node(res_array)

        return res_tensor

    def noisy_circuit_layer(self, i: int):

        """
        Applies the noise layer on the proper tensor for a given layer.
        The noise model is global depolarizing noise.

        Params:
            i: layer number. Used to specify the tensor to act on.
            Note the action when i = 0 is on the init state not
            the Sigma variable.
        Returns:
            res_tensor = tn.Node of shape self.dim_list that contains the tensor
            after the action of the layer.
        """

        if i == 0:
            res_tensor = self.rho_init_tensor
        else:
            res_tensor = self.Sigmas[i]

        res_tensor = self.noise_layer(res_tensor)

        return res_tensor

    def primal_noisy(self):

        var_tensor = self.rho_init_tensor

        for i in range(self.d):
            var_tensor = self.noise_layer(var_tensor)

        var_mat = self.tensor_2_mat(var_tensor.tensor)

        return jnp.trace(jnp.matmul(var_mat, self.H_problem))

class MaxCutDualJAXNoChannel():

    """
    A class to calculate the dual function and its gradient for a no-channel
    relaxation of the MaxCut QAOA problem.

    Members:
        lmbda

    Params:
        prob_obj = Problem object

        d = number of layers

        p = global depolarizing noise probability
    """

    def __init__(self, prob_obj: problems.Problem, d: int, p: float):

        self.prob_obj = prob_obj
        self.num_sites_in_lattice = self.prob_obj.num_sites_in_lattice
        self.local_dim = 2
        self.d = d

        self.p = p
        self.psi_init = prob_obj.init_state()
        self.H_problem = jnp.array(self.prob_obj.H.full())

        self.dim_list = tuple([self.local_dim] * 2 * self.num_sites_in_lattice)
        self.dim = self.local_dim ** self.num_sites_in_lattice

        self.init_entropy_bounds()

    def init_entropy_bounds(self):

        """
        These bounds are in base e.
        """

        q = 1 - self.p
        q_powers = jnp.array([q**i for i in range(self.d)])

        self.entropy_bounds = self.num_sites_in_lattice * self.p * jnp.log(2) * \
                              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(self.d)])

    def objective(self, vars_vec: jnp.array):

        # Compute the objective function using the list of tensors
        obj = self.cost(vars_vec)

        return -jnp.real(obj)

    def cost(self, vars_vec: jnp.array):

        a = vars_vec.at[0].get()
        lmbda = jnp.log(1 + jnp.exp(a))

        # the entropy term
        cost = lmbda * self.entropy_bounds[-1]

        # the Hamiltonian term
        Hi = self.H_problem
        Ei = jnp.linalg.eigvals(Hi)

        cost += -lmbda * jnp.log(jnp.sum(jnp.exp(-Ei/lmbda)))

        return cost

def adam_external_dual_JAX(vars_init: jnp.array, dual_obj: MaxCutDualJAX,
                           alpha: float, num_steps: int):

    init, update, get_params = optimizers.adam(alpha)
    params = vars_init

    def step(t, opt_state):
        value, grads = value_and_grad(dual_obj.objective)(get_params(opt_state))
        opt_state = update(t, grads, opt_state)
        return value, opt_state

    opt_state = init(params)

    value_array = jnp.zeros(num_steps)

    for t in range(num_steps):
        # print("Step :", t)
        value, opt_state = step(t, opt_state)
        value_array = value_array.at[t].set(value)

    return value_array, get_params(opt_state)

# @partial(jit, static_argnums = (1,))
# def objective_external_dual_JAX(vars_vec: jnp.array, dual_obj: MaxCutDualJAX):
#
#     return dual_obj.objective(vars_vec)
#
# @partial(jit, static_argnums = (1,))
# def gradient_external_dual_JAX(vars_vec: jnp.array, dual_obj: MaxCutDualJAX):
#
#     return grad(dual_obj.objective)(vars_vec)
#
# # @partial(jit, static_argnums = (1,2))
# def fd_gradient_dual_JAX(vars_vec: jnp.array, positions: Tuple, dual_obj: MaxCutDualJAX):
#
#     objective_0 = objective_external_dual_JAX(vars_vec, dual_obj)
#     delta = 1e-7
#
#     gradient_list = jnp.zeros(len(positions), dtype = complex)
#
#     for i in positions:
#
#         print(i)
#
#         vars_tmp = vars_vec
#         vars_tmp = vars_tmp.at[i].add(delta)
#         objective_plus = objective_external_dual_JAX(vars_tmp, dual_obj)
#
#         vars_tmp = vars_vec
#         vars_tmp = vars_tmp.at[i].add(-delta)
#         objective_minus = objective_external_dual_JAX(vars_tmp, dual_obj)
#
#         gradient_list = gradient_list.at[i].set((objective_plus - objective_minus)/(2 * delta))
#
#     return gradient_list
#
# def unjaxify_obj(func):
#
#     def wrap(*args):
#         return float(func(jnp.array(args[0]), args[1]))
#
#     return wrap
#
# def unjaxify_grad(func):
#
#     def wrap(*args):
#         return np.array(func(jnp.array(args[0]), args[1]), order = 'F')
#
#     return wrap
#
# def optimize_external_dual_JAX(vars_init: np.array, dual_obj: MaxCutDualJAX):
#
#     opt_args = (dual_obj,)
#
#     obj_over_opti = []
#
#     def callback_func(x):
#
#         obj_eval = unjaxify_obj(objective_external_dual_JAX)(x, opt_args[0])
#
#         obj_over_opti.append(obj_eval)
#
#         print('Dir. Iteration ', str(len(obj_over_opti)), '. Objective = ', str(obj_eval), '. x = ', x)
#
#     sigma_bound = 1e1
#     p = dual_obj.p
#     # len_vars = dual_obj.len_vars
#     len_vars = vars_init.shape[0]
#
#     bnds = scipy.optimize.Bounds(lb = [1e-2] * p + [-sigma_bound] * (len_vars - p), ub = [np.inf] * p + [-sigma_bound] * (len_vars - p))
#
#     opt_result = scipy.optimize.minimize(unjaxify_obj(objective_external_dual_JAX), vars_init, args = opt_args,
#                                          method = 'L-BFGS-B', jac = unjaxify_grad(gradient_external_dual_JAX), bounds = bnds,
#                                          options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09,
#                                          'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 3, 'iprint': 1, 'maxls': 20},
#                                          callback = callback_func)
#
#     return np.array(obj_over_opti), opt_result

# """
# Moving the action of the class MaxCutDual() to pure functions in order to be
# able to use JAX cleanly.
# """
#
# #------------ Problem parameters ------------#
#
# class MaxCutDualParams():
#
#     def __init__(self, prob_obj: problems.Problem,
#                  p: int, gamma: np.array, beta: np.array, p_noise: float):
#
#         dual_obj = dual.MaxCutDual(prob_obj = prob_obj, p = p, gamma = gamma, beta = beta, p_noise = p_noise)
#
# #------------ Tensor reshaping methods ------------#
#
# def mat_2_tensor(A: jnp.array, dim_list: Tuple, num_sites_in_lattice: int):
#
#     """
#     Converts an array A that is indexed as (i1 i2 ... iN)(i1p i2p ... iNp)
#     into a tensor indexed as (i1) (i1p) (i2) (i2p) ... (iN) (iNp). () denote
#     composite indices.
#     """
#
#     # (i1 i2 ... iN)(i1p i2p ... iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
#     T = jnp.flatten(A)
#
#     # (i1 i2 ... iN i1p i2p ... iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
#     T = jnp.reshape(T, dim_list)
#
#     # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1) (i1p) ... (iN) (iNp)
#     # [0, N, 1, N + 1, ..., N - 1, 2N - 1]
#     i1 = jnp.arange(num_sites_in_lattice)
#     i2 = num_sites_in_lattice + i1
#     i  = jnp.zeros(2 * num_sites_in_lattice, dtype = int)
#     i = i.at[::2].set(i1)
#     i = i.at[1::2].set(i2)
#
#     T = jnp.transpose(T, tuple(i))
#
#     return T
#
# def tensor_2_mat(T: jnp.array, dim: int, num_sites_in_lattice: int):
#
#     """
#     Converts a tensor T that is indexed as  (i1) (i1p) (i2) (i2p) ... (iN) (iNp)
#     into a matrix indexed as (i1 i2 ... iN)(i1p i2p ... iNp). () denote
#     composite indices.
#     """
#
#     # (i1) (i1p) (i2) (i2p) ... (iN) (iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
#     # [0, 2, 4, ..., 2N - 2, 1, 3, ..., 2N - 1]
#     i1 = jnp.arange(0, 2 * num_sites_in_lattice - 1, 2)
#     i2 = jnp.arange(1, 2 * num_sites_in_lattice, 2)
#     i = jnp.concatenate((i1, i2))
#
#     A = jnp.transpose(T, tuple(i))
#
#     # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
#     A = jnp.flatten(A)
#
#     # (i1 i2 ... iN i1p i2p ... iNp) -> (i1 i2 ... iN)(i1p i2p ... iNp)
#     A = jnp.reshape(A, (dim, dim))
#
#     return A
#
# #------------ Dual variable reshape methods ------------#
#
# def assemble_vars_into_tensors(vars_vec: jnp.array, p: int, dim: int):
#
#     utri_indices = jnp.triu_indices(dim, 1)
#     ltri_indices = (utri_indices[1], utri_indices[0])
#     num_tri_elements = utri_indices[0].shape[0]
#     num_diag_elements = dim
#
#     vars_diag_list, vars_real_list, vars_imag_list, Lambdas \
#             = unvectorize_vars(vars_vec, p, num_diag_elements, num_tri_elements)
#
#     Sigmas = []
#
#     for i in range(p):
#
#         tri_real = jnp.zeros((dim, dim), dtype = complex)
#         tri_imag = jnp.zeros((dim, dim), dtype = complex)
#
#         tri_real[utri_indices] = vars_real_list[i]
#         tri_real[ltri_indices] = vars_real_list[i]
#         tri_imag[utri_indices] = 1j * vars_imag_list[i]
#         tri_imag[ltri_indices] = -1j * vars_imag_list[i]
#
#         vars_full = jnp.diag(vars_diag_list[i]) + tri_real + tri_imag
#
#         Sigmas.append(tn.Node(mat_2_tensor(vars_full)))
#
#     return Sigmas, Lambdas
#
# def unvectorize_vars(vars_vec: jnp.array, p: int,
#                      num_diag_elements: int, num_tri_elements: int):
#
#     Lambdas = vars_vec[:p]
#
#     vars_vec_split = jnp.split(vars_vec[p:], p)
#     # split the variables into p equal arrays
#
#     vars_diag_list = []
#     vars_real_list = []
#     vars_imag_list = []
#
#     for i in range(p):
#
#         # for each circuit step the variables are arranged as:
#         # [vars_diag, vars_real, vars_imag]
#
#         vars_diag = vars_vec_split[i][:num_diag_elements]
#         vars_real = vars_vec_split[i][num_diag_elements:num_diag_elements + num_tri_elements]
#         vars_imag = vars_vec_split[i][num_diag_elements + num_tri_elements:]
#
#         vars_diag_list.append(vars_diag)
#         vars_real_list.append(vars_real)
#         vars_imag_list.append(vars_imag)
#
#     return vars_diag_list, vars_real_list, vars_imag_list, Lambdas
#
# #------------ Main dual function calc. methods ------------#
#
# def objective(vars_vec: jnp.array, ?):
#
#     # Unvectorize the vars_vec and construct Sigmas and Lambdas
#     Sigmas, Lambdas = assemble_vars_into_tensors(vars_vec)
#
#     # Compute the objective function using the list of tensors
#     obj = cost(?)
#
#     return obj
#
# def cost(?):
#
#     # the entropy term
#     cost = jnp.dot(Lambdas, entropy_bounds)
#
#     # the effective Hamiltonian term
#     for i in range(p):
#
#         # i corresponds to the layer of the circuit, 0 being the earliest
#         # i.e. i = 0 corresponds to t = 1 in notes
#
#         # construct the effective Hamiltonian for the ith step/layer
#         Hi = construct_H(i = i, p = p, Sigmas = Sigmas,
#                         rho_init_tensor = rho_init_tensor,
#                         gamma = gamma, beta = beta, p_noise = p_noise,
#                         num_sites_in_lattice = num_sites_in_lattice,
#                         graph_edges = graph_edges,
#                         graph_site_nums = graph_site_nums,
#                         H_problem = H_problem)
#
#         Ei = jnp.linalg.eigvals(Hi)
#
#         cost += -Lambdas[i] * jnp.log2(jnp.sum(jnp.exp(-Ei/Lambdas[i])))
#
#     # the initial state term
#     epsilon_1_rho = noisy_circuit_layer(i = 0, Sigmas = Sigmas,
#                     rho_init_tensor = rho_init_tensor,
#                     gamma = gamma, beta = beta, p_noise = p_noise,
#                     num_sites_in_lattice = num_sites_in_lattice,
#                     graph_edges = graph_edges,
#                     graph_site_nums = graph_site_nums).tensor
#
#     sigma1 = tensor_2_mat(Sigmas[0].tensor)
#     epsilon_1_rho = tensor_2_mat(epsilon_1_rho)
#     cost += -jnp.trace(sigma1 @ epsilon_1_rho)
#
#     return cost
#
# def construct_H(i: int, p: int, Sigmas: List, rho_init_tensor: tn.Node,
#                 gamma: jnp.array, beta: jnp.array, p_noise: float,
#                 num_sites_in_lattice: int,
#                 graph_edges: List, graph_site_nums: List,
#                 H_problem: jnp.array):
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
#     if i == p - 1:
#         # TODO: check that the tensor product/site num ordering is
#         # consistent
#         Hi = tensor_2_mat(Sigmas[i].tensor) + H_problem
#
#         return Hi
#
#     else:
#         Hi = Sigmas[i].tensor - noisy_circuit_layer(i = i + 1, Sigmas = Sigmas,
#                         rho_init_tensor = rho_init_tensor,
#                         gamma = gamma, beta = beta, p_noise = p_noise,
#                         num_sites_in_lattice = num_sites_in_lattice,
#                         graph_edges = graph_edges,
#                         graph_site_nums = graph_site_nums).tensor
#
#         return self.tensor_2_mat(Hi)
#
# def noisy_circuit_layer(i: int, Sigmas: List, rho_init_tensor: tn.Node,
#                         gamma: jnp.array, beta: jnp.array, p_noise: float,
#                         num_sites_in_lattice: int,
#                         graph_edges: List, graph_site_nums: List):
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
#     if i == 0:
#         res_tensor = circuit_layer(var_tensor = rho_init_tensor,
#                           gamma = gamma[i], beta = beta[i],
#                           num_sites_in_lattice = num_sites_in_lattice,
#                           graph_edges = graph_edges,
#                           graph_site_nums = graph_site_nums)
#     else:
#         res_tensor = circuit_layer(var_tensor = Sigmas[i],
#                           gamma = gamma[i], beta = beta[i],
#                           num_sites_in_lattice = num_sites_in_lattice,
#                           graph_edges = graph_edges,
#                           graph_site_nums = graph_site_nums)
#
#     res_tensor = noise_layer(var_tensor = res_tensor,
#                 num_sites_in_lattice = num_sites_in_lattice, p_noise = p_noise)
#
#     return res_tensor
#
# def noise_layer(var_tensor: tn.Node, num_sites_in_lattice: int, p_noise: float):
#
#     """
#     Applies depolarizing noise on the var_tensor at all sites.
#     """
#
#     res_tensor = var_tensor
#
#     X = jnp.array([[0, 1],[1, 0]])
#     Y = jnp.array([[0, -1j],[1j, 0]])
#     Z = jnp.array([[1, 0],[0, -1]])
#
#     for site_num in range(num_sites_in_lattice):
#
#         # --- applying I --- #
#         res_array = (1 - 3 * p_noise/4) * res_tensor.tensor
#
#         # --- applying X --- #
#         tmp_tensor = res_tensor
#
#         X_node = tn.Node(np.sqrt(p_noise/4) * X, axis_names = ["ja","ia"])
#         X_prime_node = tn.Node(np.sqrt(p_noise/4) * X, axis_names = ["iap","kap"])
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
#         Y_node = tn.Node(np.sqrt(p_noise/4) * Y, axis_names = ["ja","ia"])
#         Y_prime_node = tn.Node(np.sqrt(p_noise/4) * Y, axis_names = ["iap","kap"])
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
#         Z_node = tn.Node(np.sqrt(p_noise/4) * Z, axis_names = ["ja","ia"])
#         Z_prime_node = tn.Node(np.sqrt(p_noise/4) * Z, axis_names = ["iap","kap"])
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
# def circuit_layer(var_tensor: tn.Node,
#                   gamma: float, beta: float,
#                   num_sites_in_lattice: int,
#                   graph_edges: List, graph_site_nums: List):
#
#     res_tensor = var_tensor
#
#     #----- Applying the problem unitary -----#
#     # U = exp(-i gamma/2 w * (I - Z_j Z_k))
#     U = jnp.diag([1, jnp.exp(1j * gamma), jnp.exp(1j * gamma), 1])
#     U_dag = jnp.conj(jnp.transpose(U))
#
#     # (ja jb) (ia ib) -> (ja jb ia ib)
#     U_tensor = jnp.flatten(U)
#     # (ja jb ia ib) -> (ja) (jb) (ia) (ib)
#     U_tensor = jnp.reshape(U_tensor, (2,2,2,2))
#     # (ja) (jb) (ia) (ib) -> (ja) (ia) (jb) (ib)
#     U_tensor = jnp.transpose(U_tensor, [0, 2, 1, 3])
#
#     # (iap ibp) (kap kbp) -> (iap ibp kap kbp)
#     U_dag_tensor = jnp.flatten(U_dag)
#     # (iap ibp kap kbp) -> (iap) (ibp) (kap) (kbp)
#     U_dag_tensor = jnp.reshape(U_dag_tensor, (2,2,2,2))
#     # (iap) (ibp) (kap) (kbp) -> (iap) (kap) (ibp) (kbp)
#     U_dag_tensor = jnp.transpose(U_dag_tensor, [0, 2, 1, 3])
#
#     for edge in graph_edges:
#
#         site_num_a = jnp.min(graph_site_nums[edge[0]], graph_site_nums[edge[1]])
#         site_num_b = jnp.max(graph_site_nums[edge[0]], graph_site_nums[edge[1]])
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
#         edge_a = U_node["ia"] ^ res_tensor[ia]
#         edge_b = U_node["ib"] ^ res_tensor[ib]
#
#         new_edge_order = res_tensor.edges[:ia] + [U_node["ja"]] +\
#                          res_tensor.edges[ia + 1: ib] + [U_node["jb"]] +\
#                          res_tensor.edges[ib + 1:]
#
#         res_tensor = tn.contract_between(U_node, res_tensor, output_edge_order = new_edge_order)
#
#         edge_a_p = U_dag_node["iap"] ^ res_tensor[iap]
#         edge_b_p = U_dag_node["ibp"] ^ res_tensor[ibp]
#
#         new_edge_order = res_tensor.edges[:iap] + [U_dag_node["kap"]] +\
#                          res_tensor.edges[iap + 1: ibp] + [U_dag_node["kbp"]] +\
#                          res_tensor.edges[ibp + 1:]
#
#         res_tensor = tn.contract_between(U_dag_node, res_tensor, output_edge_order = new_edge_order)
#
#     #----- Applying the mixing unitary -----#
#     X = jnp.array([[0,1], [1,0]])
#     I = jnp.array([[1,0], [0,1]])
#
#     Ux = jnp.cos(beta) * I - 1j * jnp.sin(beta) * X
#     Ux_dag = jnp.conj(jnp.transpose(Ux))
#
#     for site_num in range(num_sites_in_lattice):
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
#
#         new_edge_order = res_tensor.edges[:ia] + [Ux_node["ja"]] +\
#                          res_tensor.edges[ia + 1:]
#         res_tensor = tn.contract_between(Ux_node, res_tensor, output_edge_order = new_edge_order)
#
#         new_edge_order = res_tensor.edges[:iap] + [Ux_dag_node["kap"]] +\
#                          res_tensor.edges[iap + 1:]
#         res_tensor = tn.contract_between(Ux_dag_node, res_tensor, output_edge_order = new_edge_order)
#
#     return res_tensor
#
# #------------ Other helper methods ------------#
#
# def gen_entropy_bounds(p_noise: float, num_sites_in_lattice: int):
#
#     q = 1 - p_noise
#     q_powers = [q**i for i in range(p)]
#
#     entropy_bounds = num_sites_in_lattice * p_noise * \
#                      np.array([np.sum(q_powers[:i+1]) for i in range(p)])
#
#     return entropy_bounds
