from typing import List, Tuple, Callable, Dict

import tensornetwork as tn
import numpy as np
import scipy
import qutip

from vqa import graphs
from vqa import problems
from vqa import algorithms

class MaxCutDual():

    """
    A class to calculate the dual function and its gradient for the MaxCut QAOA
    problem.

    Members:

        self.Sigmas: list of tensors each with 2N legs of dimension d (d = 2
                by default). If each sigma is self-adjoint, the dual objective
                function is real.

        self.Lambdas: list of positive real numbers.

    Params:
        prob_obj = Problem object (NB, none of the unitary methods
        from this object are utilised here; the unitary/noise operations are
        implemented from scratch for this class.)

        p = number of layers

        gamma = params of problem layer

        beta = params of mixing layer

        p_noise: depolarizing noise probability on each qubit

    Currently not implemented:
        1. Construction of the effective Hamiltonians (by applying gates)
            a. Add circuit params and local gates
            b. Add noise channels
        2. Initial state term in the cost function
        3. Entropy bounds
        4. Finite-difference gradient
        5. JAX gradient
    """

    def __init__(self, prob_obj: problems.Problem,
                 p: int, gamma: np.array, beta: np.array, p_noise: float):

        self.prob_obj = prob_obj
        self.num_sites_in_lattice = self.prob_obj.num_sites_in_lattice
        self.local_dim = 2
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.p_noise = p_noise
        self.psi_init = prob_obj.init_state()

        self.dim_list = tuple([self.local_dim] * 2 * self.num_sites_in_lattice)
        self.dim = self.local_dim ** self.num_sites_in_lattice

        self.utri_indices = np.triu_indices(self.dim, 1)
        self.ltri_indices = np.tril_indices(self.dim, -1)
        self.num_tri_elements = self.utri_indices[0].shape[0]
        self.num_diag_elements = self.dim

        # constructing initial density operator
        self.rho_init = (self.psi_init * self.psi_init.dag()).full()
        self.rho_init_tensor = tn.Node(self.mat_2_tensor(self.rho_init), name = "rho_init")

        # initialising the noise channel superoperator
        # NB - the basis is |0X0|,|0X1|, |1X0|, |1X1|

        self.noise_superop = np.array([[1 - self.p_noise/2, self.p_noise/2, self.p_noise/2, self.p_noise/2],
                                       [0                 , 1 - self.p_noise, 0           , 0             ],
                                       [0                 , 0               , 1 - self.p_noise, 0         ],
                                       [self.p_noise/2, self.p_noise/2, self.p_noise/2, 1 - self.p_noise/2]])

        self.init_entropy_bounds()

        # self.tot_num_vars_per_step = 1 + self.num_diag_elements + 2 * self.num_tri_elements
        # lambda + number of elements in sigma

    def mat_2_tensor(self, A: np.array):

        """
        Converts an array A that is indexed as (i1 i2 ... iN)(i1p i2p ... iNp)
        into a tensor indexed as (i1) (i1p) (i2) (i2p) ... (iN) (iNp). () denote
        composite indices.
        """

        # (i1 i2 ... iN)(i1p i2p ... iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
        T = A.flatten()

        # (i1 i2 ... iN i1p i2p ... iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
        T = T.reshape(tuple([self.local_dim] * self.num_sites_in_lattice * 2))

        # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1) (i1p) ... (iN) (iNp)
        # [0, N, 1, N + 1, ..., N - 1, 2N - 1]
        i1 = np.arange(self.num_sites_in_lattice)
        i2 = self.num_sites_in_lattice + i1
        i  = np.zeros(2 * self.num_sites_in_lattice, dtype = int)
        i[::2] = i1
        i[1::2] = i2

        T = T.transpose(tuple(i))

        return T

    def tensor_2_mat(self, T: np.array):

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

        A = T.transpose(tuple(i))

        # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
        A = A.flatten()

        # (i1 i2 ... iN i1p i2p ... iNp) -> (i1 i2 ... iN)(i1p i2p ... iNp)
        A = A.reshape((self.dim, self.dim))

        return A

    def init_entropy_bounds(self):

        q = 1 - self.p_noise
        q_powers = [q**i for i in range(self.p)]

        self.entropy_bounds = self.num_sites_in_lattice * self.p_noise * \
                              [np.sum(q_powers[:i+1]) for i in range(self.p)]

    def objective(self, vars_vec: np.array):

        # Unvectorize the vars_vec and construct Sigmas and Lambdas
        self.assemble_vars_into_tensors(vars_vec)

        # Compute the objective function using the list of tensors
        obj = self.cost()

        return obj

    def assemble_vars_into_tensors(self, vars_vec):

        vars_diag_list, vars_real_list, vars_imag_list = self.unvectorize_vars(vars_vec)
        self.Sigmas = []

        for i in range(self.p):

            tri_real = np.zeros((self.dim, self.dim), dtype = complex)
            tri_imag = np.zeros((self.dim, self.dim), dtype = complex)

            tri_real[self.utri_indices] = vars_real_list[i]
            tri_real[self.ltri_indices] = vars_real_list[i]
            tri_imag[self.utri_indices] = 1j * vars_imag_list[i]
            tri_imag[self.ltri_indices] = -1j * vars_imag_list[i]

            vars_full = np.diag(vars_diag_list[i]) + tri_real + tri_imag

            self.Sigmas.append(tn.Node(self.mat_2_tensor(vars_full)))

    def unvectorize_vars(self, vars_vec):

        self.Lambdas = vars_vec[:self.p]

        vars_vec_split = np.split(vars_vec[self.p:], self.p)
        # split the variables into p equal arrays

        vars_diag_list = []
        vars_real_list = []
        vars_imag_list = []

        for i in range(self.p):

            # for each circuit step the variables are arranged as:
            # [vars_diag, vars_real, vars_imag]

            vars_diag = vars_vec_split[i][:self.num_diag_elements]
            vars_real = vars_vec_split[i][self.num_diag_elements:self.num_diag_elements + self.num_tri_elements]
            vars_imag = vars_vec_split[i][self.num_diag_elements + self.num_tri_elements:-1]

            vars_diag_list.append(vars_diag)
            vars_real_list.append(vars_real)
            vars_imag_list.append(vars_imag)

        return vars_diag_list, vars_real_list, vars_imag_list

    def cost(self):

        # the entropy term
        cost = np.dot(self.Lambdas, self.entropy_bounds)

        # the effective Hamiltonian term
        for i in range(self.p):

            # i corresponds to the layer of the circuit, 0 being the earliest
            # i.e. i = 0 corresponds to t = 1 in notes

            # construct the effective Hamiltonian for the ith step/layer
            Hi = self.construct_H(i)
            Ei = np.linalg.eigvals(Hi)

            cost += -self.Lambdas[i] * np.log(np.sum(np.exp(-Ei/self.Lambdas[i])), base = 2)

        # the initial state term
        # TODO: will this be faster if expressed in terms of contractions?
        sigma1 = self.tensor_2_mat(self.Sigmas[0].tensor)
        epsilon_1_rho = self.tensor_2_mat(self.noisy_circuit_layer(i = 0).tensor)
        cost += -np.trace(sigma1 @ epsilon_1_rho)

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

        if i == self.p - 1:
            # TODO: check that the tensor product/site num ordering is
            # consistent
            Hi = self.tensor_2_mat(self.Sigmas[i].tensor) + self.prob_obj.H.full()

            return Hi

        else:
            Hi = self.Sigmas[i].tensor - self.noisy_circuit_layer(i + 1).tensor

            return self.tensor_2_mat(Hi)

    def noise_layer(self, var_tensor: tn.Node):

        """
        Applies depolarizing noise on the var_tensor at all sites.
        """

        # (i1) (i1p) (i2) (i2p) ... (iN) (iNp) -> (i1 i1p) (i2 i2p) ... (iN iNp)
        res = np.reshape(var_tensor.tensor, tuple([self.local_dim ** 2] * self.num_sites_in_lattice))
        res_tensor = tn.Node(res)

        for site_num in range(self.num_sites_in_lattice):

            noise_node = tn.Node(self.noise_superop,
                                 axis_names = ["up", "down"], name = "noise_node") # shape = 4x4

            # TODO: check
            edge_connection = noise_node["down"] ^ res_tensor[site_num]

            # perform the contraction
            new_edge_order = res_tensor.edges[:site_num] + [noise_node["up"]] +\
                             res_tensor.edges[site_num + 1:]
            res_tensor = tn.contract_between(noise_node, res_tensor, output_edge_order = new_edge_order)

        final_res = np.reshape(res_tensor.tensor, self.dim_list)
        res_tensor = tn.Node(final_res)

        return res_tensor

    def circuit_layer(self, layer_num: int, var_tensor: tn.Node):

        res_tensor = var_tensor

        #----- Applying the problem unitary -----#
        gamma = self.gamma[layer_num]

        # U = exp(-i gamma/2 w * (I - Z_j Z_k))
        U = np.diag([1, np.exp(1j * gamma), np.exp(1j * gamma), 1])
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

        for edge in self.prob_obj.graph.edges:

            site_num_a = min(self.prob_obj.site_nums[edge[0]], self.prob_obj.site_nums[edge[1]])
            site_num_b = max(self.prob_obj.site_nums[edge[0]], self.prob_obj.site_nums[edge[1]])

            ia = 2 * site_num_a
            ib = 2 * site_num_b

            iap = 2 * site_num_a + 1
            ibp = 2 * site_num_b + 1

            print("On edge: ", edge)
            print("ia: ", ia)
            print("ib: ", ib)
            print("iap: ", iap)
            print("ibp: ", ibp)

            # TODO: does this work? check with a trial unitary/state.
            U_node      = tn.Node(U_tensor, axis_names = ["ja", "ia", "jb", "ib"], name = "U_node")
            U_dag_node  = tn.Node(U_dag_tensor, axis_names = ["iap", "kap", "ibp", "kbp"], name = "U_dag_node")

            # print(self.tensor_2_mat(U_dag_node.tensor))
            # print(U_dag_node.tensor)

            # TODO: does this work? name edges for clarity?
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
        beta = self.beta[layer_num]

        sx = qutip.sigmax()
        Ux = (-1j * beta * sx).expm().full()
        Ux_dag = Ux.conj().T

        for site_num in range(self.num_sites_in_lattice):

            Ux_node = tn.Node(Ux, axis_names = ["ja","ia"])
            Ux_dag_node = tn.Node(Ux_dag, axis_names = ["iap","kap"])

            ia  = 2 * site_num
            iap = 2 * site_num + 1

            # TODO: check and name edges to improve readability
            edge_a      = Ux_node["ia"] ^ res_tensor[ia]
            edge_a_p    = Ux_dag_node["iap"] ^ res_tensor[iap]

            # TODO: check if re-assignment works
            # perform the contraction

            new_edge_order = res_tensor.edges[:ia] + [Ux_node["ja"]] +\
                             res_tensor.edges[ia + 1:]
            res_tensor = tn.contract_between(Ux_node, res_tensor, output_edge_order = new_edge_order)

            new_edge_order = res_tensor.edges[:iap] + [Ux_dag_node["kap"]] +\
                             res_tensor.edges[iap + 1:]
            res_tensor = tn.contract_between(Ux_dag_node, res_tensor, output_edge_order = new_edge_order)

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

        if i == 0:
            res_tensor = circuit_layer(layer_num = i, var_tensor = self.rho_init_tensor)
        else:
            res_tensor = circuit_layer(layer_num = i, var_tensor = self.Sigmas[i])

        res_tensor = noise_layer(res_tensor)

        return res_tensor
