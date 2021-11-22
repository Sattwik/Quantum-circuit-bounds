from typing import List, Tuple, Callable, Dict

import tensornetwork as tn
import numpy as np
import scipy

class Dual():

    """
    A class to calculate the dual function and its gradient for the MaxCut QAOA
    problem.

    Members:

        self.Sigmas: list of tensors each with N legs of dimension d^2 (d = 2
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

    def __init__(self, prob_obj: Problem,
                 p: int, gamma: np.array, beta: np.array):

        self.prob_obj = prob_obj
        self.num_sites_in_lattice = self.prob_obj.num_sites_in_lattice
        self.local_dim = 2
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.p_noise = p_noise
        self.psi_init = problem_obj.init_state()

        self.dim_list = tuple([self.local_dim ** 2] * self.num_sites_in_lattice)
        self.dim = self.local_dim ** self.num_sites_in_lattice

        self.utri_indices = np.triu_indices(self.dim, 1)
        self.ltri_indices = np.tril_indices(self.dim, -1)
        self.num_tri_elements = self.utri_indices[0].shape[0]
        self.num_diag_elements = self.dim

        # constructing initial density operator
        self.rho_init = self.psi_init * self.psi_init.dag()
        self.rho_init_tensor = tn.Node(np.reshape(self.rho_init.full(), self.dim_list))

        # initialising the noise channel superoperator
        # NB - the basis is |0X0|,|0X1|, |1X0|, |1X1|

        self.noise_superop = np.array([[1 - self.p_noise/2, self.p_noise/2, self.p_noise/2, self.p_noise/2],
                                       [0                 , 1 - self.p_noise, 0           , 0             ],
                                       [0                 , 0               , 1 - self.p_noise, 0         ],
                                       [self.p_noise/2, self.p_noise/2, self.p_noise/2, 1 - self.p_noise/2]])

        self.init_entropy_bounds()

        # self.tot_num_vars_per_step = 1 + self.num_diag_elements + 2 * self.num_tri_elements
        # lambda + number of elements in sigma

    def init_entropy_bounds(self):

        raise NotImplementedError()

    def objective(self, vars_vec: np.array):

        """
        Program flow:

        objective(vars_vec) -> assemble_vars_into_tensors() -> unvectorize_vars()
        -> assemble_vars_into_tensors() -> cost() ->
        (construct_H() -> noisy_circuit_layer() -> cost() -> ) * p ->
        """

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

            self.Sigmas.append(tn.Node(np.reshape(vars_full, self.dim_list)))

    def unvectorize_vars(self, vars_vec):

        vars_vec_split = np.split(vars_vec, self.p)
        # split the variables into p equal arrays

        vars_diag_list = []
        vars_real_list = []
        vars_imag_list = []
        self.Lambdas = []

        for i in range(self.p):

            # for each circuit step the variables are arranged as:
            # [vars_diag, vars_real, vars_imag, lambda]

            vars_diag = vars_vec_split[i][:self.num_diag_elements]
            vars_real = vars_vec_split[i][self.num_diag_elements:self.num_diag_elements + self.num_tri_elements]
            vars_imag = vars_vec_split[i][self.num_diag_elements + self.num_tri_elements:-1]
            lmbda     = vars_vec_split[i][-1]

            vars_diag_list.append(vars_diag)
            vars_real_list.append(vars_real)
            vars_imag_list.append(vars_imag)
            self.Lambdas.append(lmbda)

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
            expHi = scipy.linalg.expm(-Hi/self.Lambdas[i])

            cost += -self.Lambdas[i] * np.log(np.trace(expHi))

        # the initial state term # TODO: check sign
        sigma1 = np.reshape(self.Sigmas[0].tensor, newshape = (self.dim, self.dim))
        epsilon_1_rho = np.reshape(self.noisy_circuit_layer(i = 0).tensor, newshape = (self.dim, self.dim))
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

        Hi = self.Sigmas[i].tensor - self.noisy_circuit_layer(i + 1)).tensor

        return np.reshape(Hi, newshape = (self.dim, self.dim))

    def noise_layer(self, var_tensor: tn.Node):

        """
        Applies the noise channel on the var_tensor at all sites.
        """

        res_tensor = var_tensor

        for site_num in range(self.num_sites_in_lattice):

            noise_node = tn.Node(self.noise_superop) # shape = 4x4

            # TODO: check and name edges to improve readability
            edge_connection = noise_node[1] ^ res_tensor[site_num]

            # TODO: check if re-assignment works
            # perform the contraction
            res_tensor = noise_node @ res_tensor

        return res_tensor

    def circuit_layer(self, layer_num: int, var_tensor: tn.Node):

        res_tensor = var_tensor

        # applying the problem Hamiltonian
        gamma = self.gamma[layer_num]

        # U = exp(-i gamma/2 w * (I - Z_j Z_k))
        U = np.diag([1, np.exp(1j * gamma), np.exp(1j * gamma), 1])

        U_superop = np.reshape(np.kron(U, np.conj(U)), newshape = (4, 4, 4, 4))

        for edge in self.prob_obj.graph.edges:

            site_num_j = self.prob_obj.site_nums[edge[0]]
            site_num_k = self.prob_obj.site_nums[edge[1]]

            # TODO: does this work? check with a trial unitary/state.
            U_node = tn.Node(U_superop)

            # TODO: does this work? name edges for clarity
            new_edge_j = U_node[2] ^ res_tensor[site_num_j]
            new_edge_k = U_node[3] ^ res_tensor[site_num_k]

            # TODO: check if re-assignment works
            res_tensor = U_node @ res_tensor

        # applying the mixing Hamiltonian
        beta = self.beta[layer_num]

        sx = qutip.sigmax()
        Ux = (-1j * beta * sx).expm().full()

        Ux_superop = np.kron(Ux, np.conj(Ux))

        for site_num in range(self.num_sites_in_lattice):

            Ux_node = tn.Node(self.Ux_superop) # shape = 4x4

            # TODO: check and name edges to improve readability
            edge_connection = Ux_node[1] ^ res_tensor[site_num]

            # TODO: check if re-assignment works
            # perform the contraction
            res_tensor = Ux_node @ res_tensor

        return res_tensor

    def noisy_circuit_layer(self, i: int):

        """
        Applies the unitary corresponding to a circuit layer to the dual
        variable Sigma. Also applies the noise layer after. The noise model is
        depolarizing noise on each qubit.

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
