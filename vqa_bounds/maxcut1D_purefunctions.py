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

class MaxCut1D():

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

        # for unrestricted dual
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

        self.total_num_vars_full = (1 + self.num_diag_elements + 2 * self.num_tri_elements) * self.d

        self.I_jax = jnp.array(self.I_qutip.full())
        self.X_jax = jnp.array(self.X_qutip.full())
        self.Y_jax = jnp.array(self.Y_qutip.full())
        self.Z_jax = jnp.array(self.Z_qutip.full())

        # for local dual
        self.local_var_dim = self.local_dim ** 2

        self.num_vars_local = (self.local_dim ** 2) ** 2
        # because local H are over two sites
        self.num_vars_odd_layers = (self.d//2) * self.num_vars_local * ((self.num_sites_in_lattice - 2)//2)
        self.num_vars_even_layers = (self.d//2) * self.num_vars_local * self.num_sites_in_lattice//2

        self.total_num_vars_local = self.d + self.num_vars_odd_layers + self.num_vars_even_layers

        self.num_diag_elements_local = self.local_dim ** 2
        self.utri_indices_local = jnp.triu_indices(self.local_dim ** 2, 1)
        self.ltri_indices_local = (self.utri_indices_local[1], self.utri_indices_local[0])
        self.num_tri_elements_local = self.utri_indices_local[0].shape[0]

        self.X_left = jnp.array(qutip.tensor(self.X_qutip, self.I_qutip).full())
        self.Y_left = jnp.array(qutip.tensor(self.Y_qutip, self.I_qutip).full())
        self.Z_left = jnp.array(qutip.tensor(self.Z_qutip, self.I_qutip).full())

        self.X_right = jnp.array(qutip.tensor(self.I_qutip, self.X_qutip).full())
        self.Y_right = jnp.array(qutip.tensor(self.I_qutip, self.Y_qutip).full())
        self.Z_right = jnp.array(qutip.tensor(self.I_qutip, self.Z_qutip).full())

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

#--------------------- Unrestricted ansatz dual ---------------------#

@partial(jit, static_argnums = (1,))
def mat_2_tensor(A: jnp.array, sys_obj: MaxCut1D):

    """
    Converts an array A that is indexed as (i1 i2 ... iN)(i1p i2p ... iNp)
    into a tensor indexed as (i1) (i1p) (i2) (i2p) ... (iN) (iNp). () denote
    composite indices.
    """

    # (i1 i2 ... iN)(i1p i2p ... iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
    T = jnp.ravel(A)

    # (i1 i2 ... iN i1p i2p ... iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
    T = jnp.reshape(T, tuple([sys_obj.local_dim] * sys_obj.num_sites_in_lattice * 2))

    # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1) (i1p) ... (iN) (iNp)
    # [0, N, 1, N + 1, ..., N - 1, 2N - 1]
    i1 = np.arange(sys_obj.num_sites_in_lattice)
    i2 = sys_obj.num_sites_in_lattice + i1
    i  = np.zeros(2 * sys_obj.num_sites_in_lattice, dtype = int)
    i[::2] = i1
    i[1::2] = i2
    # i = i.at[::2].set(i1)
    # i = i.at[1::2].set(i2)

    T = jnp.transpose(T, tuple(i))

    return T

@partial(jit, static_argnums = (1,))
def tensor_2_mat(T: jnp.array, sys_obj: MaxCut1D):

    """
    Converts a tensor T that is indexed as  (i1) (i1p) (i2) (i2p) ... (iN) (iNp)
    into a matrix indexed as (i1 i2 ... iN)(i1p i2p ... iNp). () denote
    composite indices.
    """

    # (i1) (i1p) (i2) (i2p) ... (iN) (iNp) -> (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp)
    # [0, 2, 4, ..., 2N - 2, 1, 3, ..., 2N - 1]
    i1 = np.arange(0, 2 * sys_obj.num_sites_in_lattice - 1, 2)
    i2 = np.arange(1, 2 * sys_obj.num_sites_in_lattice, 2)
    i = np.concatenate((i1, i2))

    A = jnp.transpose(T, tuple(i))

    # (i1) (i2) ... (iN) (i1p) (i2p) ... (iNp) -> (i1 i2 ... iN i1p i2p ... iNp)
    A = jnp.ravel(A)

    # (i1 i2 ... iN i1p i2p ... iNp) -> (i1 i2 ... iN)(i1p i2p ... iNp)
    A = jnp.reshape(A, (sys_obj.dim, sys_obj.dim))

    return A

@partial(jit, static_argnums = (1,))
def dual_obj_full(dual_vars: jnp.array, sys_obj: MaxCut1D):

    vars_vec = dual_vars

    # Unvectorize the vars_vec and construct Sigmas and Lambdas
    a_vars = vars_vec.at[:sys_obj.d].get()
    Lambdas = jnp.log(1 + jnp.exp(a_vars))

    vars_vec_split = jnp.split(vars_vec.at[sys_obj.d:].get(), sys_obj.d)
    # split the variables into p equal arrays

    vars_diag_list = []
    vars_real_list = []
    vars_imag_list = []

    for i in range(sys_obj.d):

        # for each circuit step the variables are arranged as:
        # [vars_diag, vars_real, vars_imag]

        vars_diag = vars_vec_split[i].at[:sys_obj.num_diag_elements].get()
        vars_real = vars_vec_split[i].at[sys_obj.num_diag_elements:sys_obj.num_diag_elements + sys_obj.num_tri_elements].get()
        vars_imag = vars_vec_split[i].at[sys_obj.num_diag_elements + sys_obj.num_tri_elements:].get()

        vars_diag_list.append(vars_diag)
        vars_real_list.append(vars_real)
        vars_imag_list.append(vars_imag)

    Sigmas = []

    for i in range(sys_obj.d):

        tri_real = jnp.zeros((sys_obj.dim, sys_obj.dim), dtype = complex)
        tri_imag = jnp.zeros((sys_obj.dim, sys_obj.dim), dtype = complex)

        tri_real = tri_real.at[sys_obj.utri_indices].set(vars_real_list[i])
        tri_real = tri_real.at[sys_obj.ltri_indices].set(vars_real_list[i])
        tri_imag = tri_imag.at[sys_obj.utri_indices].set(1j * vars_imag_list[i])
        tri_imag = tri_imag.at[sys_obj.ltri_indices].set(-1j * vars_imag_list[i])

        vars_full = jnp.diag(vars_diag_list[i]) + tri_real + tri_imag

        Sigmas.append(tn.Node(mat_2_tensor(vars_full, sys_obj)))

    # Compute the objective function using the list of tensors
    # the entropy term
    cost = jnp.dot(Lambdas, sys_obj.entropy_bounds)

    # the effective Hamiltonian term
    for i in range(sys_obj.d):

        # i corresponds to the layer of the circuit, 0 being the earliest
        # i.e. i = 0 corresponds to t = 1 in notes

        # construct the effective Hamiltonian for the ith step/layer
        if i == sys_obj.d - 1:
            Hi = tensor_2_mat(Sigmas[i].tensor, sys_obj) + sys_obj.H_problem

        else:
            var_tensor = Sigmas[i + 1].tensor
            Hi = tensor_2_mat(Sigmas[i].tensor - \
                     noisy_dual_layer_full(var_tensor, i + 1, sys_obj), sys_obj)

        Ei = jnp.linalg.eigvalsh(Hi)

        cost += -Lambdas[i] * jnp.log(jnp.sum(jnp.exp(-Ei/Lambdas[i])))

    # the initial state term
    epsilon1_dag_sigma1 = tensor_2_mat(noisy_dual_layer_full(Sigmas[0].tensor, 0, sys_obj), sys_obj)

    cost += -jnp.trace(jnp.matmul(sys_obj.rho_init, epsilon1_dag_sigma1))

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad_full(dual_vars: jnp.array, sys_obj: MaxCut1D):

    return grad(dual_obj_full, argnums = 0)(dual_vars, sys_obj)

@partial(jit, static_argnums = (1,2))
def noisy_dual_layer_full(var_tensor: jnp.array, i: int, sys_obj: MaxCut1D):

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

    res_tensor = var_tensor
    res_tensor = noise_layer_full(res_tensor, sys_obj)
    res_tensor = dual_unitary_layer_full(var_tensor = res_tensor,
                                    layer_num = i + 1, sys_obj = sys_obj)
                                    # i + 1 here because unitary_layer
                                    # counts layers differently
    return res_tensor

@partial(jit, static_argnums = (1,))
def noise_layer_full(var_tensor: jnp.array, sys_obj: MaxCut1D):

    """
    Applies depolarizing noise on the var_tensor at all sites.
    """

    res_tensor = tn.Node(var_tensor)

    for site_num in range(sys_obj.num_sites_in_lattice):

        # --- applying I --- #
        res_array = (1 - 3 * sys_obj.p/4) * res_tensor.tensor

        # --- applying X --- #
        tmp_tensor = res_tensor

        X_node = tn.Node(np.sqrt(sys_obj.p/4) * sys_obj.X_jax, axis_names = ["ja","ia"])
        X_prime_node = tn.Node(np.sqrt(sys_obj.p/4) * sys_obj.X_jax, axis_names = ["iap","kap"])

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

        Y_node = tn.Node(np.sqrt(sys_obj.p/4) * sys_obj.Y_jax, axis_names = ["ja","ia"])
        Y_prime_node = tn.Node(np.sqrt(sys_obj.p/4) * sys_obj.Y_jax, axis_names = ["iap","kap"])

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

        Z_node = tn.Node(np.sqrt(sys_obj.p/4) * sys_obj.Z_jax, axis_names = ["ja","ia"])
        Z_prime_node = tn.Node(np.sqrt(sys_obj.p/4) * sys_obj.Z_jax, axis_names = ["iap","kap"])

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

    return res_tensor.tensor

@partial(jit, static_argnums = (1,2))
def dual_unitary_layer_full(var_tensor: jnp.array, layer_num: int, sys_obj: MaxCut1D):

    res_tensor = var_tensor

    if layer_num % 2 == 0:
        res_tensor = mixing_unitaries_full(res_tensor, layer_num, sys_obj)

    res_tensor = problem_unitaries_full(res_tensor, layer_num, sys_obj)

    return res_tensor

@partial(jit, static_argnums = (1,2))
def problem_unitaries_full(var_tensor: jnp.array, layer_num: int, sys_obj: MaxCut1D):

    #----- Applying the problem unitary -----#
    gamma = sys_obj.opt_gamma[layer_num - 1]

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

    res_tensor = tn.Node(var_tensor)

    if layer_num%2 == 0:
        site_tuple_list = sys_obj.site_tuple_list_even_layer
    else:
        site_tuple_list = sys_obj.site_tuple_list_odd_layer

    for site_tuple in site_tuple_list:

        if site_tuple in sys_obj.graph.edges:

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

    return res_tensor.tensor

@partial(jit, static_argnums = (1,2))
def mixing_unitaries_full(var_tensor: jnp.array, layer_num: int, sys_obj: MaxCut1D):

    assert layer_num % 2 == 0

    #----- Applying the mixing unitary -----#
    beta = sys_obj.opt_beta[(layer_num - 2)//2]

    Ux = jnp.cos(beta) * sys_obj.I_jax - 1j * jnp.sin(beta) * sys_obj.X_jax
    Ux_dag = jnp.transpose(jnp.conj(Ux))

    # adjoint because dual
    Ux = jnp.conj(jnp.transpose(Ux))
    Ux_dag = jnp.conj(jnp.transpose(Ux_dag))

    res_tensor = tn.Node(var_tensor)

    for site_num in range(sys_obj.num_sites_in_lattice):

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

    return res_tensor.tensor

#--------------------- Local H. ansatz dual ---------------------#

# def dual_obj_local(dual_vars: jnp.array, sys_obj: MaxCut1D):
#
#     # Unvectorize the vars_vec and construct Sigmas and Lambdas
#     a_vars = vars_vec.at[:self.d].get()
#     Lambdas = jnp.log(1 + jnp.exp(a_vars))
#
#     vars_vec_odd_layers = vars_vec.at[self.d: self.d + self.num_vars_odd_layers].get()
#     vars_vec_even_layers = vars_vec.at[self.d + self.num_vars_odd_layers:].get()
#
#     # split the variables into equal arrays corresponding to each layer
#     vars_vec_odd_layers_split = jnp.split(vars_vec_odd_layers, self.d//2)
#     vars_vec_even_layers_split = jnp.split(vars_vec_even_layers, self.d//2)
#
#     # list where each element corresponds to the dual variable for a layer
#     # each element is a dictionary with site tuples as keys and the local
#     # Hamiltonians as values
#     Sigmas = []
#
#     for i in range(1, self.d + 1):
#
#         Sigma_layer = {}
#
#         # for each circuit step the variables are arranged as:
#         # [vars_diag, vars_real, vars_imag]
#
#         if i % 2 == 0:
#             vars_layer = vars_vec_even_layers_split[(i-2)//2]
#             vars_layer_split = jnp.split(vars_layer, len(self.site_tuple_list_even_layer))
#             site_index_list = self.site_tuple_list_even_layer
#
#         else:
#             vars_layer = vars_vec_odd_layers_split[(i-1)//2]
#             vars_layer_split = jnp.split(vars_layer, len(self.site_tuple_list_odd_layer))
#             site_index_list = self.site_tuple_list_odd_layer
#
#         for n, site_tuple in enumerate(site_index_list):
#
#             vars_diag = vars_layer_split[n].at[:self.num_diag_elements_local].get()
#             vars_real = vars_layer_split[n].at[self.num_diag_elements_local:self.num_diag_elements_local + self.num_tri_elements_local].get()
#             vars_imag = vars_layer_split[n].at[self.num_diag_elements_local + self.num_tri_elements_local:].get()
#
#             tri_real = jnp.zeros((self.local_var_dim, self.local_var_dim), dtype = complex)
#             tri_imag = jnp.zeros((self.local_var_dim, self.local_var_dim), dtype = complex)
#
#             tri_real = tri_real.at[self.utri_indices_local].set(vars_real)
#             tri_real = tri_real.at[self.ltri_indices_local].set(vars_real)
#             tri_imag = tri_imag.at[self.utri_indices_local].set(1j * vars_imag)
#             tri_imag = tri_imag.at[self.ltri_indices_local].set(-1j * vars_imag)
#
#             vars_full = jnp.diag(vars_diag) + tri_real + tri_imag
#
#             Sigma_layer[site_tuple] = vars_full
#
#         Sigmas.append(Sigma_layer)
#
#     # Compute the objective function using the list of tensors
#     # the entropy term
#     cost = jnp.dot(Lambdas, self.entropy_bounds)
#
#     # the effective Hamiltonian term
#     for i in range(self.d):
#
#         # i corresponds to the layer of the circuit, 0 being the earliest
#         # i.e. i = 0 corresponds to t = 1 in notes
#
#         # construct the effective Hamiltonian for the ith step/layer
#         if i == self.d - 1:
#             Hi = self.make_full_dim(self.Sigmas[i]) + self.H_problem
#
#         else:
#             Hi = self.make_full_dim(self.Sigmas[i]) - \
#                  self.make_full_dim(self.noisy_dual_layer(i + 1))
#
#         Ei = jnp.linalg.eigvalsh(Hi)
#
#         cost += -self.Lambdas[i] * jnp.log(jnp.sum(jnp.exp(-Ei/self.Lambdas[i])))
#
#     # the initial state term
#     epsilon1_dag_sigma1 = self.make_full_dim(self.noisy_dual_layer(0))
#
#     cost += -jnp.trace(jnp.matmul(self.rho_init, epsilon1_dag_sigma1))
#
#     return -jnp.real(cost)
#
# def dual_grad(self, dual_vars: jnp.array):
#
#     return grad(self.dual_obj)(dual_vars)
#
#
#
# def make_full_dim(self, var_tensors: Dict):
#
#     full_mat = 0
#
#     for site_tuple, local_var_tensor in var_tensors.items():
#
#         dim_left = self.local_dim ** site_tuple[0]
#         dim_right = self.local_dim ** (self.num_sites_in_lattice - site_tuple[1] - 1)
#
#         identity_left = jnp.identity(dim_left)
#         identity_right = jnp.identity(dim_right)
#
#         full_mat += jnp.kron(identity_left, jnp.kron(local_var_tensor, identity_right))
#
#     return full_mat
#
# def noisy_dual_layer(self, i: int):
#
#     """
#     Applies the noise followed by unitary corresponding to a circuit layer
#     to the dual variable Sigma. The noise model is depolarizing noise on
#     each qubit.
#
#     Params:
#         i: layer number. Used to specify the tensors to act on and the gate
#         params to use.
#     Returns:
#         res_tensors = tensors after the dual layer
#     """
#
#     res_tensors = self.Sigmas[i].copy()
#     res_tensors = self.noise_layer(res_tensors)
#     res_tensors = self.dual_unitary_layer(layer_num = i + 1,
#                                     var_tensors = res_tensors)
#                                     # i + 1 here because unitary_layer
#                                     # counts layers differently
#
#     return res_tensors
#
# def noise_layer(self, var_tensors: Dict):
#
#     """
#     Applies depolarizing noise on the var_tensors at all sites.
#     """
#
#     res_tensors = var_tensors.copy()
#
#     for site_tuple, local_var_tensor in var_tensors.items():
#
#         local_res_tensor = (1 - 3 * self.p/4) * local_var_tensor + (self.p/4) * \
#             (jnp.matmul(self.X_left, jnp.matmul(local_var_tensor, self.X_left)) + \
#              jnp.matmul(self.Y_left, jnp.matmul(local_var_tensor, self.Y_left)) + \
#              jnp.matmul(self.Z_left, jnp.matmul(local_var_tensor, self.Z_left)))
#
#         local_res_tensor = (1 - 3 * self.p/4) * local_res_tensor + (self.p/4) * \
#             (jnp.matmul(self.X_right, jnp.matmul(local_res_tensor, self.X_right)) + \
#              jnp.matmul(self.Y_right, jnp.matmul(local_res_tensor, self.Y_right)) + \
#              jnp.matmul(self.Z_right, jnp.matmul(local_res_tensor, self.Z_right)))
#
#         res_tensors[site_tuple] = local_res_tensor
#
#     return res_tensors
#
# def dual_unitary_layer(self, layer_num: int, var_tensors: Dict):
#
#     res_tensors = var_tensors.copy()
#
#     if layer_num % 2 == 0:
#         res_tensors = self.mixing_unitaries(layer_num, res_tensors)
#
#     res_tensors = self.problem_unitaries(layer_num, res_tensors)
#
#     return res_tensors
#
# def problem_unitaries(self, layer_num: int, var_tensors: Dict):
#
#     #----- Applying the problem unitary -----#
#     gamma = self.opt_gamma[layer_num - 1]
#
#     # U = exp(-i gamma/2 w * (I - Z_j Z_k))
#     U = jnp.diag(jnp.array([1, jnp.exp(1j * gamma), jnp.exp(1j * gamma), 1]))
#     U_dag = jnp.transpose(jnp.conj(U))
#
#     res_tensors = var_tensors.copy()
#
#     for site_tuple, local_var_tensor in var_tensors.items():
#
#         if site_tuple in self.graph.edges:
#
#             local_res_tensor = jnp.matmul(U_dag, jnp.matmul(local_var_tensor, U))
#             res_tensors[site_tuple] = local_res_tensor
#
#     return res_tensors
#
# def mixing_unitaries(self, layer_num: int, var_tensors: Dict):
#
#     assert layer_num % 2 == 0
#
#     #----- Applying the mixing unitary -----#
#     beta = self.opt_beta[(layer_num - 2)//2]
#
#     Ux = jnp.cos(beta) * self.I_jax - 1j * jnp.sin(beta) * self.X_jax
#
#     Ux_2site = jnp.kron(Ux, Ux)
#     Ux_2site_dag = jnp.transpose(jnp.conj(Ux_2site))
#
#     res_tensors = var_tensors
#
#     for site_tuple, local_var_tensor in var_tensors.items():
#
#         local_res_tensor = jnp.matmul(Ux_2site_dag, jnp.matmul(local_var_tensor, Ux_2site))
#         res_tensors[site_tuple] = local_res_tensor
#
#     return res_tensors
#
# def primal_noisy(self):
#
#     rho_init_qutip = self.psi_init * self.psi_init.dag()
#
#     gamma = self.opt_gamma
#     beta = self.opt_beta
#
#     rho_after_step = rho_init_qutip
#
#     for layer_num in range(1, self.d + 1):
#
#         H_2site = qutip.tensor([self.Z_qutip] * 2) - qutip.tensor([self.I_qutip] * 2)
#         U_2site = (-1j * (gamma[layer_num - 1]/2) * H_2site).expm()
#
#         # problem unitaries
#         if layer_num % 2 == 0:
#             site_index_list = self.site_tuple_list_even_layer
#         else:
#             site_index_list = self.site_tuple_list_odd_layer
#
#         for site_tuple in site_index_list:
#
#             if site_tuple in self.graph.edges:
#
#                 U_2site_full = qutip.tensor(
#                                 [self.I_qutip] * site_tuple[0] +
#                                 [U_2site] +
#                                 [self.I_qutip] * (self.num_sites_in_lattice - site_tuple[0] - 2))
#                 rho_after_step = U_2site_full * rho_after_step * U_2site_full.dag()
#
#         # mixing unitaries
#         if layer_num % 2 == 0:
#
#             U_1site = (-1j * beta[(layer_num - 2)//2] * self.X_qutip).expm()
#
#             for site in self.graph:
#
#                 site_num = self.site_nums[site]
#
#                 U_1site_full = qutip.tensor([self.I_qutip] * site_num + [U_1site] + \
#                         [self.I_qutip] * (self.num_sites_in_lattice - site_num - 1))
#
#                 rho_after_step = U_1site_full * rho_after_step * U_1site_full.dag()
#
#         # noise
#         for site in self.lattice:
#
#             rho_after_step = (1 - 3 * self.p/4) * rho_after_step \
#             + (self.p/4) * (self.site_x_ops[site] * rho_after_step * self.site_x_ops[site] + \
#                             self.site_y_ops[site] * rho_after_step * self.site_y_ops[site] + \
#                             self.site_z_ops[site] * rho_after_step * self.site_z_ops[site])
#
#     return qutip.expect(self.H, rho_after_step)
