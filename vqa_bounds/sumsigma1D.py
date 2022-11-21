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
import jax
from jax import jit, grad, vmap, value_and_grad, hessian, jvp
from jax.example_libraries import optimizers

from vqa_bounds import graphs, meta_system

class SumSigma1D():

    def __init__(self, key, lattice, d_purity: int, d_vne: int, p: float, circ_backend = "qutip", mode = "local"):

        self.lattice = lattice
        self.graph = lattice

        self.circ_backend = circ_backend

        # preparing the Z operators
        self.site_z_ops = {}
        self.site_x_ops = {}
        self.site_y_ops = {}

        self.site_nums = {}
        op_num = 0

        self.num_sites_in_lattice = self.lattice.number_of_nodes() # assuming even
        # layer numbering starts from 1
        self.site_tuple_list = list(zip(range(0, self.num_sites_in_lattice, 2), range(1, self.num_sites_in_lattice, 2)))

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

        # for dual
        self.local_dim = 2
        self.d_purity = d_purity
        self.d_vne = d_vne
        self.d = self.d_purity + self.d_vne
        self.p = p
        self.psi_init = self.init_state()
        self.psi_init_jax = jnp.array(self.psi_init)
        self.H_problem = jnp.array(self.H.full())

        self.local_var_dim = self.local_dim ** 2

        self.num_vars_local = (self.local_dim ** 2) ** 2
        # because local H are over two sites
        self.num_vars_layers = self.d * self.num_vars_local * self.num_sites_in_lattice//2
        # self.total_num_vars = self.d_vne + self.num_vars_layers

        self.total_num_vars = self.num_vars_layers

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

        self.I_jax = jnp.array(self.I_qutip.full())
        self.X_jax = jnp.array(self.X_qutip.full())

        # # constructing initial density operator
        self.rho_init = jnp.array((self.psi_init * self.psi_init.dag()).full())

        self.key = key
        beta_half = jax.random.normal(self.key, shape = (self.num_sites_in_lattice, self.d//2))
        self.beta = jnp.column_stack((beta_half, -beta_half[:, ::-1]))

        self.init_entropy_bounds()

        if mode == "local":
            self.dual_obj = self.dual_obj_local
            self.dual_grad = self.dual_grad_local
            self.dual_hessian_prod = self.dual_hessian_prod_local

        elif mode == "nc":
            self.dual_obj = self.dual_obj_nc
            self.dual_grad = self.dual_grad_nc

        else:
            raise ValueError("Method not yet implemented")

        self.a_vars = 0

    def init_state(self):

        psi0 = qutip.basis([2] * self.num_sites_in_lattice, n = [1] * self.num_sites_in_lattice)

        return psi0

    def init_entropy_bounds(self):

        q = 1 - self.p
        q_powers = jnp.array([q**i for i in range(self.d)])

        self.entropy_bounds = self.num_sites_in_lattice * self.p * jnp.log(2) * \
                              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(self.d)])

        self.purity_bounds = jnp.exp(-self.entropy_bounds)

    def dual_obj_local(self, dual_vars: jnp.array):

        # Unvectorize the vars_vec and construct Sigmas and Lambdas
        self.assemble_vars_into_tensors(dual_vars)

        # Compute the objective function using the list of tensors
        obj = self.cost_local()

        return -jnp.real(obj)

    def dual_grad_local(self, dual_vars: jnp.array):

        return grad(self.dual_obj_local)(dual_vars)

    def dual_hessian_prod_local(self, dual_vars: jnp.array, vec: jnp.array):

        return jvp(grad(self.dual_obj_local), (dual_vars,), (vec,))[1]

    def assemble_vars_into_tensors(self, vars_vec: jnp.array):

        # a_vars = vars_vec.at[:self.d_vne].get()
        # # self.Lambdas = jnp.log(1 + jnp.exp(a_vars))
        # self.Lambdas = jnp.exp(a_vars)
        #
        # vars_vec_layers = vars_vec.at[self.d_vne:].get()

        vars_vec_layers = vars_vec

        # split the variables into equal arrays corresponding to each layer
        vars_vec_layers_split = jnp.split(vars_vec_layers, self.d)

        # list where each element corresponds to the dual variable for a layer
        # each element is a dictionary with site tuples as keys and the local
        # Hamiltonians as values
        self.Sigmas = []

        for i in range(0, self.d):

            Sigma_layer = {}

            # for each circuit step the variables are arranged as:
            # [vars_diag, vars_real, vars_imag]

            vars_layer = vars_vec_layers_split[i]
            vars_layer_split = jnp.split(vars_layer, len(self.site_tuple_list))
            site_index_list = self.site_tuple_list

            for n, site_tuple in enumerate(site_index_list):

                vars_diag = vars_layer_split[n].at[:self.num_diag_elements_local].get()
                vars_real = vars_layer_split[n].at[self.num_diag_elements_local:self.num_diag_elements_local + self.num_tri_elements_local].get()
                vars_imag = vars_layer_split[n].at[self.num_diag_elements_local + self.num_tri_elements_local:].get()

                tri_real = jnp.zeros((self.local_var_dim, self.local_var_dim), dtype = complex)
                tri_imag = jnp.zeros((self.local_var_dim, self.local_var_dim), dtype = complex)

                tri_real = tri_real.at[self.utri_indices_local].set(vars_real)
                tri_real = tri_real.at[self.ltri_indices_local].set(vars_real)
                tri_imag = tri_imag.at[self.utri_indices_local].set(1j * vars_imag)
                tri_imag = tri_imag.at[self.ltri_indices_local].set(-1j * vars_imag)

                vars_full = jnp.diag(vars_diag) + tri_real + tri_imag

                Sigma_layer[site_tuple] = vars_full

            self.Sigmas.append(Sigma_layer)

    def cost_local(self):

        # the entropy term
        # cost = jnp.dot(self.Lambdas, self.entropy_bounds)

        self.Lambdas = jnp.exp(self.a_vars)

        cost = jnp.dot(self.Lambdas, self.entropy_bounds.at[self.d_purity:].get())

        # the effective Hamiltonian term
        for i in range(self.d_purity):
            # i corresponds to the layer of the circuit, 0 being the earliest
            # i.e. i = 0 corresponds to t = 1 in notes

            # construct the effective Hamiltonian for the ith step/layer
            Hi = self.construct_H(i)
            Hi_squared = jnp.matmul(Hi, Hi)

            cost += -jnp.sqrt(self.purity_bounds.at[i].get() * jnp.trace(Hi_squared))

        for i in range(self.d_purity, self.d):

            # i corresponds to the layer of the circuit, 0 being the earliest
            # i.e. i = 0 corresponds to t = 1 in notes

            # construct the effective Hamiltonian for the ith step/layer
            Hi = self.construct_H(i)
            Ei = jnp.linalg.eigvalsh(Hi)

            cost += -self.Lambdas[0] * jnp.log(jnp.sum(jnp.exp(-Ei/self.Lambdas[0])))

        # the initial state term
        epsilon1_dag_sigma1 = self.make_full_dim(self.noisy_dual_layer(0))

        cost += -jnp.trace(jnp.matmul(self.rho_init, epsilon1_dag_sigma1))

        return cost

    # def cost_last_layer(self):
    #
    #     # the entropy term
    #     # cost = jnp.dot(self.Lambdas, self.entropy_bounds)
    #     cost = jnp.dot(self.Lambdas, self.entropy_bounds.at[self.d_purity:].get())
    #
    #     # # the effective Hamiltonian term
    #     # for i in range(self.d_purity):
    #     #     # i corresponds to the layer of the circuit, 0 being the earliest
    #     #     # i.e. i = 0 corresponds to t = 1 in notes
    #     #
    #     #     # construct the effective Hamiltonian for the ith step/layer
    #     #     Hi = self.construct_H(i)
    #     #     Hi_squared = jnp.matmul(Hi, Hi)
    #     #
    #     #     cost += -jnp.sqrt(self.purity_bounds.at[i].get() * jnp.trace(Hi_squared) + 1e-9)
    #
    #     for i in range(self.d_purity, self.d):
    #
    #         # i corresponds to the layer of the circuit, 0 being the earliest
    #         # i.e. i = 0 corresponds to t = 1 in notes
    #
    #         # construct the effective Hamiltonian for the ith step/layer
    #         Hi = self.construct_H(i)
    #         Ei = jnp.linalg.eigvalsh(Hi)
    #
    #         cost += -self.Lambdas[i] * jnp.log(jnp.sum(jnp.exp(-Ei/self.Lambdas[i])))
    #
    #     # the initial state term
    #     # epsilon1_dag_sigma1 = self.make_full_dim(self.noisy_dual_layer(0))
    #
    #     # cost += -jnp.trace(jnp.matmul(self.rho_init, epsilon1_dag_sigma1))
    #
    #     return cost
    #
    # def dual_obj_last_layer(self, dual_vars: jnp.array):
    #
    #     # Unvectorize the vars_vec and construct Sigmas and Lambdas
    #     self.assemble_vars_into_tensors(dual_vars)
    #
    #     # Compute the objective function using the list of tensors
    #     obj = self.cost_last_layer()
    #
    #     return -jnp.real(obj)
    #
    # def dual_grad_last_layer(self, dual_vars: jnp.array):
    #
    #     return grad(self.dual_obj_last_layer)(dual_vars)
    #
    # def dual_hessian_last_layer(self, dual_vars: jnp.array):
    #
    #     return hessian(self.dual_obj_last_layer)(dual_vars)

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
            Hi = self.make_full_dim(self.Sigmas[i]) + self.H_problem

        else:
            Hi = self.make_full_dim(self.Sigmas[i]) - \
                 self.make_full_dim(self.noisy_dual_layer(i + 1))

        return Hi

    def make_full_dim(self, var_tensors: Dict):

        full_mat = 0

        for site_tuple, local_var_tensor in var_tensors.items():

            dim_left = self.local_dim ** site_tuple[0]
            dim_right = self.local_dim ** (self.num_sites_in_lattice - site_tuple[1] - 1)

            identity_left = jnp.identity(dim_left)
            identity_right = jnp.identity(dim_right)

            full_mat += jnp.kron(identity_left, jnp.kron(local_var_tensor, identity_right))

        return full_mat

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

        res_tensors = self.Sigmas[i].copy()
        res_tensors = self.noise_layer(res_tensors)
        res_tensors = self.dual_unitary_layer(layer_num = i,
                                        var_tensors = res_tensors)
                                        # i + 1 here because unitary_layer
                                        # counts layers differently

        return res_tensors

    def noise_layer(self, var_tensors: Dict):

        """
        Applies depolarizing noise on the var_tensors at all sites.
        """

        res_tensors = var_tensors.copy()

        for site_tuple, local_var_tensor in var_tensors.items():

            local_res_tensor = (1 - 3 * self.p/4) * local_var_tensor + (self.p/4) * \
                (jnp.matmul(self.X_left, jnp.matmul(local_var_tensor, self.X_left)) + \
                 jnp.matmul(self.Y_left, jnp.matmul(local_var_tensor, self.Y_left)) + \
                 jnp.matmul(self.Z_left, jnp.matmul(local_var_tensor, self.Z_left)))

            local_res_tensor = (1 - 3 * self.p/4) * local_res_tensor + (self.p/4) * \
                (jnp.matmul(self.X_right, jnp.matmul(local_res_tensor, self.X_right)) + \
                 jnp.matmul(self.Y_right, jnp.matmul(local_res_tensor, self.Y_right)) + \
                 jnp.matmul(self.Z_right, jnp.matmul(local_res_tensor, self.Z_right)))

            res_tensors[site_tuple] = local_res_tensor

        return res_tensors

    def dual_unitary_layer(self, layer_num: int, var_tensors: Dict):

        res_tensors = var_tensors.copy()

        res_tensors = self.sq_unitaries(layer_num, res_tensors)

        return res_tensors

    def sq_unitaries(self, layer_num: int, var_tensors: Dict):

        #----- Applying the mixing unitary -----#
        res_tensors = var_tensors

        for site_tuple, local_var_tensor in var_tensors.items():

            beta_left = self.beta.at[site_tuple[0], layer_num].get()
            beta_right = self.beta.at[site_tuple[1], layer_num].get()

            Ux_left = jnp.cos(beta_left) * self.I_jax - 1j * jnp.sin(beta_left) * self.X_jax
            Ux_right = jnp.cos(beta_right) * self.I_jax - 1j * jnp.sin(beta_right) * self.X_jax

            Ux_2site = jnp.kron(Ux_left, Ux_right)
            Ux_2site_dag = jnp.transpose(jnp.conj(Ux_2site))

            local_res_tensor = jnp.matmul(Ux_2site_dag, jnp.matmul(local_var_tensor, Ux_2site))
            res_tensors[site_tuple] = local_res_tensor

        return res_tensors

    def primal_noisy(self):

        rho_init_qutip = self.psi_init * self.psi_init.dag()

        rho_after_step = rho_init_qutip

        for layer_num in range(0, self.d):

            site_index_list = self.site_tuple_list

            # mixing unitaries
            for site in self.graph:
                U_1site = (-1j * float(self.beta[site, layer_num]) * self.X_qutip).expm()

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

    def dual_obj_nc(self, dual_nc_vars: jnp.array):

        # lmbda = jnp.log(1 + jnp.exp(dual_nc_vars[0]))
        lmbda = jnp.exp(dual_nc_vars[0])

        cost = lmbda * self.entropy_bounds[-1]
        Hi = self.H_problem
        Ei = jnp.linalg.eigvalsh(Hi)

        cost += -lmbda * jnp.log(jnp.sum(jnp.exp(-Ei/lmbda)))

        return -jnp.real(cost)

    def dual_grad_nc(self, dual_nc_vars: jnp.array):

        return grad(self.dual_obj_nc)(dual_nc_vars)
