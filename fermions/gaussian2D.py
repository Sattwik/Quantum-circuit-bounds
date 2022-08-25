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
import jax.scipy.linalg
from jax import jit, grad, vmap, value_and_grad
from jax.example_libraries import optimizers

#--------------------------------------------------#
#--------------- Optimization tools ---------------#
#--------------------------------------------------#

def unjaxify_obj(func):
    def wrap(*args):
        return float(func(jnp.array(args[0]), args[1]))

    return wrap

def unjaxify_grad(func):
    def wrap(*args):
        return np.array(func(jnp.array(args[0]), args[1]), order = 'F')

    return wrap

def optimize(vars_init: np.array, params, obj_fun: Callable, grad_fun: Callable,
             num_iters: int = 500, bounds = None, opt_method: str = "L-BFGS-B"):

    opt_args = (params,)

    obj_over_opti = []

    def callback_func(x):
        obj_eval = unjaxify_obj(obj_fun)(x, opt_args[0])
        obj_over_opti.append(obj_eval)

    bnds = bounds

    if opt_method == "L-BFGS-B":
        opt_result = scipy.optimize.minimize(
                                unjaxify_obj(obj_fun),
                                vars_init, args = opt_args,
                                method = opt_method,
                                jac = unjaxify_grad(grad_fun),
                                options={'disp': None,
                                'maxcor': 10,
                                'ftol': 2.220446049250313e-09,
                                'gtol': 1e-05,
                                'eps': 1e-08,
                                'maxfun': 15000,
                                'maxiter': num_iters,
                                'iprint': 10,
                                'maxls': 20},
                                bounds = bnds,
                                callback = callback_func)
    elif opt_method == "BFGS":
        opt_result = scipy.optimize.minimize(
                                unjaxify_obj(obj_fun),
                                vars_init, args = opt_args,
                                method = opt_method,
                                jac = unjaxify_grad(grad_fun),
                                options={'gtol': 1e-02,
                                'eps': 1.4901161193847656e-08,
                                'maxiter': num_iters,
                                'disp': True,
                                'return_all': False},
                                callback = callback_func)

    return np.array(obj_over_opti), opt_result

def adam_optimize(obj_fun: Callable, grad_fun: Callable,
                  vars_init: jnp.array, params,
                  alpha: float, num_steps: int):

    init, update, get_params = optimizers.adam(alpha)

    def step(t, opt_state):
        value = obj_fun(get_params(opt_state), params)
        grads = grad_fun(get_params(opt_state), params)
        opt_state = update(t, grads, opt_state)
        return value, opt_state

    opt_state = init(vars_init)
    value_array = jnp.zeros(num_steps)

    for t in range(num_steps):
        # if t%(num_steps//10) == 0:
        #     print("Step :", t)
        print("Step :", t)
        value, opt_state = step(t, opt_state)
        value_array = value_array.at[t].set(value)

    return value_array, get_params(opt_state)


#--------------------------------------------------#
#------------------ Lattice tools -----------------#
#--------------------------------------------------#

def k_neighbors(k: int, M: int, N: int):
    """
    k: interaction range
    M, N: lattice dims
    """

    neighbors_list = []

    for s in range(M * N):
        i_0 = s//N
        j_0 = s%N

        neighbors_of_s = [s]

        for i in range(i_0 - k, i_0 + k + 1):
            for j in range(j_0 - k, j_0 + k + 1):
                if (np.abs(i - i_0) + np.abs(j - j_0) <= k) and \
                   0 <= i < M and \
                   0 <= j < N and \
                   s != N * i + j:
                    neighbors_of_s.append(N * i + j)

        neighbors_list.append(neighbors_of_s)

    return neighbors_list

def upper_tri_indices_to_fill(k: int, M: int, N: int):
    neighbors_list = k_neighbors(k, M, N)

    row_indices = []
    col_indices = []

    for neighbors in neighbors_list:
        s_0 = neighbors[0]

        for s in neighbors[1:]:
            if s > s_0:
                row_indices.append(s_0)
                col_indices.append(s)

            # row_indices.append(s)
            # col_indices.append(s_0)

    return (jnp.array(row_indices), jnp.array(col_indices))

#--------------------------------------------------#
#-------------------- FGS tools -------------------#
#--------------------------------------------------#

def random_2D_k_local_normal_hamiltonian_majorana(M: int, N: int, k: int, key: jnp.array):
    """
    Parameters
    ----------
    M, N: size of lattice
    k < N: range of interactions (Manhattan distance)

    Returns
    -------
    A random k-local f.g.h. (2MN x 2MN) in Majorana representation
    """
    utri_indices = upper_tri_indices_to_fill(k, M, N)
    utri_rows, utri_cols = utri_indices
    ltri_indices = (utri_cols, utri_rows)
    diag_indices = jnp.diag_indices(M * N)

    num_utri_elements = len(utri_indices[0])
    num_diag_elements = M * N

    hxx = jnp.zeros((M*N, M*N))
    key, subkey = jax.random.split(key)
    hxx = hxx.at[utri_indices].set(jax.random.normal(subkey, (num_utri_elements,)))
    hxx = hxx - hxx.T

    hpp = jnp.zeros((M*N, M*N))
    key, subkey = jax.random.split(key)
    hpp = hpp.at[utri_indices].set(jax.random.normal(subkey, (num_utri_elements,)))
    hpp = hpp - hpp.T

    hxp = jnp.zeros((M*N, M*N))
    key, subkey = jax.random.split(key)
    hxp = hxp.at[utri_indices].set(jax.random.normal(subkey, (num_utri_elements,)))
    key, subkey = jax.random.split(key)
    hxp = hxp.at[ltri_indices].set(jax.random.normal(subkey, (num_utri_elements,)))
    key, subkey = jax.random.split(key)
    hxp = hxp.at[diag_indices].set(jax.random.normal(subkey, (num_diag_elements,)))

    h = jnp.block([[hxx, hxp],
                  [-hxp.T, hpp]])

    return h, key


def Omega(N: int) -> jnp.array:
    return jnp.sqrt(1/2) * jnp.block(
                [[jnp.eye(N), jnp.eye(N)],
                 [1j * jnp.eye(N), -1j * jnp.eye(N)]])

def covariance_from_corr_major(Gamma_mjr: jnp.array):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix of f.g.s. (majorana rep.)
    """
    N = Gamma_mjr.shape[0]//2
    gamma = 1j * (2 * Gamma_mjr - jnp.identity(2 * N, dtype = complex))
    return gamma

def corr_major_from_covariance(gamma: jnp.array):
    """
    Parameters
    ----------
    gamma: Covariance matrix of f.g.s. (majorana rep.)
    """
    N = gamma.shape[0]//2
    Gamma_mjr = (jnp.identity(2 * N, dtype = complex) - 1j * gamma)/2.0
    return Gamma_mjr

def corr_from_corr_major(gamma: jnp.array):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix of f.g.s. (majorana rep.)
    """
    N = gamma.shape[0]//2
    Ome = Omega(N)
    Gamma = jnp.matmul(jnp.matmul(Ome.conj().T, Gamma_mjr), Ome)
    return Gamma

def corr_major_from_parenth(h: jnp.array):
    """
    Parameters
    ----------
    h: Parent Hamiltonian of f.g.h.
    """
    N = h.shape[0]//2
    w, v = jnp.linalg.eig(-2j * h)
    w_Gamma_mjr = jnp.diag((jnp.ones(2*N) + jnp.exp(w)) ** (-1))

    Gamma_mjr = jnp.matmul(jnp.matmul(v, w_Gamma_mjr), v.conj().T)
    return Gamma_mjr

#--------------------------------------------------#
#----------------- Primal methods -----------------#
#--------------------------------------------------#

def energy(Gamma_mjr: jnp.array, h: jnp.array):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix of f.g.s. (majorana rep.)
    h: Hamiltonian of which to calculate expectation (majorana rep.)

    Returns
    -------
    Expected energy
    """
    return -1j * jnp.trace(jnp.matmul(h, Gamma_mjr))

@jit
def unitary_on_fgstate(Gamma_mjr: jnp.array, h: jnp.array):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix of f.g.s. (majorana rep.)
    h: Generator of Gaussian unitary in majorana rep..
       Gaussian unitary = e^{-iH} where H = i r^{\dagger} h r.

    Returns
    -------
    Gamma_mjr_prime: Correlation matrix of f.g.s. after unitary.
    """
    w, v = jnp.linalg.eig(2 * h)

    exp_p2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(w))), jnp.conj(jnp.transpose(v)))
    exp_m2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(-w))), jnp.conj(jnp.transpose(v)))

    return jnp.matmul(jnp.matmul(exp_p2h, Gamma_mjr), exp_m2h)

class PrimalParams():
    """
    Class to store parameters required to define the primal problem.
    """

    def __init__(self, M: int, N: int, d: int, local_d: int,
                 key: jnp.array,
                 init_state_desc: str = "all zero", k: int = 1):
        """
        d: depth of circuit
        init_state_desc: description of the initial state wanted

        Assuming (d - local_d) is even
        """
        self.M = M
        self.N = N
        self.d = d
        self.local_d = local_d
        self.k = k

        assert((d - local_d)%2 == 0)

        self.generate_layer_hamiltonians(key)
        self.generate_parent_hamiltonian()
        self.generate_init_state(init_state_desc)

    def generate_layer_hamiltonians(self, key: jnp.array):
        self.layer_hamiltonians = []

        for i in range(self.local_d):
            random_local_h, key = \
            random_2D_k_local_normal_hamiltonian_majorana(self.M, self.N, self.k, key)
            self.layer_hamiltonians.append(random_local_h/self.M/self.N)

        for i in range((self.d - self.local_d)//2):
            random_local_h, key = \
            random_2D_k_local_normal_hamiltonian_majorana(self.M, self.N, self.k, key)
            self.layer_hamiltonians.append(random_local_h/self.M/self.N)
            self.layer_hamiltonians.append(-random_local_h/self.M/self.N)

        self.layer_hamiltonians = jnp.array(self.layer_hamiltonians)
        self.key_after_ham_gen = key

    def generate_parent_hamiltonian(self):
        Ome = Omega(self.M * self.N)
        epsilon = jnp.linspace(start = 1, stop = 0, num = self.M * self.N)
        D = jnp.diag(jnp.concatenate((-epsilon, epsilon)))
        d_parent = -1j * jnp.matmul(Ome, jnp.matmul(D, Ome.conj().T))
        h_parent = d_parent

        for i in range(self.local_d):
            h_parent = unitary_on_fghamiltonian(h_parent, self.layer_hamiltonians.at[i,:,:].get())
        self.h_parent = h_parent

    def generate_init_state(self, init_state_desc: str = "all zero"):
        if init_state_desc == "all zero":
            I = jnp.identity(self.M * self.N, dtype = complex)

            self.Gamma_mjr_init = 0.5 * jnp.block(
            [[I      , 1j * I],
             [-1j * I, I     ]])

def energy_after_circuit(params: PrimalParams):
    Gamma_mjr = params.Gamma_mjr_init

    for i in range(params.local_d):
        Gamma_mjr = unitary_on_fgstate(Gamma_mjr, params.layer_hamiltonians.at[i, :, :].get())

    return jnp.real(energy(Gamma_mjr, params.h_parent))

def noise_on_kth_mode(k: int, args: Tuple):
    Gamma_mjr, p = args
    MN = Gamma_mjr.shape[0]//2
    gamma = covariance_from_corr_major(Gamma_mjr)

    gamma_replaced = gamma.at[k, :].set(jnp.zeros(2 * MN))
    gamma_replaced = gamma_replaced.at[:, k].set(jnp.zeros(2 * MN))
    gamma_replaced = gamma_replaced.at[k + MN, :].set(jnp.zeros(2 * MN))
    gamma_replaced = gamma_replaced.at[:, k + MN].set(jnp.zeros(2 * MN))

    gamma_noisy = p * gamma_replaced + (1 - p) * gamma
    Gamma_mjr_noisy = corr_major_from_covariance(gamma_noisy)

    return (Gamma_mjr_noisy, p)

def noisy_primal_ith_layer(i: int, args: Tuple):
    Gamma_mjr, layer_hamiltonians, MN, p = args
    Gamma_mjr = unitary_on_fgstate(Gamma_mjr, layer_hamiltonians.at[i, :, :].get())
    init_args = (Gamma_mjr, p)
    Gamma_mjr, _ = jax.lax.fori_loop(0, MN, noise_on_kth_mode, init_args)
    return (Gamma_mjr, layer_hamiltonians, MN, p)

def noisy_primal(params:PrimalParams, p: float):
    Gamma_mjr = params.Gamma_mjr_init
    layer_hamiltonians = params.layer_hamiltonians
    M = params.M
    N = params.N
    init_args = (Gamma_mjr, layer_hamiltonians, M * N, p)
    Gamma_mjr, _, _, _ = jax.lax.fori_loop(0, params.d, noisy_primal_ith_layer, init_args)

    return jnp.real(energy(Gamma_mjr, params.h_parent))

#--------------------------------------------------#
#------------------ Dual methods ------------------#
#--------------------------------------------------#

class DualParams():
    def __init__(self, circ_params: PrimalParams, p: float, k_dual: int,
                 lambda_lower_bounds: jnp.array, scale: float = 1):
        self.circ_params = circ_params
        self.p = p
        self.k_dual = k_dual
        self.lambda_lower_bounds = lambda_lower_bounds
        self.scale = scale

        self.block_utri_indices = \
        upper_tri_indices_to_fill(self.k_dual, circ_params.M, circ_params.N)
        utri_rows, utri_cols = self.block_utri_indices
        self.block_ltri_indices = (utri_cols, utri_rows)
        self.block_diag_indices = jnp.diag_indices(circ_params.M * circ_params.N)

        ltri_rows, ltri_cols = self.block_ltri_indices
        diag_rows, diag_cols = self.block_diag_indices

        xp_rows = jnp.concatenate((utri_rows, ltri_rows, diag_rows))
        xp_cols = jnp.concatenate((utri_cols, ltri_cols, diag_cols))

        self.xp_indices = (xp_rows, xp_cols)

        self.num_utri_elements = len(self.block_utri_indices[0])
        self.num_diag_elements = circ_params.M * circ_params.N

        self.num_vars_xx = self.num_utri_elements
        self.num_vars_pp = self.num_vars_xx
        self.num_vars_xp = 2 * self.num_utri_elements + self.num_diag_elements

        self.num_sigma_vars_layer = self.num_vars_xx + self.num_vars_pp + self.num_vars_xp

        self.total_num_dual_vars = self.circ_params.d + \
        self.circ_params.d * self.num_sigma_vars_layer

def unvec_layer_i(i: int, args: Tuple):
    sigmas, sigma_vars, \
    block_utri_indices, block_ltri_indices, block_diag_indices, xp_indices, \
    zeros_MN = args

    # number of variables in blocks of the full sigma dual var
    l_xx = block_utri_indices[0].shape[0]
    l_pp = l_xx
    l_xp = xp_indices[0].shape[0]

    # total_num_vars
    l = l_xp + l_xx + l_pp

    # slicing the input dual vars vector
    sigma_slice = jax.lax.dynamic_slice_in_dim(sigma_vars, i * l, l)
    sigma_slice_xx = sigma_slice.at[:l_xx].get()
    sigma_slice_pp = sigma_slice.at[l_xx : l_xx + l_pp].get()
    sigma_slice_xp = sigma_slice.at[l_xx + l_pp:].get()

    # making block matrices
    sigma_layer_xx = zeros_MN.at[block_utri_indices].set(sigma_slice_xx)
    sigma_layer_xx = sigma_layer_xx.at[block_ltri_indices].set(-sigma_slice_xx)

    sigma_layer_pp = zeros_MN.at[block_utri_indices].set(sigma_slice_pp)
    sigma_layer_pp = sigma_layer_pp.at[block_ltri_indices].set(-sigma_slice_pp)

    sigma_layer_xp = zeros_MN.at[xp_indices].set(sigma_slice_xp)

    # putting blocks together
    sigma_layer = jnp.block([[sigma_layer_xx, sigma_layer_xp],
                             [-sigma_layer_xp.T, sigma_layer_pp]])

    # putting layer var into composite structure
    sigmas = sigmas.at[:,:,i].set(sigma_layer)

    return (sigmas, sigma_vars,
            block_utri_indices, block_ltri_indices,
            block_diag_indices, xp_indices,
            zeros_MN)

@partial(jit, static_argnums = (1,))
def unvec_and_process_dual_vars(dual_vars: jnp.array, dual_params: DualParams):
    M = dual_params.circ_params.M
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    MN = M*N

    block_utri_indices = dual_params.block_utri_indices
    block_ltri_indices = dual_params.block_ltri_indices
    block_diag_indices = dual_params.block_diag_indices
    xp_indices = dual_params.xp_indices
    num_sigma_vars_layer = dual_params.num_sigma_vars_layer

    sigmas = jnp.zeros((2 * MN, 2 * MN, d))
    zeros_MN = jnp.zeros((MN, MN))

    a_vars = dual_vars.at[:d].get()
    lambdas = dual_params.lambda_lower_bounds + jnp.log(1 + jnp.exp(a_vars))

    sigma_vars = dual_vars.at[d:].get()
    init_args = (sigmas, sigma_vars,
                 block_utri_indices, block_ltri_indices,
                 block_diag_indices, xp_indices,
                 zeros_MN)
    sigmas, _, _, _, _, _, _ = jax.lax.fori_loop(0, d, unvec_layer_i, init_args)

    return lambdas, sigmas

# # @partial(jit, static_argnums = (2,))
@jit
def noisy_dual_layer(h_layer: jnp.array, sigma_layer: jnp.array, p: float):
    """
    h_layer: Generating Hamiltonian (majorana rep.) of the unitary layer of the
    circuit. Intended to be already scaled by circuit params.
    """
    sigma = noise_on_fghamiltonian(sigma_layer, p)
    sigma = unitary_on_fghamiltonian(sigma, -h_layer) # -ve because dual

    return sigma

def dual_free_energy_ith_term(i: int, args: Tuple):

    lambdas, sigmas, layer_hamiltonians, p, cost = args

    hi = sigmas.at[:,:,i].get() - \
         noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(),
                          sigmas.at[:,:,i+1].get(), p)

    cost += -lambdas.at[i].get() * log_trace_fgstate(-hi/lambdas.at[i].get())

    return (lambdas, sigmas, layer_hamiltonians, p, cost)

@partial(jit, static_argnums = (1,))
def dual_obj(dual_vars: jnp.array, dual_params: DualParams):
    M = dual_params.circ_params.M
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_parent = dual_params.circ_params.h_parent
    layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
    p = dual_params.p
    Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init

    lambdas, sigmas = unvec_and_process_dual_vars(dual_vars, dual_params)

    cost = 0
    # log Tr exp terms

    # first d - 1 layers
    init_args = (lambdas, sigmas, layer_hamiltonians, p, cost)
    _, _, _, _, cost = jax.lax.fori_loop(0, d - 1, dual_free_energy_ith_term, init_args)

    # last layer
    hi = h_parent + sigmas.at[:,:,d-1].get()
    cost += -lambdas.at[d-1].get() * log_trace_fgstate(-hi/lambdas[d-1])

    # init. state term
    epsilon_1_dag_sigma1 = \
    noisy_dual_layer(layer_hamiltonians.at[0, :, :].get(), sigmas.at[:,:,0].get(), p)

    cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)

    # entropy term
    q = 1 - p
    q_powers = jnp.array([q**i for i in range(d)])

    entropy_bounds = M * N * p * jnp.log(2) * \
             jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])

    cost += jnp.dot(lambdas, entropy_bounds)

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad(dual_vars: jnp.array, dual_params: DualParams):
    return grad(dual_obj, argnums = 0)(dual_vars, dual_params)

@jit
def log_trace_fgstate(parent_h: jnp.array):
    """
    Parameters
    ----------
    parent_h: Parent Hamiltonian of the f.g.s (Majorana rep.)

    Returns
    -------
    log of trace of f.g.s.
    """
    N = parent_h.shape[0]//2
    w, v = jnp.linalg.eigh(1j * parent_h)

    # print(w)

    positive_eigs = w.at[N:].get()
    eps_max = positive_eigs.at[-1].get()

    # print(positive_eigs)

    log_trace_of_shifted = jnp.sum(jnp.log(jnp.exp(positive_eigs - eps_max) +
                                           jnp.exp(-positive_eigs - eps_max)))
    log_trace = N * eps_max + log_trace_of_shifted

    return log_trace

@jit
def unitary_on_fghamiltonian(s: jnp.array, h: jnp.array):
    """
    Parameters
    ----------
    s: Hamiltonian (from dual variable) to act on (Majorana representation)
    h: Generator of unitary (Majorana representation)

    Returns
    -------
    Majorana rep after unitary
    """
    w, v = jnp.linalg.eig(2 * h)

    exp_p2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(w))), jnp.conj(jnp.transpose(v)))
    exp_m2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(-w))), jnp.conj(jnp.transpose(v)))

    return jnp.matmul(jnp.matmul(exp_p2h, s), exp_m2h)


def noise_on_fghamiltonian_at_index(k: int, args: Tuple):
    s_prime, p = args
    N = s_prime.shape[0]//2

    s_prime_zeroed_out = s_prime.at[k, :].set(jnp.zeros(2 * N))
    s_prime_zeroed_out = s_prime_zeroed_out.at[:, k].set(jnp.zeros(2 * N))
    s_prime_zeroed_out = s_prime_zeroed_out.at[k + N, :].set(jnp.zeros(2 * N))
    s_prime_zeroed_out = s_prime_zeroed_out.at[:, k + N].set(jnp.zeros(2 * N))

    s_prime = (1 - p) * s_prime + p * (s_prime_zeroed_out)

    return (s_prime, p)

# @partial(jit, static_argnums = (1,))
@jit
def noise_on_fghamiltonian(s: jnp.array, p: float):
    """
    Parameters
    ----------
    s: Hamiltonian (from dual variable) to act on (Majorana representation)
    p: noise probability

    Returns
    -------
    Majorana rep after depol. noise on individual fermions
    """
    N = s.shape[0]//2
    s_prime = s

    init_args = (s_prime, p)
    s_prime, _ = jax.lax.fori_loop(0, N, noise_on_fghamiltonian_at_index, init_args)

    return s_prime

#--------------------------------------------------#
#------------ No-channel dual methods -------------#
#--------------------------------------------------#

@partial(jit, static_argnums = (1,))
def dual_obj_no_channel(dual_vars: jnp.array, dual_params: DualParams):
    M = dual_params.circ_params.M
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_parent = dual_params.circ_params.h_parent
    p = dual_params.p

    q = 1 - p
    q_powers = jnp.array([q**i for i in range(d)])
    entropy_bounds = M * N * p * jnp.log(2) * \
             jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
    Sd = entropy_bounds.at[-1].get()

    a = dual_vars.at[0].get()
    lmbda = dual_params.lambda_lower_bounds.at[-1].get() + jnp.log(1 + jnp.exp(a))
    # lmbda = 1 + jnp.log(1 + jnp.exp(a))

    # cost = -lmbda * jnp.log(trace_fgstate(-h_parent/lmbda)) + lmbda * Sd
    cost = -lmbda * log_trace_fgstate(-h_parent/lmbda) + lmbda * Sd

    cost += jnp.dot(dual_params.lambda_lower_bounds.at[:-1].get(),
                    entropy_bounds.at[:-1].get() - M * N * np.log(2))

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad_no_channel(dual_vars: jnp.array, dual_params: DualParams):
    return grad(dual_obj_no_channel, argnums = 0)(dual_vars, dual_params)
