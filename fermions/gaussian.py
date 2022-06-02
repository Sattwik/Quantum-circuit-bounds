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

#--------------- Tools ---------------#
def unjaxify_obj(func):

    def wrap(*args):
        return float(func(jnp.array(args[0]), args[1]))

    return wrap

def unjaxify_grad(func):

    def wrap(*args):
        return np.array(func(jnp.array(args[0]), args[1]), order = 'F')

    return wrap

def optimize(vars_init: np.array, params, obj_fun: Callable, grad_fun: Callable,
             num_iters: int = 50, bounds = None, opt_method: str = "L-BFGS-B"):

    opt_args = (params,)

    obj_over_opti = []

    def callback_func(x):
        obj_eval = unjaxify_obj(obj_fun)(x, opt_args[0])
        obj_over_opti.append(obj_eval)

    if bounds is not None:
        bnds = bounds
    else:
        bnds = scipy.optimize.Bounds(lb = -np.inf, ub = np.inf)

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
                                options={'gtol': 1e-05,
                                'eps': 1.4901161193847656e-08,
                                'maxiter': num_iters,
                                'disp': True,
                                'return_all': False},
                                callback = callback_func)

    return np.array(obj_over_opti), opt_result

def random_normal_hamiltonian_majorana(N: int, key: jnp.array):
    """
    Parameters
    ----------
    N: number of fermionic modes

    Returns
    -------
    A random f.g.h. (2N x 2N) in Majorana representation
    """

    key, subkey = jax.random.split(key)
    h = jax.random.normal(subkey, (2*N, 2*N))

    return h - h.T, key

def random_normal_corr_majorana(N: int, Ome: jnp.array, key: jnp.array):
    """
    Parameters
    ----------
    N: number of fermionic modes

    Returns
    -------
    Correlation matrix (2N x 2N) in Majorana representation of a random f.g.s.
    """
    # generate occupation probabilities
    key, subkey = jax.random.split(key)
    f = jax.random.uniform(subkey, (N,))

    F = jnp.diag(jnp.concatenate((f, 1.0-f)))

    key, subkey = jax.random.split(key)
    random_symm_mat = jax.random.normal(subkey, (2*N, 2*N))

    random_symm_mat = random_symm_mat + random_symm_mat.conj().T

    _, O = jnp.linalg.eigh(random_symm_mat)

    V = jnp.matmul(O.T, Ome)

    return jnp.matmul(V, jnp.matmul(F, V.conj().T)), f, V, O, key

def Omega(N: int) -> jnp.array:
    return jp.sqrt(1/2) * jnp.block(
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

#--------------- Primal methods ---------------#
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

@jit
def weighted_unitary_on_fgstate(Gamma_mjr: jnp.array, h: jnp.array, theta: float):
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

    exp_p2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(theta * w))), jnp.conj(jnp.transpose(v)))
    exp_m2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(-theta * w))), jnp.conj(jnp.transpose(v)))

    return jnp.matmul(jnp.matmul(exp_p2h, Gamma_mjr), exp_m2h)

class PrimalParams():
    """
    Class to store parameters required to define the primal problem.
    """

    def __init__(self, N: int, d: int, key: jnp.array,
                 init_state_desc: str = "all zero"):
        """
        d: depth of circuit
        init_state_desc: description of the initial state wanted
        """
        self.N = N
        self.d = d
        self.generate_layer_hamiltonians(key)
        self.generate_target_hamiltonian(key)
        self.generate_init_state(init_state_desc)

    def generate_layer_hamiltonians(self, key: jnp.array):
        self.layer_hamiltonians = []

        for i in range(self.d):
            random_h, key = random_normal_hamiltonian_majorana(self.N, key)
            self.layer_hamiltonians.append(random_h/self.N)

        self.key_after_ham_gen = key

    def generate_target_hamiltonian(self, key: jnp.array):
        h_target, key = random_normal_hamiltonian_majorana(self.N, key)
        self.h_target = h_target/self.N

    def generate_init_state(self, init_state_desc: str = "all zero"):
        if init_state_desc == "all zero":
            I = jnp.identity(self.N, dtype = complex)

            self.Gamma_mjr_init = 0.5 * jnp.block(
            [[I      , 1j * I],
             [-1j * I, I     ]])

@partial(jit, static_argnums = (1,))
def circ_obj(theta: jnp.array, params: PrimalParams):
    Gamma_mjr = params.Gamma_mjr_init

    for i in range(params.d):
        Gamma_mjr = weighted_unitary_on_fgstate(Gamma_mjr,
                                                params.layer_hamiltonians[i],
                                                theta[i])

    return jnp.real(energy(Gamma_mjr, params.h_target))

@partial(jit, static_argnums = (1,))
def circ_grad(theta: jnp.array, params: PrimalParams):
    return grad(circ_obj, argnums = 0)(theta, params)

def optimize_circuit(theta_init: jnp.array, params: PrimalParams):
    bounds = scipy.optimize.Bounds(lb = 0.0, ub = 2 * np.pi)

    return optimize(np.array(theta_init),
                    params, circ_obj, circ_grad,
                    num_iters = 50, bounds = bounds)

def noise_on_fgstate_mc_sample(Gamma_mjr: jnp.array, p: float, key: jnp.array):
    """
    Parameters
    ----------
    Gamma_mjr: Correlation matrix (Majorana rep.)
    p: noise probability
    key: to generate random mc sample

    Returns
    -------
    Correlation matrix (Majorana rep.) of f.g.s. after one MC sampling of noise
    (on every mode).
    """
    N = Gamma_mjr.shape[0]//2
    gamma = covariance_from_corr_major(Gamma_mjr, N)

    # print('gamma = ', gamma)

    key, subkey = jax.random.split(key)
    mc_probs = jax.random.uniform(key, shape = (N,))

    # print(mc_probs)

    gamma_noisy = gamma

    for k in range(N):
        if mc_probs[k] <= p:
            # print('k = ', k)
            gamma_noisy = gamma_noisy.at[k, :].set(jnp.zeros(2 * N))
            gamma_noisy = gamma_noisy.at[:, k].set(jnp.zeros(2 * N))
            gamma_noisy = gamma_noisy.at[k + N, :].set(jnp.zeros(2 * N))
            gamma_noisy = gamma_noisy.at[:, k + N].set(jnp.zeros(2 * N))

    # print('gamma_noisy = ', gamma_noisy)

    Gamma_mjr_noisy = corr_major_from_covariance(gamma_noisy, N)

    return Gamma_mjr_noisy, key

#--------------- Dual methods ---------------#
class DualParams():
    def __init__(self, circ_params: PrimalParams, theta_opt: jnp.array, p: float,
                 scale: float = 1):
        self.circ_params = circ_params
        self.theta_opt = theta_opt
        self.p = p
        self.scale = scale

        self.total_num_dual_vars = self.circ_params.d + \
        (2*self.circ_params.N - 1) * self.circ_params.N * self.circ_params.d

@partial(jit, static_argnums = (1,2))
def unvec_and_process_dual_vars(dual_vars: jnp.array, d: int, N: int):
    utri_indices = jnp.triu_indices(2*N, 1)
    ltri_indices = (utri_indices[1], utri_indices[0])

    sigmas = jnp.zeros((2 * N, 2 * N, d))

    a_vars = dual_vars.at[:d].get()
    lambdas = jnp.log(1 + jnp.exp(a_vars))

    dual_vars_split = jnp.split(dual_vars.at[d:].get(), d)

    for i in range(d):
        sigma_layer = jnp.zeros((2*N, 2*N))
        sigma_layer = sigma_layer.at[utri_indices].set(dual_vars_split[i])
        sigma_layer = sigma_layer.at[ltri_indices].set(-dual_vars_split[i])
        sigmas = sigmas.at[:,:,i].set(sigma_layer)

    return lambdas, sigmas

@partial(jit, static_argnums = (2,))
def noisy_dual_layer(h_layer: jnp.array, sigma_layer: jnp.array, p: float):
    """
    h_layer: Generating Hamiltonian (majorana rep.) of the unitary layer of the
    circuit. Intended to be already scaled by circuit params.
    """
    sigma = noise_on_fghamiltonian(sigma_layer, p)
    sigma = unitary_on_fghamiltonian(sigma, -h_layer) # -ve because dual

    return sigma

@partial(jit, static_argnums = (1,))
def dual_obj(dual_vars: jnp.array, dual_params: DualParams):
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_target = dual_params.circ_params.h_target
    layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
    theta_opt = dual_params.theta_opt
    p = dual_params.p
    Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init

    lambdas, sigmas = unvec_and_process_dual_vars(dual_vars, d, N)

    cost = 0
    # log Tr exp terms
    for i in range(d):
        if i == d-1:
            hi = h_target + sigmas.at[:,:,i].get()
        else:
            hi = sigmas.at[:,:,i].get() - \
                 noisy_dual_layer(theta_opt[i+1] * layer_hamiltonians[i+1],
                                  sigmas.at[:,:,i+1].get(), p)

        cost += -lambdas[i] * jnp.log(trace_fgstate(-hi/lambdas[i]))

    # init. state term
    epsilon_1_dag_sigma1 = \
    noisy_dual_layer(theta_opt[0] * layer_hamiltonians[0], sigmas.at[:,:,0].get(), p)

    cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)

    # entropy term
    q = 1 - p
    q_powers = jnp.array([q**i for i in range(d)])

    entropy_bounds = N * p * jnp.log(2) * \
             jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])

    cost += jnp.dot(lambdas, entropy_bounds)

    return -dual_params.scale * jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad(dual_vars: jnp.array, dual_params: DualParams):
    return grad(dual_obj, argnums = 0)(dual_vars, dual_params)

def optimize_dual(dual_vars_init: jnp.array, dual_params: DualParams,
                  bnds: scipy.optimize.Bounds = None):

    return optimize(np.array(dual_vars_init),
                    dual_params, dual_obj, dual_grad,
                    num_iters = 250, bounds = bnds)

# @partial(jit, static_argnums = (2,))
def fd_dual_grad_at_index(dual_vars: jnp.array, i: int, dual_params: DualParams):
    delta = 1e-7
    dual_vars_plus = dual_vars.at[i].add(delta)
    dual_obj_plus = dual_obj(dual_vars_plus, dual_params)

    dual_vars_minus = dual_vars.at[i].add(-delta)
    dual_obj_minus = dual_obj(dual_vars_minus, dual_params)

    return (dual_obj_plus - dual_obj_minus)/(2 * delta)

def fd_dual_grad(dual_vars: jnp.array, dual_params: DualParams):
    dual_grad = jnp.zeros((len(dual_vars),))

    for i in range(len(dual_vars)):
        print(i)
        dual_grad = dual_grad.at[i].set(fd_dual_grad_at_index(dual_vars, i, dual_params))

    return dual_grad

def adam_optimize_dual(dual_vars_init: jnp.array, dual_params: DualParams,
                           alpha: float, num_steps: int):

    init, update, get_params = optimizers.adam(alpha)

    def step(t, opt_state):
        value = dual_obj(get_params(opt_state), dual_params)
        grads = dual_grad(get_params(opt_state), dual_params)
        opt_state = update(t, grads, opt_state)
        return value, opt_state

    opt_state = init(dual_vars_init)
    value_array = jnp.zeros(num_steps)

    for t in range(num_steps):
        if t%(num_steps//10) == 0:
            print("Step :", t)
        # print("Step :", t)
        value, opt_state = step(t, opt_state)
        value_array = value_array.at[t].set(value)

    return value_array, get_params(opt_state)

@jit
def trace_fgstate(parent_h: jnp.array):
    """
    Parameters
    ----------
    parent_h: Parent Hamiltonian of the f.g.s (Majorana rep.)

    Returns
    -------
    Trace of f.g.s.
    """
    N = parent_h.shape[0]//2
    w, v = jnp.linalg.eigh(1j * parent_h)

    positive_eigs = w[N:]

    return jnp.prod(jnp.exp(positive_eigs) + jnp.exp(-positive_eigs))

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

@partial(jit, static_argnums = (1,))
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

    # for k in range(N):
    #     s_prime_zeroed_out = s_prime.at[k, :].set(jnp.zeros(2 * N))
    #     s_prime_zeroed_out = s_prime_zeroed_out.at[:, k].set(jnp.zeros(2 * N))
    #     s_prime_zeroed_out = s_prime_zeroed_out.at[k + N, :].set(jnp.zeros(2 * N))
    #     s_prime_zeroed_out = s_prime_zeroed_out.at[:, k + N].set(jnp.zeros(2 * N))
    #
    #     s_prime = (1 - p) * s_prime + p * (s_prime_zeroed_out)
    return s_prime



# def noise_on_fgstate_mc_realisation(Gamma_mjr: jnp.array, key: jnp.array,
#                                     p: float, N: int):
#     """
#     Parameters
#     ----------
#     Gamma_mjr: Correlation matrix (Majorana rep.)
#     key: to generate random mc samples
#     p: noise probability
#     N: number of fermionic modes
#
#     Returns
#     -------
#     Correlation matrix (Majorana rep.) of f.g.s. after MC sim. of noise
#     """
#
#     s_prime = s
#
#     for k in range(N):
#         s_prime_zeroed_out = s_prime.at[k, :].set(jnp.zeros(2 * N))
#         s_prime_zeroed_out = s_prime_zeroed_out.at[:, k].set(jnp.zeros(2 * N))
#         s_prime_zeroed_out = s_prime_zeroed_out.at[k + N, :].set(jnp.zeros(2 * N))
#         s_prime_zeroed_out = s_prime_zeroed_out.at[:, k + N].set(jnp.zeros(2 * N))
#
#         s_prime = (1 - p) * s_prime + p * (s_prime_zeroed_out)
#
#     return s_prime
