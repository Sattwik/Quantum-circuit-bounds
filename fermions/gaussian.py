from typing import List, Tuple, Callable, Dict
from functools import partial

import numpy as np
import scipy
import networkx as nx
import tensornetwork as tn
tn.set_default_backend("jax")
import jax.numpy as jnp
import jax.scipy.linalg
from jax import jit, grad, vmap, value_and_grad
from jax.example_libraries import optimizers
import optax


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
             num_iters: int = 500, bounds = None, opt_method: str = "L-BFGS-B", tol_scale: float = 1.0):

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
                                'ftol': 2.220446049250313e-09 * tol_scale,
                                'gtol': 1e-05,
                                'eps': 1e-08,
                                'maxfun': 2 * num_iters,
                                'maxiter': num_iters,
                                'iprint': 98,
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

def optax_optimize(obj_fun: Callable, grad_fun: Callable,
                  vars_init: jnp.array, params, alpha: float, 
                  num_steps: int, method: str):

    if method == 'adam':
        optimizer = optax.adam(learning_rate=alpha)

    if method == 'amsgrad':
        optimizer = optax.amsgrad(learning_rate=alpha)

    if method == 'adabelief':
        optimizer = optax.adabelief(learning_rate=alpha)
    
    if method == "adagrad":
        optimizer = optax.adagrad(learning_rate=alpha)
    
    if method == "rmsprop":
        optimizer = optax.rmsprop(learning_rate=alpha)

    @jit
    def step(vars, opt_state):
        value = obj_fun(vars, params)
        grads = grad_fun(vars, params)

        updates, opt_state = optimizer.update(grads, opt_state, vars)
        vars = optax.apply_updates(vars, updates)

        return vars, opt_state, value

    opt_state = optimizer.init(vars_init)
    vars = vars_init
    value_array = jnp.zeros(num_steps)

    for t in range(num_steps):
        if t%(num_steps//100) == 0:
            print("Step :", t)
        # print("Step :", t)
        vars, opt_state, value = step(vars, opt_state)
        value_array = value_array.at[t].set(value)

    return value_array, vars

def subgrad_descent(obj_fun: Callable, grad_fun: Callable,
                  vars_init: jnp.array, params, alpha: float, 
                  num_steps: int, method: str):
    
    # @partial(jit, static_argnums = (1,))
    def step(vars, params):
        value = obj_fun(vars, params)
        grads = grad_fun(vars, params)

        # vars = vars - alpha * grads/jnp.linalg.norm(grads)
        vars = vars - alpha * grads/jnp.linalg.norm(grads)

        return vars, params, value
    
    vars = vars_init
    value_array = jnp.zeros(num_steps)

    for t in range(num_steps):
        if t%(num_steps//100) == 0:
            print("Step :", t)
        # print("Step :", t)
        vars, params, value = step(vars, params)
        value_array = value_array.at[t].set(value)

    return value_array, vars


# def adam_optimize(obj_fun: Callable, grad_fun: Callable,
#                   vars_init: jnp.array, params,
#                   alpha: float, num_steps: int):

#     init, update, get_params = optimizers.adam(alpha)

#     def step(t, opt_state):
#         value = obj_fun(get_params(opt_state), params)
#         grads = grad_fun(get_params(opt_state), params)
#         opt_state = update(t, grads, opt_state)
#         return value, opt_state

#     opt_state = init(vars_init)
#     value_array = jnp.zeros(num_steps)

#     for t in range(num_steps):
#         # if t%(num_steps//10) == 0:
#         #     print("Step :", t)
#         print("Step :", t)
#         value, opt_state = step(t, opt_state)
#         value_array = value_array.at[t].set(value)

#     return value_array, get_params(opt_state)

# def adagrad_optimize(obj_fun: Callable, grad_fun: Callable,
#                   vars_init: jnp.array, params,
#                   step_size: float, num_steps: int):

#     init, update, get_params = optimizers.adam(step_size)

#     def step(t, opt_state):
#         value = obj_fun(get_params(opt_state), params)
#         grads = grad_fun(get_params(opt_state), params)
#         opt_state = update(t, grads, opt_state)
#         return value, opt_state

#     opt_state = init(vars_init)
#     value_array = jnp.zeros(num_steps)

#     for t in range(num_steps):
#         # if t%(num_steps//10) == 0:
#         #     print("Step :", t)
#         print("Step :", t)
#         value, opt_state = step(t, opt_state)
#         value_array = value_array.at[t].set(value)

#     return value_array, get_params(opt_state)

#--------------------------------------------------#
#-------------------- FGS tools -------------------#
#--------------------------------------------------#

def random_NN_odd_bond_normal_hamiltonian_majorana(N: int, key: jnp.array):
    """
    Parameters
    ----------
    N: number of fermionic modes (assuming even)

    Returns
    -------
    A random NN odd bond f.g.h. (2N x 2N) in Majorana representation
    """
    key, subkey = jax.random.split(key)
    hxx = jax.random.normal(subkey, (N, N))
    hxx = hxx - hxx.T
    hxx = hxx.at[jnp.triu_indices(N, 1 + 1)].set(0.0)
    hxx = hxx.at[jnp.tril_indices(N, -1 - 1)].set(0.0)
    hxx = hxx.at[(list(range(1, N - 1, 2)), list(range(2, N, 2)))].set(0.0)
    hxx = hxx.at[(list(range(2, N, 2)), list(range(1, N - 1, 2)))].set(0.0)

    key, subkey = jax.random.split(key)
    hpp = jax.random.normal(subkey, (N, N))
    hpp = hpp - hpp.T
    hpp = hpp.at[jnp.triu_indices(N, 1 + 1)].set(0.0)
    hpp = hpp.at[jnp.tril_indices(N, -1 - 1)].set(0.0)
    hpp = hpp.at[(list(range(1, N - 1, 2)), list(range(2, N, 2)))].set(0.0)
    hpp = hpp.at[(list(range(2, N, 2)), list(range(1, N - 1, 2)))].set(0.0)

    key, subkey = jax.random.split(key)
    hxp = jax.random.normal(subkey, (N, N))
    hxp = hxp.at[jnp.triu_indices(N, 1 + 1)].set(0.0)
    hxp = hxp.at[jnp.tril_indices(N, -1 - 1)].set(0.0)
    hxp = hxp.at[(list(range(1, N - 1, 2)), list(range(2, N, 2)))].set(0.0)
    hxp = hxp.at[(list(range(2, N, 2)), list(range(1, N - 1, 2)))].set(0.0)

    h = jnp.block([[hxx, hxp],
                  [-hxp.T, hpp]])

    return h, key

def random_NN_even_bond_normal_hamiltonian_majorana(N: int, key: jnp.array):
    """
    Parameters
    ----------
    N: number of fermionic modes (assuming even)

    Returns
    -------
    A random NN odd bond f.g.h. (2N x 2N) in Majorana representation
    """
    key, subkey = jax.random.split(key)
    hxx = jax.random.normal(subkey, (N, N))
    hxx = hxx - hxx.T
    hxx = hxx.at[jnp.triu_indices(N, 1 + 1)].set(0.0)
    hxx = hxx.at[jnp.tril_indices(N, -1 - 1)].set(0.0)
    hxx = hxx.at[(list(range(0, N, 2)), list(range(1, N, 2)))].set(0.0)
    hxx = hxx.at[(list(range(1, N, 2)), list(range(0, N, 2)))].set(0.0)

    key, subkey = jax.random.split(key)
    hpp = jax.random.normal(subkey, (N, N))
    hpp = hpp - hpp.T
    hpp = hpp.at[jnp.triu_indices(N, 1 + 1)].set(0.0)
    hpp = hpp.at[jnp.tril_indices(N, -1 - 1)].set(0.0)
    hpp = hpp.at[(list(range(0, N, 2)), list(range(1, N, 2)))].set(0.0)
    hpp = hpp.at[(list(range(1, N, 2)), list(range(0, N, 2)))].set(0.0)

    key, subkey = jax.random.split(key)
    hxp = jax.random.normal(subkey, (N, N))
    hxp = hxp.at[jnp.triu_indices(N, 1 + 1)].set(0.0)
    hxp = hxp.at[jnp.tril_indices(N, -1 - 1)].set(0.0)
    hxp = hxp.at[(list(range(0, N, 2)), list(range(1, N, 2)))].set(0.0)
    hxp = hxp.at[(list(range(1, N, 2)), list(range(0, N, 2)))].set(0.0)

    h = jnp.block([[hxx, hxp],
                  [-hxp.T, hpp]])

    return h, key

def random_k_local_normal_hamiltonian_majorana(N: int, k: int, key: jnp.array):
    """
    Parameters
    ----------
    N: number of fermionic modes
    k < N: range of interactions

    Returns
    -------
    A random k-local f.g.h. (2N x 2N) in Majorana representation
    """
    key, subkey = jax.random.split(key)
    hxx = jax.random.normal(subkey, (N, N))
    hxx = hxx - hxx.T
    hxx = hxx.at[jnp.triu_indices(N, k + 1)].set(0.0)
    hxx = hxx.at[jnp.tril_indices(N, -k - 1)].set(0.0)

    key, subkey = jax.random.split(key)
    hpp = jax.random.normal(subkey, (N, N))
    hpp = hpp - hpp.T
    hpp = hpp.at[jnp.triu_indices(N, k + 1)].set(0.0)
    hpp = hpp.at[jnp.tril_indices(N, -k - 1)].set(0.0)

    key, subkey = jax.random.split(key)
    hxp = jax.random.normal(subkey, (N, N))
    hxp = hxp.at[jnp.triu_indices(N, k + 1)].set(0.0)
    hxp = hxp.at[jnp.tril_indices(N, -k - 1)].set(0.0)

    h = jnp.block([[hxx, hxp],
                  [-hxp.T, hpp]])

    return h, key

def k_local_hamiltonian_indicators(N: int, k: int):
    """
    Parameters
    ----------
    N: number of fermionic modes
    k < N: range of interactions

    Returns
    -------
    A matrix with ones where a k-nearest neighbours interaction
    Hamiltonian can have elements
    """

    hxx = jnp.ones((N, N))
    hxx = hxx.at[jnp.diag_indices(N)].set(0.0)
    hxx = hxx.at[jnp.triu_indices(N, k + 1)].set(0.0)
    hxx = hxx.at[jnp.tril_indices(N, -k - 1)].set(0.0)

    hpp = jnp.ones((N, N))
    hpp = hpp.at[jnp.diag_indices(N)].set(0.0)
    hpp = hpp.at[jnp.triu_indices(N, k + 1)].set(0.0)
    hpp = hpp.at[jnp.tril_indices(N, -k - 1)].set(0.0)

    hxp = jnp.ones((N, N))
    hxp = hxp.at[jnp.triu_indices(N, k + 1)].set(0.0)
    hxp = hxp.at[jnp.tril_indices(N, -k - 1)].set(0.0)

    h = jnp.block([[hxx, hxp],
                  [hxp.T, hpp]])

    return h

def ssh(N: int, t1: float, t2: float):
    diag_entries = jnp.zeros(N-1)
    diag_entries = diag_entries.at[::2].set(t1)
    diag_entries = diag_entries.at[1::2].set(t2) 

    zer = jnp.zeros((N,N))
    Haa = zer

    k = 1
    block_upper_indices_to_zero = jnp.triu_indices(N, k + 1)

    ones_N = jnp.ones((N, N))
    upper_ones_N = jnp.triu(ones_N, 1)
    upper_diag_indices = jnp.where(upper_ones_N.at[block_upper_indices_to_zero].set(0.0) == 1)
    
    Haa = Haa.at[upper_diag_indices].set(diag_entries)
    Haa = Haa + Haa.conj().T

    H = jnp.block([[-Haa.conj(), zer], [zer,  Haa]])
    Ome = Omega(N)
    h = -1j * jnp.matmul(jnp.matmul(Ome, H), Ome.conj().T)

    return h

def corr_major_gs(h: jnp.array):
    N = h.shape[0]//2
    Ome = Omega(N)

    H = jnp.matmul(jnp.matmul(Ome.conj().T, 1j * h), Ome)
    w, v = jnp.linalg.eigh(H)

    I = jnp.identity(N, dtype = complex)
    zer = jnp.zeros((N,N), dtype = complex)

    Gamma_0 = jnp.block([[zer, zer], [zer,  I]])

    Gamma = jnp.matmul(v, jnp.matmul(Gamma_0, v.conj().T))

    Gamma_mjr = jnp.matmul(jnp.matmul(Ome, Gamma), Ome.conj().T)

    return Gamma_mjr

def gse(h: jnp.array):
    N = h.shape[0]//2
    w, v = jnp.linalg.eigh(1j * h)

    gse = jnp.sum(w[:N])

    return gse

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

def corr_major_product(Gamma_mjr_1: jnp.array, Gamma_mjr_2: jnp.array):

    """
    Return the correlation matrix of the product of f.g.ses corresponding to
    the input correlation mats.
    """
    N = Gamma_mjr_1.shape[0]//2

    gamma_1 = covariance_from_corr_major(Gamma_mjr_1)
    gamma_2 = covariance_from_corr_major(Gamma_mjr_2)
    I = jnp.identity(2 * N, dtype = complex)

    denom = I - jnp.matmul(gamma_2, gamma_1)

    gamma = I - jnp.matmul(jnp.matmul((I - 1j * gamma_1),
                                       jnp.linalg.inv(denom)),
                           (I - 1j * gamma_2))

    Gamma_mjr = corr_major_from_covariance(gamma)

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

    def __init__(self, N: int, d: int, local_d: int,
                 key: jnp.array,
                 init_state_desc: str = "GS", k: int = 1, mode: str = 'adjacent', 
                 h_mode: str = "ssh"):
        """
        d: depth of circuit
        init_state_desc: description of the initial state wanted

        Assuming (d - local_d) is even
        """
        self.N = N
        self.d = d
        self.local_d = local_d
        self.k = k

        assert((d - local_d)%2 == 0)

        self.generate_layer_hamiltonians(key, mode)
        self.generate_parent_hamiltonian(h_mode)
        self.generate_init_state(init_state_desc)

    def generate_layer_hamiltonians(self, key: jnp.array, mode = 'adjacent'):
        if mode == "NN":
            print("NN circuit")
            self.layer_hamiltonians = []
            for i in range(self.local_d//2):
                random_even_h, key = random_NN_even_bond_normal_hamiltonian_majorana(self.N, key)
                self.layer_hamiltonians.append(random_even_h)

                random_odd_h, key = random_NN_odd_bond_normal_hamiltonian_majorana(self.N, key)
                self.layer_hamiltonians.append(random_odd_h)

            h_list = []
            for i in range((self.d - self.local_d)//4):
                random_even_h, key = random_NN_even_bond_normal_hamiltonian_majorana(self.N, key)
                self.layer_hamiltonians.append(random_even_h)
                h_list.append(random_even_h)

                random_odd_h, key = random_NN_odd_bond_normal_hamiltonian_majorana(self.N, key)
                self.layer_hamiltonians.append(random_odd_h)
                h_list.append(random_odd_h)

            for i in range((self.d - self.local_d)//4):
                self.layer_hamiltonians.append(-h_list[::-1][i])

        elif mode == "NN_k1":
            print("NN circuit, k = 1 parent H")
            self.layer_hamiltonians = []
            for i in range(self.local_d):
                random_even_h, key = random_NN_even_bond_normal_hamiltonian_majorana(self.N, key)
                self.layer_hamiltonians.append(random_even_h)

            h_list = []
            for i in range((self.d - self.local_d)//2):
                if i%2 == 0:
                    random_even_h, key = random_NN_even_bond_normal_hamiltonian_majorana(self.N, key)
                    self.layer_hamiltonians.append(random_even_h)
                    h_list.append(random_even_h)
                else:
                    random_odd_h, key = random_NN_odd_bond_normal_hamiltonian_majorana(self.N, key)
                    self.layer_hamiltonians.append(random_odd_h)
                    h_list.append(random_odd_h)

            for i in range((self.d - self.local_d)//2):
                self.layer_hamiltonians.append(-h_list[::-1][i])

        elif mode == "ssh":
            print("NN circuit, ssh parent H")
            # d_local = 0

            self.layer_hamiltonians = []
            h_list = []
            for i in range((self.d)//2):
                if i%2 == 0:
                    random_even_h, key = random_NN_even_bond_normal_hamiltonian_majorana(self.N, key)
                    self.layer_hamiltonians.append(random_even_h)
                    h_list.append(random_even_h)
                else:
                    random_odd_h, key = random_NN_odd_bond_normal_hamiltonian_majorana(self.N, key)
                    self.layer_hamiltonians.append(random_odd_h)
                    h_list.append(random_odd_h)

            for i in range((self.d)//2):
                self.layer_hamiltonians.append(-h_list[::-1][i])
        
        else:
            self.layer_hamiltonians = []

            for i in range(self.local_d):
                random_local_h, key = random_k_local_normal_hamiltonian_majorana(self.N, self.k, key)
                self.layer_hamiltonians.append(random_local_h/self.N)

            if mode == 'adjacent':
                for i in range((self.d - self.local_d)//2):
                    self.layer_hamiltonians.append(random_local_h/self.N)
                    self.layer_hamiltonians.append(-random_local_h/self.N)
            elif mode == 'block':
                h_list = []
                for i in range((self.d - self.local_d)//2):
                    random_local_h, key = random_k_local_normal_hamiltonian_majorana(self.N, self.k, key)
                    self.layer_hamiltonians.append(random_local_h/self.N)
                    h_list.append(random_local_h)

                for i in range((self.d - self.local_d)//2):
                    self.layer_hamiltonians.append(-h_list[::-1][i]/self.N)

        self.layer_hamiltonians = jnp.array(self.layer_hamiltonians)
        self.key_after_ham_gen = key

    def generate_parent_hamiltonian(self, h_mode):
        if h_mode == "ssh":
            self.h_parent = ssh(self.N, t1 = 1, t2 = 0.3)

        else:
            Ome = Omega(self.N)
            # epsilon = jnp.arange(start = self.N, stop = 0, step = -1)
            epsilon = jnp.linspace(start = 1, stop = 0, num = self.N)
            D = jnp.diag(jnp.concatenate((-epsilon, epsilon)))
            d_parent = -1j * jnp.matmul(Ome, jnp.matmul(D, Ome.conj().T))
            h_parent = d_parent

            for i in range(self.local_d):
                h_parent = unitary_on_fghamiltonian(h_parent, self.layer_hamiltonians.at[i,:,:].get())
            self.h_parent = h_parent

    def generate_init_state(self, init_state_desc: str = "all zero"):
        if init_state_desc == "all zero":
            I = jnp.identity(self.N, dtype = complex)

            self.Gamma_mjr_init = 0.5 * jnp.block(
            [[I      , 1j * I],
             [-1j * I, I     ]])
        elif init_state_desc == "GS":
            self.Gamma_mjr_init = corr_major_gs(self.h_parent)


def energy_after_circuit(params: PrimalParams):
    Gamma_mjr = params.Gamma_mjr_init

    for i in range(params.local_d):
        Gamma_mjr = unitary_on_fgstate(Gamma_mjr, params.layer_hamiltonians.at[i, :, :].get())

    return jnp.real(energy(Gamma_mjr, params.h_parent))

def noise_on_kth_mode(k: int, args: Tuple):
    Gamma_mjr, p = args
    N = Gamma_mjr.shape[0]//2
    gamma = covariance_from_corr_major(Gamma_mjr)

    gamma_replaced = gamma.at[k, :].set(jnp.zeros(2 * N))
    gamma_replaced = gamma_replaced.at[:, k].set(jnp.zeros(2 * N))
    gamma_replaced = gamma_replaced.at[k + N, :].set(jnp.zeros(2 * N))
    gamma_replaced = gamma_replaced.at[:, k + N].set(jnp.zeros(2 * N))

    gamma_noisy = p * gamma_replaced + (1 - p) * gamma
    Gamma_mjr_noisy = corr_major_from_covariance(gamma_noisy)

    return (Gamma_mjr_noisy, p)

def noisy_primal_ith_layer(i: int, args: Tuple):
    Gamma_mjr, layer_hamiltonians, N, p = args
    Gamma_mjr = unitary_on_fgstate(Gamma_mjr, layer_hamiltonians.at[i, :, :].get())
    init_args = (Gamma_mjr, p)
    Gamma_mjr, _ = jax.lax.fori_loop(0, N, noise_on_kth_mode, init_args)
    return (Gamma_mjr, layer_hamiltonians, N, p)

def noisy_primal(params:PrimalParams, p: float):
    Gamma_mjr = params.Gamma_mjr_init
    layer_hamiltonians = params.layer_hamiltonians
    N = params.N
    init_args = (Gamma_mjr, layer_hamiltonians, N, p)
    Gamma_mjr, _, _, _ = jax.lax.fori_loop(0, params.d, noisy_primal_ith_layer, init_args)

    return jnp.real(energy(Gamma_mjr, params.h_parent))

def entropy_parent(lmbda: jnp.array, h_parent: jnp.array):
    beta = 1/lmbda.at[0].get()
    Gamma_mjr = corr_major_from_parenth(h_parent * beta)

    entropy = jnp.log(trace_fgstate(-h_parent * beta)) \
            + beta * energy(Gamma_mjr, h_parent)

    return entropy

# for i in range(params.d):
#     Gamma_mjr = unitary_on_fgstate(Gamma_mjr, params.layer_hamiltonians.at[i, :, :].get())
#     init_args = (Gamma_mjr, p)
#     Gamma_mjr, _ = jax.lax.fori_loop(0, params.N, noise_on_kth_mode, init_args)


# def average_Gamma_mjr_noisy(Gamma_mjr: jnp.array, p: float):
#     N = Gamma_mjr.shape[0]//2
#     gamma = covariance_from_corr_major(Gamma_mjr)
#     gamma_noisy = gamma
#
#     for k in range(N):
#         gamma_replaced = gamma_noisy.at[k, :].set(jnp.zeros(2 * N))
#         gamma_replaced = gamma_replaced.at[:, k].set(jnp.zeros(2 * N))
#         gamma_replaced = gamma_replaced.at[k + N, :].set(jnp.zeros(2 * N))
#         gamma_replaced = gamma_replaced.at[:, k + N].set(jnp.zeros(2 * N))
#
#         gamma_noisy = p * gamma_replaced + (1 - p) * gamma_noisy
#
#     Gamma_mjr_noisy = corr_major_from_covariance(gamma_noisy)
#     return Gamma_mjr_noisy
#
# def noise_on_fgstate_mc_sample(Gamma_mjr: jnp.array, p: float, key: jnp.array):
#     """
#     Parameters
#     ----------
#     Gamma_mjr: Correlation matrix (Majorana rep.)
#     p: noise probability
#     key: to generate random mc sample
#
#     Returns
#     -------
#     Correlation matrix (Majorana rep.) of f.g.s. after one MC sampling of noise
#     (on every mode).
#     """
#     N = Gamma_mjr.shape[0]//2
#     gamma = covariance_from_corr_major(Gamma_mjr)
#
#     # print('gamma = ', gamma)
#
#     key, subkey = jax.random.split(key)
#
#     # print('subkey = ', subkey)
#
#     mc_probs = jax.random.uniform(subkey, shape = (N,))
#
#     # print(mc_probs)
#
#     # print(mc_probs)
#
#     gamma_noisy = gamma
#
#     for k in range(N):
#         if mc_probs.at[k].get() <= p:
#             gamma_noisy = gamma_noisy.at[k, :].set(jnp.zeros(2 * N))
#             gamma_noisy = gamma_noisy.at[:, k].set(jnp.zeros(2 * N))
#             gamma_noisy = gamma_noisy.at[k + N, :].set(jnp.zeros(2 * N))
#             gamma_noisy = gamma_noisy.at[:, k + N].set(jnp.zeros(2 * N))
#
#     # print('gamma_noisy = ', gamma_noisy)
#
#     Gamma_mjr_noisy = corr_major_from_covariance(gamma_noisy)
#
#     return Gamma_mjr_noisy, key

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

        block_upper_indices_to_zero = jnp.triu_indices(self.circ_params.N, self.k_dual + 1)
        block_lower_indices_to_zero = jnp.tril_indices(self.circ_params.N, -self.k_dual - 1)

        ones_N = jnp.ones((self.circ_params.N, self.circ_params.N))
        upper_ones_N = jnp.triu(ones_N, 1)
        lower_ones_N = jnp.tril(ones_N, -1)

        block_upper_band = upper_ones_N.at[block_upper_indices_to_zero].set(0.0)
        block_lower_band = lower_ones_N.at[block_lower_indices_to_zero].set(0.0)
        block_band = ones_N.at[block_upper_indices_to_zero].set(0.0)
        block_band = block_band.at[block_lower_indices_to_zero].set(0.0)

        self.block_upper_band_indices = jnp.where(block_upper_band == 1)
        self.block_lower_band_indices = jnp.where(block_lower_band == 1)
        self.block_band_indices = jnp.where(block_band == 1)

        num_vars_xx = self.block_upper_band_indices[0].shape[0]
        num_vars_pp = num_vars_xx
        num_vars_xp = self.block_band_indices[0].shape[0]

        self.num_sigma_vars_layer = num_vars_xx + num_vars_pp + num_vars_xp

        self.total_num_dual_vars = self.circ_params.d + \
        self.circ_params.d * self.num_sigma_vars_layer

        # self.total_num_dual_vars = self.circ_params.d + \
        # (2*self.circ_params.N - 1) * self.circ_params.N * self.circ_params.d

def sigmas_to_vec(sigmas: jnp.array, dual_params:DualParams):

    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    block_upper_band_indices = dual_params.block_upper_band_indices
    block_lower_band_indices = dual_params.block_lower_band_indices
    block_band_indices = dual_params.block_band_indices

    vars = jnp.zeros((dual_params.circ_params.d * dual_params.num_sigma_vars_layer,))

    init_args = (vars, sigmas, block_upper_band_indices, block_lower_band_indices, block_band_indices)

    vars, _, _, _, _ = jax.lax.fori_loop(0, d, vec_layer_i, init_args)

    return vars

def vec_layer_i(i: int, args: Tuple):

    vars, sigmas, block_upper_band_indices, block_lower_band_indices, block_band_indices = args

    # number of variables in blocks of the full sigma dual var
    l_xx = block_upper_band_indices[0].shape[0]
    l_pp = l_xx
    l_xp = block_band_indices[0].shape[0]

    # total_num_vars
    l = l_xp + l_xx + l_pp

    sigma_layer = sigmas.at[:,:,i].get()

    N = sigmas.shape[0]//2

    sigma_layer_xx = sigma_layer.at[:N, :N].get()
    sigma_layer_xp = sigma_layer.at[:N, N:].get()
    sigma_layer_pp = sigma_layer.at[N:, N:].get()

    sigma_slice_xx = sigma_layer_xx.at[block_upper_band_indices].get()
    sigma_slice_pp = sigma_layer_pp.at[block_upper_band_indices].get()
    sigma_slice_xp = sigma_layer_xp.at[block_band_indices].get()

    sigma_slice = jnp.concatenate((sigma_slice_xx, sigma_slice_pp, sigma_slice_xp))

    vars = jax.lax.dynamic_update_slice_in_dim(vars, sigma_slice, i * l, axis = 0)

    return (vars, sigmas, block_upper_band_indices, block_lower_band_indices, block_band_indices)

def unvec_layer_i(i: int, args: Tuple):
    sigmas, sigma_vars, \
    block_upper_band_indices, block_lower_band_indices, block_band_indices, \
    zeros_N = args

    # number of variables in blocks of the full sigma dual var
    l_xx = block_upper_band_indices[0].shape[0]
    l_pp = l_xx
    l_xp = block_band_indices[0].shape[0]

    # total_num_vars
    l = l_xp + l_xx + l_pp

    # slicing the input dual vars vector
    sigma_slice = jax.lax.dynamic_slice_in_dim(sigma_vars, i * l, l)
    sigma_slice_xx = sigma_slice.at[:l_xx].get()
    sigma_slice_pp = sigma_slice.at[l_xx : l_xx + l_pp].get()
    sigma_slice_xp = sigma_slice.at[l_xx + l_pp:].get()

    # making block matrices
    sigma_layer_xx = zeros_N.at[block_upper_band_indices].set(sigma_slice_xx)
    sigma_layer_xx = sigma_layer_xx - sigma_layer_xx.T
    # sigma_layer_xx = sigma_layer_xx.at[block_lower_band_indices].set(-sigma_slice_xx)

    sigma_layer_pp = zeros_N.at[block_upper_band_indices].set(sigma_slice_pp)
    sigma_layer_pp = sigma_layer_pp - sigma_layer_pp.T
    # sigma_layer_pp = sigma_layer_pp.at[block_lower_band_indices].set(-sigma_slice_pp)

    sigma_layer_xp = zeros_N.at[block_band_indices].set(sigma_slice_xp)

    # putting blocks together
    sigma_layer = jnp.block([[sigma_layer_xx, sigma_layer_xp],
                             [-sigma_layer_xp.T, sigma_layer_pp]])

    # putting layer var into composite structure
    sigmas = sigmas.at[:,:,i].set(sigma_layer)

    return (sigmas, sigma_vars,
            block_upper_band_indices, block_lower_band_indices,
            block_band_indices, zeros_N)

@partial(jit, static_argnums = (1,))
def unvec_and_process_dual_vars(dual_vars: jnp.array, dual_params: DualParams):
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    block_upper_band_indices = dual_params.block_upper_band_indices
    block_lower_band_indices = dual_params.block_lower_band_indices
    block_band_indices = dual_params.block_band_indices

    sigmas = jnp.zeros((2 * N, 2 * N, d))
    zeros_N = jnp.zeros((N, N))

    a_vars = dual_vars.at[:d].get()
    lambdas = dual_params.lambda_lower_bounds + jnp.log(1 + jnp.exp(a_vars))
    # lambdas = jnp.ones(a_vars.shape) + jnp.log(1 + jnp.exp(a_vars))

    sigma_vars = dual_vars.at[d:].get()
    init_args = (sigmas, sigma_vars,
                 block_upper_band_indices, block_lower_band_indices,
                 block_band_indices, zeros_N)
    sigmas, _, _, _, _, _ = jax.lax.fori_loop(0, d, unvec_layer_i, init_args)

    return lambdas, sigmas

# @partial(jit, static_argnums = (1,))
# def unvec_and_process_dual_vars_direct_lambda(dual_vars: jnp.array, dual_params: DualParams):
#     N = dual_params.circ_params.N
#     d = dual_params.circ_params.d
#     block_upper_band_indices = dual_params.block_upper_band_indices
#     block_lower_band_indices = dual_params.block_lower_band_indices
#     block_band_indices = dual_params.block_band_indices
#
#     sigmas = jnp.zeros((2 * N, 2 * N, d))
#     zeros_N = jnp.zeros((N, N))
#
#     lambdas = dual_vars.at[:d].get()
#     # lambdas = dual_params.lambda_lower_bounds + jnp.log(1 + jnp.exp(a_vars))
#     # lambdas = jnp.ones(a_vars.shape) + jnp.log(1 + jnp.exp(a_vars))
#
#     sigma_vars = dual_vars.at[d:].get()
#     init_args = (sigmas, sigma_vars,
#                  block_upper_band_indices, block_lower_band_indices,
#                  block_band_indices, zeros_N)
#     sigmas, _, _, _, _, _ = jax.lax.fori_loop(0, d, unvec_layer_i, init_args)
#
#     return lambdas, sigmas

# @partial(jit, static_argnums = (2,))
@jit
def noisy_dual_layer(h_layer: jnp.array, sigma_layer: jnp.array, p: float):
    """
    h_layer: Generating Hamiltonian (majorana rep.) of the unitary layer of the
    circuit. Intended to be already scaled by circuit params.
    """
    sigma = noise_on_fghamiltonian(sigma_layer, p)
    sigma = unitary_on_fghamiltonian(sigma, -h_layer) # -ve because dual

    return sigma

# def dual_free_energy_ith_term(i: int, args: Tuple):
#
#     lambdas, sigmas, layer_hamiltonians, p, cost = args
#
#     hi = sigmas.at[:,:,i].get() - \
#          noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(),
#                           sigmas.at[:,:,i+1].get(), p)
#
#     cost += -lambdas.at[i].get() * jnp.log(trace_fgstate(-hi/lambdas.at[i].get()))
#
#     return (lambdas, sigmas, layer_hamiltonians, p, cost)

def dual_free_energy_ith_term(i: int, args: Tuple):

    lambdas, sigmas, layer_hamiltonians, p, cost = args

    hi = sigmas.at[:,:,i].get() - \
         noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(),
                          sigmas.at[:,:,i+1].get(), p)

    cost += -lambdas.at[i].get() * log_trace_fgstate(-hi/lambdas.at[i].get())

    return (lambdas, sigmas, layer_hamiltonians, p, cost)

# def dual_free_energy_ith_term_1q(i: int, args: Tuple):
#
#     lambdas, sigmas, layer_hamiltonians, p, cost = args
#
#     hi = sigmas.at[:,:,i].get() - \
#          noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(),
#                           sigmas.at[:,:,i+1].get(), p)
#
#     cost += -lambdas.at[i].get() * jnp.log(trace_fgstate_1q(-hi/lambdas.at[i].get()))
#
#     return (lambdas, sigmas, layer_hamiltonians, p, cost)

# @partial(jit, static_argnums = (1,))
# def dual_obj(dual_vars: jnp.array, dual_params: DualParams):
#     N = dual_params.circ_params.N
#     d = dual_params.circ_params.d
#     h_parent = dual_params.circ_params.h_parent
#     layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
#     p = dual_params.p
#     Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init
#
#     # !!
#     lambdas, sigmas = unvec_and_process_dual_vars(dual_vars, dual_params)
#
#     cost = 0
#     # log Tr exp terms
#
#     # first d - 1 layers
#     # !!
#     init_args = (lambdas, sigmas, layer_hamiltonians, p, cost)
#     _, _, _, _, cost = jax.lax.fori_loop(0, d - 1, dual_free_energy_ith_term, init_args)
#
#     # last layer
#     # !!
#     hi = h_parent + sigmas.at[:,:,d-1].get()
#     cost += -lambdas.at[d-1].get() * jnp.log(trace_fgstate(-hi/lambdas[d-1]))
#
#     # init. state term
#     # !!
#     epsilon_1_dag_sigma1 = \
#     noisy_dual_layer(layer_hamiltonians.at[0, :, :].get(), sigmas.at[:,:,0].get(), p)
#
#     cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)
#
#     # entropy term
#     # !!
#     q = 1 - p
#     q_powers = jnp.array([q**i for i in range(d)])
#
#     entropy_bounds = N * p * jnp.log(2) * \
#              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
#
#     # !!
#     cost += jnp.dot(lambdas, entropy_bounds)
#
#     return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_obj(dual_vars: jnp.array, dual_params: DualParams):
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_parent = dual_params.circ_params.h_parent
    layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
    p = dual_params.p
    Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init

    # !!
    lambdas, sigmas = unvec_and_process_dual_vars(dual_vars, dual_params)

    cost = 0
    # log Tr exp terms

    # first d - 1 layers
    # !!
    init_args = (lambdas, sigmas, layer_hamiltonians, p, cost)
    _, _, _, _, cost = jax.lax.fori_loop(0, d - 1, dual_free_energy_ith_term, init_args)

    # last layer
    # !!
    hi = h_parent + sigmas.at[:,:,d-1].get()
    cost += -lambdas.at[d-1].get() * log_trace_fgstate(-hi/lambdas[d-1])

    # init. state term
    # !!
    epsilon_1_dag_sigma1 = \
    noisy_dual_layer(layer_hamiltonians.at[0, :, :].get(), sigmas.at[:,:,0].get(), p)

    cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)

    # entropy term
    # !!
    q = 1 - p
    q_powers = jnp.array([q**i for i in range(d)])

    entropy_bounds = N * p * jnp.log(2) * \
             jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])

    # !!
    cost += jnp.dot(lambdas, entropy_bounds)

    return -jnp.real(cost)

# @partial(jit, static_argnums = (1,))
# def dual_obj_direct_lambda(dual_vars: jnp.array, dual_params: DualParams):
#     N = dual_params.circ_params.N
#     d = dual_params.circ_params.d
#     h_parent = dual_params.circ_params.h_parent
#     layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
#     p = dual_params.p
#     Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init
#
#     # !!
#     lambdas, sigmas = unvec_and_process_dual_vars_direct_lambda(dual_vars, dual_params)
#
#     cost = 0
#     # log Tr exp terms
#
#     # first d - 1 layers
#     # !!
#     init_args = (lambdas, sigmas, layer_hamiltonians, p, cost)
#     _, _, _, _, cost = jax.lax.fori_loop(0, d - 1, dual_free_energy_ith_term, init_args)
#
#     # last layer
#     # !!
#     hi = h_parent + sigmas.at[:,:,d-1].get()
#     cost += -lambdas.at[d-1].get() * jnp.log(trace_fgstate(-hi/lambdas[d-1]))
#
#     # init. state term
#     # !!
#     epsilon_1_dag_sigma1 = \
#     noisy_dual_layer(layer_hamiltonians.at[0, :, :].get(), sigmas.at[:,:,0].get(), p)
#
#     cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)
#
#     # entropy term
#     # !!
#     q = 1 - p
#     q_powers = jnp.array([q**i for i in range(d)])
#
#     entropy_bounds = N * p * jnp.log(2) * \
#              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
#
#     # !!
#     cost += jnp.dot(lambdas, entropy_bounds)
#
#     return -jnp.real(cost)

# def binary_entropy(p: float):
#     return -p * jnp.log(p) -(1-p) * jnp.log(1-p)
#
# @partial(jit, static_argnums = (1,))
# def dual_obj_1q_test(dual_vars: jnp.array, dual_params: DualParams):
#     N = dual_params.circ_params.N
#     d = dual_params.circ_params.d
#     h_parent = dual_params.circ_params.h_parent
#     layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
#     p = dual_params.p
#     Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init
#
#     lambdas, sigmas = unvec_and_process_dual_vars(dual_vars, dual_params)
#
#     cost = 0
#     # log Tr exp terms
#
#     # first d - 1 layers
#     init_args = (lambdas, sigmas, layer_hamiltonians, p, cost)
#     _, _, _, _, cost = jax.lax.fori_loop(0, d - 1, dual_free_energy_ith_term_1q, init_args)
#
#     # last layer
#     hi = h_parent + sigmas.at[:,:,d-1].get()
#     cost += -lambdas.at[d-1].get() * jnp.log(trace_fgstate_1q(-hi/lambdas[d-1]))
#
#     # init. state term
#     epsilon_1_dag_sigma1 = \
#     noisy_dual_layer(layer_hamiltonians.at[0, :, :].get(), sigmas.at[:,:,0].get(), p)
#
#     cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)
#
#     # entropy term
#     q = 1 - p
#     q_powers = jnp.array([q**i for i in range(d)])
#
#     entropy_bounds = N * p * jnp.log(2) * \
#              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
#
#     entropy_bounds = jnp.ones(entropy_bounds.shape) * binary_entropy(p/2)
#
#     cost += jnp.dot(lambdas, entropy_bounds)
#
#     return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad(dual_vars: jnp.array, dual_params: DualParams):
    return grad(dual_obj, argnums = 0)(dual_vars, dual_params)

# @partial(jit, static_argnums = (1,))
# def dual_grad_direct_lambda(dual_vars: jnp.array, dual_params: DualParams):
#     return grad(dual_obj_direct_lambda, argnums = 0)(dual_vars, dual_params)

# @partial(jit, static_argnums = (1,))
# def dual_grad_1q_test(dual_vars: jnp.array, dual_params: DualParams):
    # return grad(dual_obj_1q_test, argnums = 0)(dual_vars, dual_params)

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

# def fd_dual_grad_at_index_direct_lambda(dual_vars: jnp.array, i: int, dual_params: DualParams):
#     delta = 1e-7
#     dual_vars_plus = dual_vars.at[i].add(delta)
#     dual_obj_plus = dual_obj_direct_lambda(dual_vars_plus, dual_params)
#
#     dual_vars_minus = dual_vars.at[i].add(-delta)
#     dual_obj_minus = dual_obj_direct_lambda(dual_vars_minus, dual_params)
#
#     return (dual_obj_plus - dual_obj_minus)/(2 * delta)
#
# def fd_dual_grad_direct_lambda(dual_vars: jnp.array, dual_params: DualParams):
#     dual_grad = jnp.zeros((len(dual_vars),))
#
#     for i in range(len(dual_vars)):
#         print(i)
#         dual_grad = dual_grad.at[i].set(fd_dual_grad_at_index_direct_lambda(dual_vars, i, dual_params))
#
#     return dual_grad

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

    # print(w)

    positive_eigs = w[N:]

    # print(positive_eigs)

    return jnp.prod(jnp.exp(positive_eigs) + jnp.exp(-positive_eigs))

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

# @jit
# def trace_fgstate_1q(parent_h: jnp.array):
#     """
#     Parameters
#     ----------
#     parent_h: Parent Hamiltonian of the f.g.s (Majorana rep.)
#
#     Returns
#     -------
#     Trace of f.g.s.
#     """
#     w, v = jnp.linalg.eigh(1j * parent_h)
#
#     # positive_eigs = w[N:]
#     return jnp.sum(jnp.exp(w))

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
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_parent = dual_params.circ_params.h_parent
    p = dual_params.p

    q = 1 - p
    q_powers = jnp.array([q**i for i in range(d)])
    entropy_bounds = N * p * jnp.log(2) * \
             jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
    Sd = entropy_bounds.at[-1].get()

    a = dual_vars.at[0].get()
    lmbda = dual_params.lambda_lower_bounds.at[-1].get() + jnp.log(1 + jnp.exp(a))
    # lmbda = 1 + jnp.log(1 + jnp.exp(a))

    # cost = -lmbda * jnp.log(trace_fgstate(-h_parent/lmbda)) + lmbda * Sd
    cost = -lmbda * log_trace_fgstate(-h_parent/lmbda) + lmbda * Sd

    cost += jnp.dot(dual_params.lambda_lower_bounds.at[:-1].get(),
                    entropy_bounds.at[:-1].get() - N * np.log(2))

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad_no_channel(dual_vars: jnp.array, dual_params: DualParams):
    return grad(dual_obj_no_channel, argnums = 0)(dual_vars, dual_params)

#--------------------------------------------------#
#------- Sigma projection, scaling analysis -------#
#--------------------------------------------------#

def cond_fun(args):
    i, _, _, _, _ = args
    return i >= 0

def set_ith_sigma_projected(args):
    i, sigmas, layer_hamiltonians, p, proj = args

    epsilon_dag_sigma_i_plus_1 = \
    noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(),
                     sigmas.at[:,:,i+1].get(), p)

    sigmas = sigmas.at[:, :, i].set(jnp.real(proj * jnp.real(epsilon_dag_sigma_i_plus_1)))

    i = i - 1

    return (i, sigmas, layer_hamiltonians, p, proj)

def set_all_sigmas(dual_params: DualParams):
    """
    sets all sigmas to the appropriate local Hamiltonian projection of
    the dual channel action on next sigma.
    adds the sigma_proj variable to dual_params
    """

    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_parent = dual_params.circ_params.h_parent
    layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
    p = dual_params.p
    Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init
    k_dual = dual_params.k_dual

    proj = k_local_hamiltonian_indicators(N, k_dual)

    sigmas = jnp.zeros((2 * N, 2 * N, d))

    # setting all the sigmas to the dual channel projected onto local Ham.
    # last layer sigma is just -H_parent
    sigmas = sigmas.at[:, :, d - 1].set(-jnp.real(h_parent))

    init_args = (d - 2, sigmas, layer_hamiltonians, p, proj)
    _, sigmas, _, _, _ = jax.lax.while_loop(cond_fun, set_ith_sigma_projected, init_args)

    dual_params.sigmas_proj = sigmas

@partial(jit, static_argnums = (1,))
def dual_obj_projected(a_vars: jnp.array, dual_params: DualParams):
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_parent = dual_params.circ_params.h_parent
    layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
    p = dual_params.p
    Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init
    k_dual = dual_params.k_dual

    sigmas = dual_params.sigmas_proj
    lambdas = jnp.exp(a_vars)

    cost = 0
    # log Tr exp terms

    # first d - 1 layers
    init_args = (lambdas, sigmas, layer_hamiltonians, p, cost)
    _, _, _, _, cost = jax.lax.fori_loop(0, d - 1, dual_free_energy_ith_term, init_args)

    # last layer
    hi = h_parent + sigmas.at[:,:,d-1].get()
    cost += -lambdas.at[d-1].get() * log_trace_fgstate(-hi/lambdas[d-1])

    # init. state term
    # !!
    epsilon_1_dag_sigma1 = \
    noisy_dual_layer(layer_hamiltonians.at[0, :, :].get(), sigmas.at[:,:,0].get(), p)

    cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)

    # entropy term
    # !!
    q = 1 - p
    q_powers = jnp.array([q**i for i in range(d)])

    entropy_bounds = N * p * jnp.log(2) * \
             jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])

    # !!
    cost += jnp.dot(lambdas, entropy_bounds)

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad_projected(a_vars: jnp.array, dual_params: DualParams):
    return grad(dual_obj_projected, argnums = 0)(a_vars, dual_params)

#--------------------------------------------------#
#-------------- Purity dual methods ---------------#
#--------------------------------------------------#

@jit
def purity(h: jnp.array):
    N = h.shape[0]//2
    return -1 * (2**(N-1)) * jnp.trace(jnp.matmul(h, h))

class DualParamsPurity():
    def __init__(self, circ_params: PrimalParams, p: float, k_dual: int,
                 lambda_lower_bounds: jnp.array, scale: float = 1):
        self.circ_params = circ_params
        self.p = p
        self.k_dual = k_dual
        self.lambda_lower_bounds = lambda_lower_bounds
        self.scale = scale

        block_upper_indices_to_zero = jnp.triu_indices(self.circ_params.N, self.k_dual + 1)
        block_lower_indices_to_zero = jnp.tril_indices(self.circ_params.N, -self.k_dual - 1)

        ones_N = jnp.ones((self.circ_params.N, self.circ_params.N))
        upper_ones_N = jnp.triu(ones_N, 1)
        lower_ones_N = jnp.tril(ones_N, -1)

        block_upper_band = upper_ones_N.at[block_upper_indices_to_zero].set(0.0)
        block_lower_band = lower_ones_N.at[block_lower_indices_to_zero].set(0.0)
        block_band = ones_N.at[block_upper_indices_to_zero].set(0.0)
        block_band = block_band.at[block_lower_indices_to_zero].set(0.0)

        self.block_upper_band_indices = jnp.where(block_upper_band == 1)
        self.block_lower_band_indices = jnp.where(block_lower_band == 1)
        self.block_band_indices = jnp.where(block_band == 1)

        num_vars_xx = self.block_upper_band_indices[0].shape[0]
        num_vars_pp = num_vars_xx
        num_vars_xp = self.block_band_indices[0].shape[0]

        num_sigma_vars_layer = num_vars_xx + num_vars_pp + num_vars_xp

        self.total_num_dual_vars = 1 + \
        self.circ_params.d * num_sigma_vars_layer

@partial(jit, static_argnums = (1,))
def unvec_and_process_dual_vars_purity(dual_vars: jnp.array, dual_params: DualParamsPurity):
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    block_upper_band_indices = dual_params.block_upper_band_indices
    block_lower_band_indices = dual_params.block_lower_band_indices
    block_band_indices = dual_params.block_band_indices

    sigmas = jnp.zeros((2 * N, 2 * N, d))
    zeros_N = jnp.zeros((N, N))

    a_vars = dual_vars.at[0].get()
    lambdas = dual_params.lambda_lower_bounds + jnp.log(1 + jnp.exp(a_vars))

    sigma_vars = dual_vars.at[1:].get()
    init_args = (sigmas, sigma_vars,
                 block_upper_band_indices, block_lower_band_indices,
                 block_band_indices, zeros_N)
    sigmas, _, _, _, _, _ = jax.lax.fori_loop(0, d, unvec_layer_i, init_args)

    return lambdas, sigmas

def dual_free_energy_ith_term_purity(i: int, args: Tuple):

    purity_bounds, sigmas, layer_hamiltonians, p, cost = args

    hi = sigmas.at[:,:,i].get() - noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(), sigmas.at[:,:,i+1].get(), p)

    cost += -jnp.sqrt(purity_bounds.at[i].get() * purity(hi))

    return (purity_bounds, sigmas, layer_hamiltonians, p, cost)


@partial(jit, static_argnums = (1,))
def dual_obj_purity(dual_vars: jnp.array, dual_params: DualParamsPurity):
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_parent = dual_params.circ_params.h_parent
    layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
    p = dual_params.p
    Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init

    lambdas, sigmas = unvec_and_process_dual_vars_purity(dual_vars, dual_params)

    q = 1 - p
    q_powers = jnp.array([q**i for i in range(d)])

    entropy_bounds = N * p * jnp.log(2) * \
             jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
    purity_bounds = jnp.exp(-entropy_bounds)

    cost = 0
    # purity terms
    # first d - 1 layers
    init_args = (purity_bounds, sigmas, layer_hamiltonians, p, cost)
    _, _, _, _, cost = jax.lax.fori_loop(0, d - 1, dual_free_energy_ith_term_purity, init_args)

    # last layer
    hi = h_parent + sigmas.at[:,:,d-1].get()
    cost += -lambdas.at[0].get() * log_trace_fgstate(-hi/lambdas[0])

    # init. state term
    epsilon_1_dag_sigma1 = \
    noisy_dual_layer(layer_hamiltonians.at[0, :, :].get(), sigmas.at[:,:,0].get(), p)

    cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)

    # entropy term
    # cost += jnp.dot(lambdas, entropy_bounds.at[-1].get())
    cost += lambdas.at[0].get() * entropy_bounds.at[-1].get()

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad_purity(dual_vars: jnp.array, dual_params: DualParamsPurity):
    return grad(dual_obj_purity, argnums = 0)(dual_vars, dual_params)

def optimize_dual_purity(dual_vars_init: jnp.array, dual_params: DualParamsPurity,
                  bnds: scipy.optimize.Bounds = None):

    return optimize(np.array(dual_vars_init),
                    dual_params, dual_obj_purity, dual_grad_purity,
                    num_iters = 250, bounds = bnds)


class DualParamsPuritySmooth():
    def __init__(self, circ_params: PrimalParams, p: float, k_dual: int,
                 lambda_lower_bounds: jnp.array, scale: float = 1):
        self.circ_params = circ_params
        self.p = p
        self.k_dual = k_dual
        self.lambda_lower_bounds = lambda_lower_bounds
        self.scale = scale

        block_upper_indices_to_zero = jnp.triu_indices(self.circ_params.N, self.k_dual + 1)
        block_lower_indices_to_zero = jnp.tril_indices(self.circ_params.N, -self.k_dual - 1)

        ones_N = jnp.ones((self.circ_params.N, self.circ_params.N))
        upper_ones_N = jnp.triu(ones_N, 1)
        lower_ones_N = jnp.tril(ones_N, -1)

        block_upper_band = upper_ones_N.at[block_upper_indices_to_zero].set(0.0)
        block_lower_band = lower_ones_N.at[block_lower_indices_to_zero].set(0.0)
        block_band = ones_N.at[block_upper_indices_to_zero].set(0.0)
        block_band = block_band.at[block_lower_indices_to_zero].set(0.0)

        self.block_upper_band_indices = jnp.where(block_upper_band == 1)
        self.block_lower_band_indices = jnp.where(block_lower_band == 1)
        self.block_band_indices = jnp.where(block_band == 1)

        num_vars_xx = self.block_upper_band_indices[0].shape[0]
        num_vars_pp = num_vars_xx
        num_vars_xp = self.block_band_indices[0].shape[0]

        num_sigma_vars_layer = num_vars_xx + num_vars_pp + num_vars_xp

        self.total_num_dual_vars = self.circ_params.d + \
        self.circ_params.d * num_sigma_vars_layer

@partial(jit, static_argnums = (1,))
def unvec_and_process_dual_vars_purity_smooth(dual_vars: jnp.array, dual_params: DualParamsPuritySmooth):
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    block_upper_band_indices = dual_params.block_upper_band_indices
    block_lower_band_indices = dual_params.block_lower_band_indices
    block_band_indices = dual_params.block_band_indices

    sigmas = jnp.zeros((2 * N, 2 * N, d))
    zeros_N = jnp.zeros((N, N))

    a_vars = dual_vars.at[:d].get()
    lambdas = dual_params.lambda_lower_bounds + jnp.log(1 + jnp.exp(a_vars))

    sigma_vars = dual_vars.at[d:].get()
    init_args = (sigmas, sigma_vars,
                 block_upper_band_indices, block_lower_band_indices,
                 block_band_indices, zeros_N)
    sigmas, _, _, _, _, _ = jax.lax.fori_loop(0, d, unvec_layer_i, init_args)

    return lambdas, sigmas

def dual_free_energy_ith_term_purity_smooth(i: int, args: Tuple):

    purity_bounds, lambdas, sigmas, layer_hamiltonians, p, cost = args

    hi = sigmas.at[:,:,i].get() - noisy_dual_layer(layer_hamiltonians.at[i+1, :, :].get(), sigmas.at[:,:,i+1].get(), p)

    cost += -purity(hi)/(4 * lambdas.at[i].get()) - lambdas.at[i].get() * purity_bounds.at[i].get()

    return (purity_bounds, lambdas, sigmas, layer_hamiltonians, p, cost)


@partial(jit, static_argnums = (1,))
def dual_obj_purity_smooth(dual_vars: jnp.array, dual_params: DualParamsPurity):
    N = dual_params.circ_params.N
    d = dual_params.circ_params.d
    h_parent = dual_params.circ_params.h_parent
    layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
    p = dual_params.p
    Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init

    lambdas, sigmas = unvec_and_process_dual_vars_purity_smooth(dual_vars, dual_params)

    q = 1 - p
    q_powers = jnp.array([q**i for i in range(d)])

    entropy_bounds = N * p * jnp.log(2) * \
                jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
    purity_bounds = jnp.exp(-entropy_bounds)

    cost = 0
    # purity terms
    # first d - 1 layers
    init_args = (purity_bounds, lambdas, sigmas, layer_hamiltonians, p, cost)
    _, _, _, _, _, cost = jax.lax.fori_loop(0, d - 1, dual_free_energy_ith_term_purity_smooth, init_args)

    # last layer
    hi = h_parent + sigmas.at[:,:,d-1].get()
    cost += -lambdas.at[0].get() * log_trace_fgstate(-hi/lambdas[0])

    # init. state term
    epsilon_1_dag_sigma1 = \
    noisy_dual_layer(layer_hamiltonians.at[0, :, :].get(), sigmas.at[:,:,0].get(), p)

    cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)

    # entropy term
    # cost += jnp.dot(lambdas, entropy_bounds.at[-1].get())
    cost += lambdas.at[0].get() * entropy_bounds.at[-1].get()

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad_purity_smooth(dual_vars: jnp.array, dual_params: DualParamsPurity):
    return grad(dual_obj_purity_smooth, argnums = 0)(dual_vars, dual_params)

# @partial(jit, static_argnums = (1,))
# def dual_obj_no_channel_direct_lambda(dual_vars: jnp.array, dual_params: DualParams):
#     N = dual_params.circ_params.N
#     d = dual_params.circ_params.d
#     h_parent = dual_params.circ_params.h_parent
#     p = dual_params.p
#
#     q = 1 - p
#     q_powers = jnp.array([q**i for i in range(d)])
#     entropy_bounds = N * p * jnp.log(2) * \
#              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
#     Sd = entropy_bounds.at[-1].get()
#
#     lmbda = dual_vars.at[0].get()
#     # lmbda = dual_params.lambda_lower_bounds.at[-1].get() + jnp.log(1 + jnp.exp(a))
#     # lmbda = 1 + jnp.log(1 + jnp.exp(a))
#
#     cost = -lmbda * jnp.log(trace_fgstate(-h_parent/lmbda)) + lmbda * Sd
#
#     cost += jnp.dot(dual_params.lambda_lower_bounds.at[:-1].get(),
#                     entropy_bounds.at[:-1].get() - N * np.log(2))
#
#     return -jnp.real(cost)
#
# @partial(jit, static_argnums = (1,))
# def dual_grad_no_channel_direct_lambda(dual_vars: jnp.array, dual_params: DualParams):
#     return grad(dual_obj_no_channel_direct_lambda, argnums = 0)(dual_vars, dual_params)

# @partial(jit, static_argnums = (1,))
# def dual_obj_no_channel_1q_test(dual_vars: jnp.array, dual_params: DualParams):
#     N = dual_params.circ_params.N
#     d = dual_params.circ_params.d
#     h_parent = dual_params.circ_params.h_parent
#     p = dual_params.p
#
#     q = 1 - p
#     q_powers = jnp.array([q**i for i in range(d)])
#     entropy_bounds = N * p * jnp.log(2) * \
#              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
#     entropy_bounds = jnp.ones(entropy_bounds.shape) * binary_entropy(p/2)
#     Sd = entropy_bounds.at[-1].get()
#
#     a = dual_vars.at[0].get()
#     lmbda = jnp.log(1 + jnp.exp(a))
#
#     cost = -lmbda * jnp.log(trace_fgstate_1q(-h_parent/lmbda)) + lmbda * Sd
#
#     return -jnp.real(cost)
#
# @partial(jit, static_argnums = (1,))
# def dual_grad_no_channel_1q_test(dual_vars: jnp.array, dual_params: DualParams):
#     return grad(dual_obj_no_channel_1q_test, argnums = 0)(dual_vars, dual_params)

# @partial(jit, static_argnums = (1,2))
# def unvec_and_process_dual_vars(dual_vars: jnp.array, d: int, N: int):
#     utri_indices = jnp.triu_indices(2*N, 1)
#     ltri_indices = (utri_indices[1], utri_indices[0])
#
#     sigmas = jnp.zeros((2 * N, 2 * N, d))
#     zeros_2N = jnp.zeros((2*N, 2*N))
#
#     a_vars = dual_vars.at[:d].get()
#     lambdas = jnp.log(1 + jnp.exp(a_vars))
#
#     sigma_vars = dual_vars.at[d:].get()
#     init_args = (sigmas, sigma_vars, utri_indices, ltri_indices, zeros_2N)
#     sigmas, _, _, _, _ = jax.lax.fori_loop(0, d, unvec_layer_i, init_args)
#
#     return lambdas, sigmas

# def unvec_layer_i(i: int, args: Tuple):
#     sigmas, sigma_vars, utri_indices, ltri_indices, zeros_2N = args
#
#     N = zeros_2N.shape[0]//2
#     num_sigma_vars_layer = (2*N - 1) * N
#     l = num_sigma_vars_layer
#
#     sigma_layer = zeros_2N
#     sigma_slice = jax.lax.dynamic_slice_in_dim(sigma_vars, i * l, l)
#     sigma_layer = sigma_layer.at[utri_indices].set(sigma_slice)
#     sigma_layer = sigma_layer.at[ltri_indices].set(-sigma_slice)
#     sigmas = sigmas.at[:,:,i].set(sigma_layer)
#
#     return (sigmas, sigma_vars, utri_indices, ltri_indices, zeros_2N)

# @partial(jit, static_argnums = (1,2))
# def unvec_and_process_dual_vars(dual_vars: jnp.array, d: int, N: int):
#     utri_indices = jnp.triu_indices(2*N, 1)
#     ltri_indices = (utri_indices[1], utri_indices[0])
#
#     sigmas = jnp.zeros((2 * N, 2 * N, d))
#     zeros_2N = jnp.zeros((2*N, 2*N))
#
#     a_vars = dual_vars.at[:d].get()
#     lambdas = jnp.log(1 + jnp.exp(a_vars))
#
#     dual_vars_split = jnp.split(dual_vars.at[d:].get(), d)
#     for i in range(d):
#         sigma_layer = jnp.zeros((2*N, 2*N))
#         sigma_layer = sigma_layer.at[utri_indices].set(dual_vars_split[i])
#         sigma_layer = sigma_layer.at[ltri_indices].set(-dual_vars_split[i])
#         sigmas = sigmas.at[:,:,i].set(sigma_layer)
#
#     return lambdas, sigmas

# @partial(jit, static_argnums = (1,))
# def dual_obj(dual_vars: jnp.array, dual_params: DualParams):
#     N = dual_params.circ_params.N
#     d = dual_params.circ_params.d
#     h_parent = dual_params.circ_params.h_parent
#     layer_hamiltonians = dual_params.circ_params.layer_hamiltonians
#     p = dual_params.p
#     Gamma_mjr_init = dual_params.circ_params.Gamma_mjr_init
#
#     lambdas, sigmas = unvec_and_process_dual_vars(dual_vars, d, N)
#
#     cost = 0
#     # log Tr exp terms
#     for i in range(d):
#         if i == d-1:
#             hi = h_parent + sigmas.at[:,:,i].get()
#         else:
#             hi = sigmas.at[:,:,i].get() - \
#                  noisy_dual_layer(layer_hamiltonians[i+1],
#                                   sigmas.at[:,:,i+1].get(), p)
#
#         cost += -lambdas[i] * jnp.log(trace_fgstate(-hi/lambdas[i]))
#
#     # init. state term
#     epsilon_1_dag_sigma1 = \
#     noisy_dual_layer(layer_hamiltonians[0], sigmas.at[:,:,0].get(), p)
#
#     cost += -energy(Gamma_mjr_init, epsilon_1_dag_sigma1)
#
#     # entropy term
#     q = 1 - p
#     q_powers = jnp.array([q**i for i in range(d)])
#
#     entropy_bounds = N * p * jnp.log(2) * \
#              jnp.array([jnp.sum(q_powers.at[:i+1].get()) for i in range(d)])
#
#     cost += jnp.dot(lambdas, entropy_bounds)
#
#     return -jnp.real(cost)


# for k in range(N):
#     s_prime_zeroed_out = s_prime.at[k, :].set(jnp.zeros(2 * N))
#     s_prime_zeroed_out = s_prime_zeroed_out.at[:, k].set(jnp.zeros(2 * N))
#     s_prime_zeroed_out = s_prime_zeroed_out.at[k + N, :].set(jnp.zeros(2 * N))
#     s_prime_zeroed_out = s_prime_zeroed_out.at[:, k + N].set(jnp.zeros(2 * N))
#
#     s_prime = (1 - p) * s_prime + p * (s_prime_zeroed_out)


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

# @partial(jit, static_argnums = (1,))
# def circ_obj(theta: jnp.array, params: PrimalParams):
#     Gamma_mjr = params.Gamma_mjr_init
#
#     for i in range(params.d):
#         Gamma_mjr = weighted_unitary_on_fgstate(Gamma_mjr,
#                                                 params.layer_hamiltonians[i],
#                                                 theta[i])
#
#     return jnp.real(energy(Gamma_mjr, params.h_target))
#
# @partial(jit, static_argnums = (1,))
# def circ_grad(theta: jnp.array, params: PrimalParams):
#     return grad(circ_obj, argnums = 0)(theta, params)
#
# def optimize_circuit(theta_init: jnp.array, params: PrimalParams):
#     bounds = scipy.optimize.Bounds(lb = 0.0, ub = 2 * np.pi)
#
#     return optimize(np.array(theta_init),
#                     params, circ_obj, circ_grad,
#                     num_iters = 50, bounds = bounds)

# @jit
# def weighted_unitary_on_fgstate(Gamma_mjr: jnp.array, h: jnp.array, theta: float):
#     """
#     Parameters
#     ----------
#     Gamma_mjr: Correlation matrix of f.g.s. (majorana rep.)
#     h: Generator of Gaussian unitary in majorana rep..
#        Gaussian unitary = e^{-iH} where H = i r^{\dagger} h r.
#
#     Returns
#     -------
#     Gamma_mjr_prime: Correlation matrix of f.g.s. after unitary.
#     """
#     w, v = jnp.linalg.eig(2 * h)
#
#     exp_p2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(theta * w))), jnp.conj(jnp.transpose(v)))
#     exp_m2h = jnp.matmul(jnp.matmul(v, jnp.diag(jnp.exp(-theta * w))), jnp.conj(jnp.transpose(v)))
#
#     return jnp.matmul(jnp.matmul(exp_p2h, Gamma_mjr), exp_m2h)

# def generate_target_hamiltonian(self, key: jnp.array):
#     h_target, key = random_normal_hamiltonian_majorana(self.N, key)
#     self.h_target = h_target/self.N
