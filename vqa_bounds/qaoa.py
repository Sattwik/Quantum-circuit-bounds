import abc
from typing import List, Tuple, Callable, Dict
from functools import partial
import copy

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.config import config
config.update("jax_enable_x64", True)
import tensornetwork as tn
tn.set_default_backend("jax")
import scipy.optimize

# from vqa_bounds import graphs
import graphs

def maxcut_ham(graph):
    """
    Returns the MAX-CUT problem Hamiltonian on a given graph.
    """
    Z = jnp.array([[1, 0], [0, -1]], dtype = complex)

    num_nodes = len(graph.nodes)
    Z_ops = {}

    identity_all = jnp.identity(2 ** num_nodes)
    for i, node in enumerate(graph.nodes):
        identity_left = jnp.identity(2 ** i)
        identity_right = jnp.identity(2 ** (num_nodes - i - 1))
        Z_ops[node] = jnp.kron(identity_left, jnp.kron(Z, identity_right))

    H = 0
    for edge in graph.edges:
        Zj = Z_ops[edge[0]]
        Zk = Z_ops[edge[1]]
        H += -1/2 * (identity_all - Zj * Zk)

    return H

@partial(jit, static_argnums = (1,2))
def qaoa_state(params, depth, graph):
    """
    Returns the state obtained by acting with the QAOA circuit on
    |+>^{\otimes len(graph.nodes)}.
    """
    H = maxcut_ham(graph)
    xplus = 1/jnp.sqrt(2) * jnp.array([1, 1], dtype = complex)
    X = jnp.array([[0,1],[1,0]], dtype = complex)
    num_nodes = len(graph.nodes)

    Hmixing = 0
    for i, node in enumerate(graph.nodes):
        identity_left = jnp.identity(2 ** i)
        identity_right = jnp.identity(2 ** (num_nodes - i - 1))
        Hmixing += jnp.kron(identity_left, jnp.kron(X, identity_right))

    state = xplus
    for i in range(num_nodes - 1):
        state = jnp.kron(state, xplus)

    gamma = params.at[:depth].get()
    beta = params.at[depth:].get()

    for i in range(depth):
        U = jax.scipy.linalg.expm(-1j * gamma.at[i].get() * H)
        state = jnp.matmul(U, state)

        sq_mixing = jax.scipy.linalg.expm(-1j * beta.at[i].get()/2 * X)
        Umixing = 1
        for i in range(num_nodes):
            Umixing = jnp.kron(Umixing, sq_mixing)
        state = jnp.matmul(Umixing, state)

    return state

@partial(jit, static_argnums = (1,2,3))
def expect_on_edge(params, depth, graph, edge):
    """
    Calculates the expectation value of the MAX-CUT Ham. term corresponding to
    an edge of the graph in the state obtained after the QAOA circuit. For use
    with the light cone version of the objective.
    """
    Z = jnp.array([[1, 0], [0, -1]], dtype = complex)

    num_nodes = len(graph.nodes)
    Z_ops = {}

    identity_all = jnp.identity(2 ** num_nodes)
    for i, node in enumerate(graph.nodes):
        identity_left = jnp.identity(2 ** i)
        identity_right = jnp.identity(2 ** (num_nodes - i - 1))
        Z_ops[node] = jnp.kron(identity_left, jnp.kron(Z, identity_right))
    edge_ham = -1/2 * (identity_all - Z_ops[edge[0]] * Z_ops[edge[1]])

    state = qaoa_state(params, depth, graph)

    return jnp.dot(state.conj(), jnp.matmul(edge_ham, state))

@partial(jit, static_argnums = (1,2))
def expect_on_full_ham(params, depth, graph):
    """
    Calculates the expectation value of the full MAX-CUT Ham. (all edges)
    in the state obtained after the QAOA circuit.
    """
    state = qaoa_state(params, depth, graph)
    H = maxcut_ham(graph)

    return jnp.dot(state.conj(), jnp.matmul(H, state))

@partial(jit, static_argnums = (1,2))
def obj_qaoa_maxcut_full(params, depth, graph):
    return jnp.real(expect_on_full_ham(params, depth, graph))

@partial(jit, static_argnums = (1,2))
def obj_qaoa_maxcut_cone(params, depth, graph):
    edges, subgraphs = graphs.all_dist_p_subgraphs(graph, p = depth)
    obj = 0
    for edge, subgraph in zip(edges, subgraphs):
        obj += expect_on_edge(params, depth, subgraph, edge)
    return jnp.real(obj)

@partial(jit, static_argnums = (1,2))
def grad_qaoa_maxcut_full(params, depth, graph):
    return grad(obj_qaoa_maxcut_full, argnums = 0)(params, depth, graph)

@partial(jit, static_argnums = (1,2))
def grad_qaoa_maxcut_cone(params, depth, graph):
    return grad(obj_qaoa_maxcut_cone, argnums = 0)(params, depth, graph)

def unjaxify_obj(func):
    def wrap(*args):
        return float(func(jnp.array(args[0]), args[1], args[2]))
    return wrap

def unjaxify_grad(func):
    def wrap(*args):
        return np.array(func(jnp.array(args[0]), args[1], args[2]), order = 'F')
    return wrap

def optimize_circ_params(mode,
                params_init: np.array, depth, graph,
                num_iters: int, opt_method: str = "L-BFGS-B"):
    opt_args = (depth, graph)

    if mode == "cone":
        obj_fun = obj_qaoa_maxcut_cone
        grad_fun = grad_qaoa_maxcut_cone
    elif mode == "full":
        obj_fun = obj_qaoa_maxcut_full
        grad_fun = grad_qaoa_maxcut_full
    else:
        raise ValueError

    obj_over_opti = []
    def callback_func(x):
        obj_eval = unjaxify_obj(obj_fun)(x, *opt_args)
        obj_over_opti.append(obj_eval)

    bnds = scipy.optimize.Bounds(lb = 0, ub = 2 * np.pi)
    if opt_method == "L-BFGS-B":
        opt_result = scipy.optimize.minimize(
                                unjaxify_obj(obj_fun),
                                params_init, args = opt_args,
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
    else:
        raise NotImplementedError

    return np.array(obj_over_opti), opt_result

if __name__ == "__main__":
    lattice = graphs.define_lattice((1,10))
    graph = graphs.create_random_connectivity(lattice)
    depth = 2

    key = jax.random.PRNGKey(60)
    params_init = jax.random.uniform(key = key, shape = (2 * depth,),
                                     minval = 0, maxval = 2 * np.pi)

    # obj_full = obj_qaoa_maxcut_full(params_init, depth, graph)
    # obj_cone = obj_qaoa_maxcut_cone(params_init, depth, graph)

    obj_over_opti, opt_result = optimize_circ_params("cone",
                                params_init, depth, graph,
                                num_iters = 50)

    print(obj_qaoa_maxcut_full(jnp.array(opt_result.x), depth, graph))
    print(obj_qaoa_maxcut_cone(jnp.array(opt_result.x), depth, graph))


    # print(obj_full)
    # print(obj_cone)
