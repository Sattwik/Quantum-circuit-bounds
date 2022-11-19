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
import optax

from vqa_bounds import meta_system

@partial(jit, static_argnums = (1,))
def dual_obj_ext(dual_vars: jnp.array, sys_obj: meta_system.System):

    """
    NB the default assumption with the dual variables being typed as jax arrays
     that the dual objective is JAXed.
    """

    return sys_obj.dual_obj(dual_vars)

@partial(jit, static_argnums = (1,))
def dual_grad_ext(dual_vars: jnp.array, sys_obj: meta_system.System):

    """
    NB the default assumption with the dual variables being typed as jax arrays
     that the dual gradient is JAXed.
    """

    return sys_obj.dual_grad(dual_vars)

def dual_hessp_ext(dual_vars: jnp.array, vec: jnp.array, sys_obj: meta_system.System):

    return sys_obj.dual_hessian_prod(dual_vars, vec)

def fd_grad_ext(dual_vars: jnp.array, positions: Tuple, sys_obj: meta_system.System):

    objective_0 = dual_obj_ext(vars_vec, sys_obj)
    delta = 1e-7

    gradient_list = jnp.zeros(len(positions), dtype = complex)

    for i in positions:

        print(i)

        vars_tmp = vars_vec
        vars_tmp = vars_tmp.at[i].add(delta)
        objective_plus = dual_obj_ext(vars_tmp, sys_obj)

        vars_tmp = vars_vec
        vars_tmp = vars_tmp.at[i].add(-delta)
        objective_minus = dual_obj_ext(vars_tmp, sys_obj)

        gradient_list = gradient_list.at[i].set((objective_plus - objective_minus)/(2 * delta))

    return gradient_list

def unjaxify_obj(func):

    def wrap(*args):
        return float(func(jnp.array(args[0]), args[1]))

    return wrap

def unjaxify_grad(func):

    def wrap(*args):
        return np.array(func(jnp.array(args[0]), args[1]), order = 'F')

    return wrap

def unjaxify_hessp(func):

    def wrap(*args):
        return np.array(func(jnp.array(args[0]), jnp.array(args[1]), args[2]), order = 'F')

    return wrap

def optimize_dual(dual_vars_init: np.array, sys_obj: meta_system.System,
                  num_iters: int, a_bound: float, sigma_bound: float, use_bounds,
                  opt_method: str = "L-BFGS-B"):

    opt_args = (sys_obj,)

    obj_over_opti = []

    def callback_func(x):

        obj_eval = unjaxify_obj(dual_obj_ext)(x, opt_args[0])

        obj_over_opti.append(obj_eval)

        # print('Dir. Iteration ', str(len(obj_over_opti)), '. Objective = ', str(obj_eval), '. x = ', x)

    # sigma_bound = 1e1
    # p = dual_obj.p
    # len_vars = dual_obj.len_vars
    # len_vars = vars_init.shape[0]

    if use_bounds:

        bnds = scipy.optimize.Bounds(lb = [a_bound] * sys_obj.d + [-sigma_bound] * (sys_obj.total_num_vars - sys_obj.d),
                                     ub = [np.inf] * sys_obj.d + [sigma_bound] * (sys_obj.total_num_vars - sys_obj.d))
        if opt_method == "L-BFGS-B":

            opt_result = scipy.optimize.minimize(
                                    unjaxify_obj(dual_obj_ext),
                                    dual_vars_init, args = opt_args,
                                    method = opt_method,
                                    jac = unjaxify_grad(dual_grad_ext),
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
                                    unjaxify_obj(dual_obj_ext),
                                    dual_vars_init, args = opt_args,
                                    method = opt_method,
                                    jac = unjaxify_grad(dual_grad_ext),
                                    options={'gtol': 1e-05,
                                    'eps': 1.4901161193847656e-08,
                                    'maxiter': 300,
                                    'disp': True,
                                    'return_all': False},
                                    callback = callback_func)

        elif opt_method == "Newton-CG":

            opt_result = scipy.optimize.minimize(
                                    unjaxify_obj(dual_obj_ext),
                                    dual_vars_init, args = opt_args,
                                    method = opt_method,
                                    jac = unjaxify_grad(dual_grad_ext),
                                    options={'xtol': 1e-05,
                                    'eps': 1.4901161193847656e-08,
                                    'maxiter': 300,
                                    'disp': True,
                                    'return_all': False},
                                    callback = callback_func)

        elif opt_method == "CG":

            opt_result = scipy.optimize.minimize(
                                    unjaxify_obj(dual_obj_ext),
                                    dual_vars_init, args = opt_args,
                                    method = opt_method,
                                    jac = unjaxify_grad(dual_grad_ext),
                                    options={'gtol': 1e-05,
                                    'eps': 1.4901161193847656e-08,
                                    'maxiter': 300,
                                    'disp': True,
                                    'return_all': False,
                                    'finite_diff_rel_step': None},
                                    callback = callback_func)

        else:

            raise ValueError("Method not yet implemented")
    else:
        print('not using dual variable bounds')

        if opt_method == "L-BFGS-B":

            opt_result = scipy.optimize.minimize(
                                    unjaxify_obj(dual_obj_ext),
                                    dual_vars_init, args = opt_args,
                                    method = opt_method,
                                    jac = unjaxify_grad(dual_grad_ext),
                                    options={'disp': None,
                                    'maxcor': 10,
                                    'ftol': 2.220446049250313e-15,
                                    'gtol': 1e-05,
                                    'eps': 1e-08,
                                    'maxfun': 15000,
                                    'maxiter': num_iters,
                                    'iprint': 10,
                                    'maxls': 20},
                                    callback = callback_func)

        elif opt_method == "BFGS":

            opt_result = scipy.optimize.minimize(
                                    unjaxify_obj(dual_obj_ext),
                                    dual_vars_init, args = opt_args,
                                    method = opt_method,
                                    jac = unjaxify_grad(dual_grad_ext),
                                    options={'gtol': 1e-05,
                                    'eps': 1.4901161193847656e-08,
                                    'maxiter': 300,
                                    'disp': True,
                                    'return_all': False},
                                    callback = callback_func)

        elif opt_method == "Newton-CG":

            opt_result = scipy.optimize.minimize(
                                    unjaxify_obj(dual_obj_ext),
                                    dual_vars_init, args = opt_args,
                                    method = opt_method,
                                    jac = unjaxify_grad(dual_grad_ext),
                                    hessp = unjaxify_hessp(dual_hessp_ext),
                                    options={'xtol': 1e-05,
                                    'eps': 1.4901161193847656e-08,
                                    'maxiter': 300,
                                    'disp': True,
                                    'return_all': False},
                                    callback = callback_func)

        elif opt_method == "CG":

            opt_result = scipy.optimize.minimize(
                                    unjaxify_obj(dual_obj_ext),
                                    dual_vars_init, args = opt_args,
                                    method = opt_method,
                                    jac = unjaxify_grad(dual_grad_ext),
                                    options={'gtol': 1e-05,
                                    'eps': 1.4901161193847656e-08,
                                    'maxiter': 300,
                                    'disp': True,
                                    'return_all': False,
                                    'finite_diff_rel_step': None},
                                    callback = callback_func)

        else:

            raise ValueError("Method not yet implemented")

    return np.array(obj_over_opti), opt_result

def optax_optimize(sys_obj: meta_system.System, vars_init: jnp.array, alpha: float, num_steps: int, method: str):

    if method == 'adam':
        optimizer = optax.adam(learning_rate=alpha)

    if method == 'amsgrad':
        optimizer = optax.amsgrad(learning_rate=alpha)

    if method == 'adabelief':
        optimizer = optax.adabelief(learning_rate=alpha)

    @jit
    def step(vars, opt_state):
        value = dual_obj_ext(vars, sys_obj)
        grads = dual_grad_ext(vars, sys_obj)

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

def gd_optimize(sys_obj: meta_system.System, vars_init: jnp.array, eta: float, num_steps: int):

    @jit
    def step(t, vars):
        value = dual_obj_ext(vars, sys_obj)
        grads = dual_grad_ext(vars, sys_obj)
        vars = vars - eta * grads
        return value, grads, vars

    vars = vars_init
    value_array = jnp.zeros(num_steps)

    for t in range(num_steps):

        # print("Step :", t)
        value, grads, vars = step(t, vars)

        if t > 10 and jnp.linalg.norm(grads) <= 1e-5:
            return value_array, vars

        prev_grads = grads

        value_array = value_array.at[t].set(value)

        if t%(num_steps//100) == 0:
            print("Step :" + str(t) + ", |grad| = " + str(jnp.linalg.norm(grads)))

    return value_array, vars
