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

def unjaxify_obj(func):

    def wrap(*args):
        return float(func(jnp.array(args[0]), args[1]))

    return wrap

def unjaxify_grad(func):

    def wrap(*args):
        return np.array(func(jnp.array(args[0]), args[1]), order = 'F')

    return wrap

def optimize_dual(dual_vars_init: np.array, sys_obj, dual_obj_fun: Callable, dual_grad_fun: Callable,
                  num_iters: int, a_bound: float, sigma_bound: float, use_bounds,
                  opt_method: str = "L-BFGS-B"):

    opt_args = (sys_obj,)

    obj_over_opti = []

    def callback_func(x):

        obj_eval = unjaxify_obj(dual_obj_fun)(x, opt_args[0])

        obj_over_opti.append(obj_eval)

        # print('Dir. Iteration ', str(len(obj_over_opti)), '. Objective = ', str(obj_eval), '. x = ', x)

    # sigma_bound = 1e1
    # p = dual_obj.p
    # len_vars = dual_obj.len_vars
    # len_vars = vars_init.shape[0]

    if use_bounds:

        bnds = scipy.optimize.Bounds(lb = [a_bound] * sys_obj.d + [-sigma_bound] * (sys_obj.total_num_vars_full - sys_obj.d),
                                     ub = [np.inf] * sys_obj.d + [sigma_bound] * (sys_obj.total_num_vars_full - sys_obj.d))

    else:

        bnds = scipy.optimize.Bounds(lb = -np.inf, ub = np.inf)

    if opt_method == "L-BFGS-B":

        opt_result = scipy.optimize.minimize(
                                unjaxify_obj(dual_obj_fun),
                                dual_vars_init, args = opt_args,
                                method = opt_method,
                                jac = unjaxify_grad(dual_grad_fun),
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
                                unjaxify_obj(dual_obj_fun),
                                dual_vars_init, args = opt_args,
                                method = opt_method,
                                jac = unjaxify_grad(dual_grad_fun),
                                options={'gtol': 1e-05,
                                'eps': 1.4901161193847656e-08,
                                'maxiter': 300,
                                'disp': True,
                                'return_all': False},
                                callback = callback_func)

    elif opt_method == "Newton-CG":

        opt_result = scipy.optimize.minimize(
                                unjaxify_obj(dual_obj_fun),
                                dual_vars_init, args = opt_args,
                                method = opt_method,
                                jac = unjaxify_grad(dual_grad_fun),
                                options={'xtol': 1e-05,
                                'eps': 1.4901161193847656e-08,
                                'maxiter': 300,
                                'disp': True,
                                'return_all': False},
                                callback = callback_func)

    elif opt_method == "CG":

        opt_result = scipy.optimize.minimize(
                                unjaxify_obj(dual_obj_fun),
                                dual_vars_init, args = opt_args,
                                method = opt_method,
                                jac = unjaxify_grad(dual_grad_fun),
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
