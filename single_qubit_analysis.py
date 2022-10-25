import abc
from typing import List, Tuple, Callable, Dict
from functools import partial
import time

import pickle
import sys
import argparse
import io
import copy
import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import jax.numpy as jnp
import jax
import scipy
from jax.config import config
from jax import jit, grad, vmap, value_and_grad
config.update("jax_enable_x64", True)

from fermions import gaussian

H_parent = jnp.array([])

class SQDualParams():
    def __init__(self, theta: float, gap: float, p: float):
        self.theta = theta
        self.H_parent = jnp.array([[0, 0,],[0, gap]])
        self.gap = gap
        self.psi_init = jnp.array([jnp.cos(theta/2), jnp.sin(theta/2)])
        self.p = p
        self.q = 1-p
        self.entropy = -(self.p/2) * jnp.log(self.p/2) -(1-self.p/2) * jnp.log(1-self.p/2)

@partial(jit, static_argnums = (1,))
def dual_obj_sq(dual_vars: jnp.array, dual_params: SQDualParams):
    a, s, rx, ry = dual_vars

    lmbda = jnp.log(1 + jnp.exp(a))

    H_parent = dual_params.H_parent
    p = dual_params.p
    S2 = dual_params.entropy
    I = jnp.eye(2)
    psi_init = dual_params.psi_init
    gap = dual_params.gap

    sigma = jnp.array([[s, rx + 1j * ry], [rx - 1j * ry, -s]])

    epsilon_rho_0 = p * I/2 + (1-p) * jnp.outer(psi_init, psi_init.conj())

    # cost = -jnp.trace(jnp.matmul(sigma, epsilon_rho_0)) \
    # -lmbda * jnp.log(jnp.trace(jnp.exp(-(H_parent + sigma)/lmbda))) \
    # + lmbda * S2

    Omega = jnp.sqrt((s - gap/2)**2 + rx**2 + ry**2)

    cost = -jnp.trace(jnp.matmul(sigma, epsilon_rho_0)) \
    + gap/2 - lmbda * jnp.log(jnp.exp(Omega/lmbda) + jnp.exp(-Omega/lmbda)) \
    + lmbda * S2

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad_sq(dual_vars: jnp.array, dual_params: SQDualParams):
    return grad(dual_obj_sq, argnums = 0)(dual_vars, dual_params)

def primal(theta, gap, p):
    dual_params = SQDualParams(theta, gap, p)
    H_parent = jnp.array([[0, 0,],[0, gap]])
    I = jnp.eye(2)
    psi_init = jnp.array([jnp.cos(theta/2), jnp.sin(theta/2)])

    epsilon_rho_0 = p * I/2 + (1-p) * jnp.outer(psi_init, psi_init.conj())

    return jnp.trace(jnp.matmul(H_parent, epsilon_rho_0))

theta = np.pi/2
gap = 1.4
p = 0.01

dual_params = SQDualParams(theta, gap, p)
dual_vars_init = jnp.zeros((4,))

num_steps = int(5e3)
dual_obj_over_opti, dual_opt_result = \
    gaussian.optimize(dual_vars_init, dual_params,
                      dual_obj_sq, dual_grad_sq,
                      num_iters = num_steps)
noisy_bound = -dual_obj_sq(jnp.array(dual_opt_result.x), dual_params)
print("noisy bound = ", noisy_bound)

theta_list = np.linspace(0, np.pi, 100)
primal_list = jax.vmap(primal, in_axes = [0, None, None])(theta_list, gap, p)
noisy_bound_list = []

for theta in theta_list:
    dual_params = SQDualParams(theta, gap, p)
    dual_vars_init = jnp.zeros((4,))
    num_steps = int(5e3)
    dual_obj_over_opti, dual_opt_result = \
        gaussian.optimize(dual_vars_init, dual_params,
                          dual_obj_sq, dual_grad_sq,
                          num_iters = num_steps)
    noisy_bound = -dual_obj_sq(jnp.array(dual_opt_result.x), dual_params)

    noisy_bound_list.append(noisy_bound)

duality_gap_list = primal_list - np.array(noisy_bound_list)
