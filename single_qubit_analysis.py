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
from plotter import set_size

matplotlib.rcParams["image.cmap"] = "inferno"
matplotlib.rcParams["axes.titlesize"] = 25
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 7
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["font.family"] = "Times New Roman"
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

H_parent = jnp.array([])

class SQDualParams():
    def __init__(self, theta: float, gap: float, p: float):
        self.theta = theta
        # self.H_parent = jnp.array([[0, 0,],[0, gap]])
        self.H_parent = gap * jnp.array([[1, 0,],[0, -1]])
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

    w, v = jnp.linalg.eigh(H_parent + sigma)
    exps = jnp.exp(-w/lmbda)

    cost = -jnp.trace(jnp.matmul(sigma, epsilon_rho_0)) \
    -lmbda * jnp.log(jnp.sum(exps)) \
    + lmbda * S2

    # Omega = jnp.sqrt((s - gap/2)**2 + rx**2 + ry**2)

    # cost = -jnp.trace(jnp.matmul(sigma, epsilon_rho_0)) \
    # + gap/2 - lmbda * jnp.log(jnp.exp(Omega/lmbda) + jnp.exp(-Omega/lmbda)) \
    # + lmbda * S2

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_grad_sq(dual_vars: jnp.array, dual_params: SQDualParams):
    return grad(dual_obj_sq, argnums = 0)(dual_vars, dual_params)

@partial(jit, static_argnums = (1,))
def dual_nc_sq(dual_vars: jnp.array, dual_params: SQDualParams):
    lmbda = jnp.log(1 + jnp.exp(dual_vars.at[0].get()))

    H_parent = dual_params.H_parent
    p = dual_params.p
    S2 = dual_params.entropy

    w, v = jnp.linalg.eigh(H_parent)
    exps = jnp.exp(-w/lmbda)

    cost = -lmbda * jnp.log(jnp.sum(exps)) + lmbda * S2

    return -jnp.real(cost)

@partial(jit, static_argnums = (1,))
def dual_nc_grad_sq(dual_vars: jnp.array, dual_params: SQDualParams):
    return grad(dual_nc_sq, argnums = 0)(dual_vars, dual_params)

def primal(theta, gap, p):
    dual_params = SQDualParams(theta, gap, p)
    H_parent = gap * jnp.array([[1, 0,],[0, -1]])
    I = jnp.eye(2)
    psi_init = jnp.array([jnp.cos(theta/2), jnp.sin(theta/2)])

    epsilon_rho_0 = p * I/2 + (1-p) * jnp.outer(psi_init, psi_init.conj())

    return jnp.trace(jnp.matmul(H_parent, epsilon_rho_0))

theta = np.pi/2
gap = 2
p = 0.1

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
noisy_nc_list = []

for theta in theta_list:
    dual_params = SQDualParams(theta, gap, p)
    dual_vars_init = jnp.zeros((4,))
    num_steps = int(5e3)
    dual_obj_over_opti, dual_opt_result = \
        gaussian.optimize(dual_vars_init, dual_params,
                          dual_obj_sq, dual_grad_sq,
                          num_iters = num_steps)
    noisy_bound = -dual_obj_sq(jnp.array(dual_opt_result.x), dual_params)

    dual_vars_init = jnp.zeros((1,))
    num_steps = int(5e3)
    dual_obj_over_opti, dual_opt_result = \
        gaussian.optimize(dual_vars_init, dual_params,
                          dual_nc_sq, dual_nc_grad_sq,
                          num_iters = num_steps)
    noisy_nc = -dual_nc_sq(jnp.array(dual_opt_result.x), dual_params)

    noisy_bound_list.append(noisy_bound)
    noisy_nc_list.append(noisy_nc)

duality_gap_list = primal_list - np.array(noisy_bound_list)


width = 510/2
fig = plt.figure(figsize=set_size(width, fraction = 1, subplots = (1,1)))
ax = fig.add_subplot(111)
ax.plot(theta_list/np.pi, noisy_bound_list, label = "Circuit dual")
# ax.plot(theta_list, primal_list, ls = '--', label = "Primal")
# ax.plot(theta_list, [gap * p/2] * len(theta_list), label = "Dual (no channel)", ls = '--')
# ax.plot(theta_list, [gap * (p-1)] * len(theta_list), label = "Dual (no channel)", ls = '--')
ax.plot(theta_list/np.pi, noisy_nc_list, label = "Entropic", ls = '--', color = 'k')
ax.set_xlabel('Rotation, ' + r"$\theta$")
ax.legend()
# ax.set_ylabel('Bound')
# ax.set_ylim((0, np.max(primal_list) + 0.2))
ax.set_yticks([-gap*(1-p), 0, gap * (1-p)])
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$'])
ax.set_yticklabels([r'$- \Delta (1 - p)$', r'$0$', r'$\Delta (1 - p)$'])
plt.tight_layout()
data_path = "./../vqa_data/"
figname = "sq_analysis.pdf"
plt.savefig(os.path.join(data_path, figname), bbox_inches = 'tight', format = 'pdf')