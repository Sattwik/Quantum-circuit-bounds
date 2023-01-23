"""
V1:
    sigma_d = -H.
    lambda_d = 0.
    OBC.

    TODO:
        1. test canonicalization with D < actual dim
        2. lax fori_loops?
"""

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

class Hamiltonian():
    def __init__(self, num_sites: int):
        self.num_sites = num_sites
        self.site_tuple_list = [(i , i + 1) for i in range(num_sites - 1)]
        self.terms = {}

    def full_ham(self):
        full_mat = 0

        for site_tuple, term in self.terms.items():
            dim_left = 2 ** site_tuple[0]
            dim_right = 2 ** (self.num_sites - site_tuple[-1] - 1)

            identity_left = jnp.identity(dim_left)
            identity_right = jnp.identity(dim_right)

            full_mat += jnp.kron(identity_left, jnp.kron(term, identity_right))

        return full_mat

#------------------------------------------------------------------------------#
# MPO methods:
# MPOs are just a list of jnp arrays (tensors) with shapes consistent
# with the bonds (l,u,r,d).
#------------------------------------------------------------------------------#

def full_contract(tensors: List[jnp.array]):
    """
    WARNING: memory expensive!!
    """
    num_sites = len(tensors)

    res = tensors[0]

    for i in range(num_sites-1):
        res = jnp.tensordot(res, tensors[i + 1], axes=((-2,), (0,))) # (l) (u) (d) (u') (r') (d')

    res = jnp.swapaxes(res, -1, -2) # l, u1, d1, u2, d2, ..., dN, rN
    res = jnp.squeeze(res) # u1, d1, u2, d2, ..., dN

    # (u1) (d1) (u2) (d2) ... (uN) (dN) -> (u1) (u2) ... (uN) (d1) (d2) ... (dN)
    # [0, 2, 4, ..., 2N - 2, 1, 3, ..., 2N - 1]
    i1 = np.arange(0, 2 * num_sites - 1, 2)
    i2 = np.arange(1, 2 * num_sites, 2)
    i = np.concatenate((i1, i2))
    res = jnp.transpose(res, tuple(i))

    res = jnp.reshape(res, (2**num_sites, 2**num_sites))

    return res

@partial(jit, static_argnums = 1)
def left_split_lurd_tensor(tensor: jnp.array, D = None):
    tensor_copy = jnp.swapaxes(tensor, -1, -2) # l, u, d, r
    tensor_copy = jnp.reshape(tensor_copy,
    (tensor_copy.shape[0] * tensor_copy.shape[1] * tensor_copy.shape[2],
    tensor_copy.shape[3])) # lud, r
    u, s, vh = jnp.linalg.svd(tensor_copy, full_matrices = False)

    # u.shape = (lud, D)
    # s.shape = (D,)
    # vh.shape = (D, r)

    if D is not None:
        # D = jnp.min(jnp.array((D, K))) #JAX conveniently takes care of this
        # with how an out of bounds index is handled
        s = s[:D]
        u = u[:, :D]
        vh = vh[:D, :]

    u = jnp.reshape(u, (tensor.shape[0], tensor.shape[1], tensor.shape[3], u.shape[1])) # l,u,d,D
    u = jnp.swapaxes(u, -1, -2) # l, u, D, d

    return u, s, vh

@partial(jit, static_argnums = 1)
def right_split_lurd_tensor(tensor: jnp.array, D = None):
    tensor_copy = jnp.swapaxes(tensor, -1, -2) # l, u, d, r
    tensor_copy = jnp.reshape(tensor_copy,
    (tensor_copy.shape[0],
    tensor_copy.shape[1] * tensor_copy.shape[2] * tensor_copy.shape[3])) # l, udr
    u, s, vh = jnp.linalg.svd(tensor_copy, full_matrices = False)

    # u.shape = (l, D)
    # s.shape = (D,)
    # vh.shape = (D, udr)

    if D is not None:
        # D = jnp.min(jnp.array((D, K))) #JAX conveniently takes care of this
        # with how an out of bounds index is handled
        s = s[:D]
        u = u[:, :D]
        vh = vh[:D, :]

    vh = jnp.reshape(vh, (vh.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[2])) # D, u, d, r
    vh = jnp.swapaxes(vh, -1, -2) # D, u, r, d

    return u, s, vh

@partial(jit, static_argnums = (1,))
def left_canonicalize(tensors: List[jnp.array], D:int = None):
    """
    Left canonicalize (leave last site uncanonicalized)
    and compress if D is specified.
    """

    num_sites = len(tensors)
    dims_max = (2 * 2) ** np.concatenate((np.arange(1, num_sites//2 + 1),
                                     np.arange(num_sites//2 - 1, 0, -1)))
    num_bonds = num_sites - 1

    if D is not None:
        compressed_dims = jnp.where(dims_max <= D, dims_max, D)
    else:
        compressed_dims = [None] * num_bonds

    for i in range(num_sites - 1):
        u, s, vh = left_split_lurd_tensor(tensors[i], D = compressed_dims[i])
        svh = jnp.matmul(jnp.diag(s), vh) # D, r
        new_right = jnp.tensordot(svh, tensors[i + 1], axes=((-1,), (0,))) # D, u, r, d

        tensors[i] = u
        tensors[i + 1] = new_right

    return tensors

@partial(jit, static_argnums = (1,))
def right_canonicalize(tensors: List[jnp.array], D:int = None):
    """
    Right canonicalize (leave first site uncanonicalized)
    and compress if D is specified.
    """

    num_sites = len(tensors)
    dims_max = (2 * 2) ** np.concatenate((np.arange(1, num_sites//2 + 1),
                                     np.arange(num_sites//2 - 1, 0, -1)))
    num_bonds = num_sites - 1

    if D is not None:
        compressed_dims = jnp.where(dims_max <= D, dims_max, D)
    else:
        compressed_dims = [None] * num_bonds

    for i in range(num_sites - 1, 0, -1):
        u, s, vh = right_split_lurd_tensor(tensors[i], D = compressed_dims[i - 1])
        us = jnp.matmul(u, jnp.diag(s)) # l, D
        new_left = jnp.tensordot(us, tensors[i - 1], axes=((0,), (2,))) #  D l u d
        new_left = jnp.transpose(new_left, [1, 2, 0, 3])

        tensors[i] = vh
        tensors[i - 1] = new_left

    return tensors

@partial(jit, static_argnums = (1,))
def check_canon(tensors: List[jnp.array], canon = "left"):
    num_sites = len(tensors)
    norm_list = []

    if canon == "left":
        for i in range(num_sites - 1):
            check = jnp.tensordot(tensors[i], tensors[i].conj(),
                    axes = ((0, 1, 3), (0, 1, 3)))
            norm_list.append(jnp.linalg.norm(check - jnp.identity(check.shape[0])))
    elif canon == "right":
        for i in range(num_sites - 1, 0, -1):
            check = jnp.tensordot(tensors[i], tensors[i].conj(),
                    axes = ((2, 1, 3), (2, 1, 3)))
            norm_list.append(jnp.linalg.norm(check - jnp.identity(check.shape[0])))
    else:
        raise ValueError
    return norm_list

@jit
def subtract_MPO(tensorsA: List[jnp.array], tensorsB: List[jnp.array]):
    """
    Direct sum.
    """
    num_sites = len(tensorsA)

    sum_tensors = []

    i = 0
    tA = tensorsA[i]
    tB = tensorsB[i]

    new_tensor = jnp.zeros((tA.shape[0], tA.shape[1],
    tA.shape[2] + tB.shape[2], tA.shape[3]), dtype = complex)

    new_tensor = new_tensor.at[:, :, :tA.shape[2], :].set(tA)
    new_tensor = new_tensor.at[:, :, tA.shape[2]:, :].set(-tB)
    sum_tensors.append(new_tensor)

    for i in range(1, num_sites - 1):
        tA = tensorsA[i]
        tB = tensorsB[i]

        new_tensor = jnp.zeros((tA.shape[0] + tB.shape[0], tA.shape[1],
        tA.shape[2] + tB.shape[2], tA.shape[3]), dtype = complex)

        new_tensor = new_tensor.at[:tA.shape[0], :, :tA.shape[2], :].set(tA)
        new_tensor = new_tensor.at[tA.shape[0]:, :, tA.shape[2]:, :].set(-tB)

        sum_tensors.append(new_tensor)

    i = num_sites - 1
    tA = tensorsA[i]
    tB = tensorsB[i]

    new_tensor = jnp.zeros((tA.shape[0] + tB.shape[0], tA.shape[1],
    tA.shape[2], tA.shape[3]), dtype = complex)

    new_tensor = new_tensor.at[:tA.shape[0], :, :, :].set(tA)
    new_tensor = new_tensor.at[tA.shape[0]:, :, :, :].set(-tB)
    sum_tensors.append(new_tensor)

    return sum_tensors

@jit
def trace_MPO_squared(tensors: List[jnp.array]):
    """
    """
    # bond_dims = [tensor.shape[2] for tensor in tensors[:-1]]
    num_sites = len(tensors)
    tooth = tensors[0]

    for i in range(0, num_sites - 1):
        # Di = bond_dims[i]
        # zip = jnp.zeros((Di, Di), dtype = complex)
        zip = jnp.tensordot(tooth, tensors[i], axes = ((0,1,3), (0,3,1)))
        # Di (tooth), Di (tensor)
        tooth = jnp.tensordot(zip, tensors[i+1], axes = ((0,),(0,)))
        # Di (tooth), Di (tensor) tdot l,u,r,d -> Di (tensor), u,r,d

    res = jnp.tensordot(tooth, tensors[-1], axes = ((0,1,2,3), (0,3,2,1)))

    return res

#------------------------------------------------------------------------------#
# Models and circuits
#------------------------------------------------------------------------------#

def HamSumZ(num_sites: int):
    """
    Returns Hamiltonian and MPO representations of \sum Z.
    """
    H = Hamiltonian(num_sites)

    I = jnp.array([[1, 0],[0, 1]], dtype = complex)
    Z = jnp.array([[1, 0],[0, -1]], dtype = complex)

    for site_tuple in H.site_tuple_list:
        H.terms[site_tuple] = jnp.kron(I, Z)

    H.terms[(0,1)] += jnp.kron(Z, I)

    arrs = []

    op = jnp.block([Z, I]) # shape = (bs) (b' s')) # b is bond, s is physical
    op = jnp.reshape(op, (1, 2, 2, 2)) # shape = (b) (s) (b') (s')
    arrs.append(op)

    op = jnp.block([[I, jnp.zeros((2,2))], [Z, I]]) # shape = (bs) (b' s'))
    op = jnp.reshape(op, (2, 2, 2, 2)) # shape = (b) (s) (b') (s')
    arrs += [op] * (num_sites - 2)

    op = jnp.row_stack((I, Z))
    op = jnp.reshape(op, (2, 2, 1, 2)) # shape = (b) (s) (b') (s')
    arrs.append(op)

    return H, arrs

@partial(jit, static_argnums=(1,2))
def gate_to_MPO(gate: jnp.array, num_sites: int, D: int = None):
    # gate.shape = s1s2s3..., s1's2's3'...
    gate = jnp.reshape(gate, [2] * (2 * num_sites)) # s1,s2,...,s1',s2',...

    # (s1) (s2) ... (sN) (s1') (s2') ... (sN') -> (s1) (s1') ... (sN) (sN')
    # [0, N, 1, N + 1, ..., N - 1, 2N - 1]
    i1 = np.arange(num_sites)
    i2 = num_sites + i1
    i  = np.zeros(2 * num_sites, dtype = int)
    i[::2] = i1
    i[1::2] = i2
    gate = jnp.transpose(gate, tuple(i))

    gate = jnp.expand_dims(gate, 0) # (l) (s1) (s1') ... (sN) (sN')

    tensors = []

    for i in range(num_sites - 1):
        lshape = gate.shape[0] # (l)
        rshape = gate.shape[3:] # (s2) (s2')...

        newshape = (gate.shape[0] * gate.shape[1] * gate.shape[2], np.prod(gate.shape[3:]))
        gate = jnp.reshape(gate, newshape)
        # (l s1 s1') (s2 s2' ...)
        u, s, vh = jnp.linalg.svd(gate, full_matrices = False)
        # u -> (l s1 s1')(D)
        # s -> (D)
        # vh -> (D)(s2 s2' ...)
        if D is not None:
            # D = jnp.min(jnp.array((D, K))) #JAX conveniently takes care of this
            # with how an out of bounds index is handled
            s = s[:D]
            u = u[:, :D]
            vh = vh[:D, :]

        u = jnp.reshape(u, (lshape, 2, 2, s.shape[0]))
        u = jnp.swapaxes(u, -1, -2)

        gate = jnp.tensordot(jnp.diag(s), vh, axes = ((1), (0)))
        # gate -> (D) (s2 s2'...)
        newshape = (gate.shape[0],) + rshape
        gate = jnp.reshape(gate, newshape) # l, s2, s2'...

        tensors.append(u)

    gate = jnp.expand_dims(gate, 2)
    tensors.append(gate)

    return tensors, s

#------------------------------------------------------------------------------#
# Common gates
#------------------------------------------------------------------------------#

@jit
def RXX(theta: float):
    # not clifford
    # D = 2
    c = jnp.cos(theta)
    s = jnp.sin(theta)

    U = jnp.array([[c, 0.0, 0.0, -1j*s],
                   [0.0, c, -1j*s, 0.0],
                   [0.0, -1j*s, c, 0.0],
                   [-1j*s, 0.0, 0.0, c]], dtype = complex)

    tensors = gate_to_MPO(U, num_sites = 2, D = 2)

    return U, tensors

@jit
def CNOT():
    U = jnp.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype = complex)

    tensors = gate_to_MPO(U, num_sites = 2, D = 2)

    return U, tensors

#------------------------------------------------------------------------------#
# Circuits and noise
#------------------------------------------------------------------------------#
@jit
def singleq_gate(gate: jnp.array, tensor: jnp.array):
    ### WARNING: INCORRECT> CONJUGATE
    res = jnp.tensordot(gate, tensor, axes = ((1), (1)))
    # u,d tdot l,u,r,d -> u,l,r,d
    res = jnp.swapaxes(res, 0, 1) # l,u,r,d
    res = jnp.tensordot(gate.conj().T, res, axes = ((0), (3)))
    # u,d tdot l,u,r,d -> d, u, l, r
    res = jnp.transpose(res, (2, 1, 3, 0))

    return res

@jit
def twoq_gate(gates: List[jnp.array], tensors: List[jnp.array]):
    ### WARNING: INCORRECT> CONJUGATE
    res = []

    num_sites = len(gates)

    for i in range(num_sites):
        res = jnp.tensordot(gate, tensor, axes = ((3), (1)))
        # gl,gu,gr,gd tdot l,u,r,d -> gl, gu, gr, l, r, d
        res = jnp.transpose(res, (0, 3, 1, 2, 4, 5)) # gl, l, gu, gr, r, d

        gate_herm_conj = jnp.transpose(gate)
        res = jnp.tensordot(gate, res, axes = ((0), (3)))
        # u,d tdot l,u,r,d -> d, u, l, r
        res = jnp.transpose(res, (2, 1, 3, 0))

    return res

@jit
def noise_layer(tensors: List[jnp.array], p: float):
    X = jnp.array([[0,1],[1,0]], dtype = complex) # u, d
    Y = jnp.array([[0,-1j],[1j,0]], dtype = complex) # u, d
    Z = jnp.array([[1,0],[0,-1]], dtype = complex) # u, d

    num_sites = len(tensors)

    for i in range(num_sites):
        tmpX = singleq_gate_on_tensor(X, tensors[i])
        tmpY = singleq_gate_on_tensor(Y, tensors[i])
        tmpZ = singleq_gate_on_tensor(Z, tensors[i])

        tensors[i] = (1 - 3 * p/4) * tensors[i] + \
                     (p/4) * (tmpX + tmpY + tmpZ)

    return tensors







# managing bond dims?
    # optimization will need fixed sizes for MPOs
# subtract/sum with negation
    # probably easiest to implement subtraction directly
# dual circuit
# vec to MPO
    # vec to operators first
    # sum with conj tr to get Herm
# tr(.^2)
    # optimal contraction order?
# local Ham to MPO
    # ??