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

def mpo_tensors_shape_from_bond_dim(N: int, D: int):
    compressed_dims = gen_compression_dims(D, N)

    tensor_shapes = []
    for i in range(N):
        if i == 0:
            tensor_shapes.append((1, 2, compressed_dims[i], 2))
        elif i == N - 1:
            tensor_shapes.append((compressed_dims[i - 1], 2, 1, 2))
        else:
            tensor_shapes.append((compressed_dims[i - 1], 2, compressed_dims[i], 2))
    return tensor_shapes

def mpo_shape(tensors):
    return [t.shape for t in tensors]

def bond_dims(tensors: List[jnp.array]):
    bdims = [tensors[0].shape[0]]

    for i in range(len(tensors) - 1):
        bdims.append(tensors[i].shape[2])
    
    bdims.append(tensors[-1].shape[2])

    return bdims

def gen_compression_dims(D: int, N: int):
    dims_max = (2 * 2) ** np.concatenate((np.arange(1, N//2 + 1),
                                     np.arange(N//2 - 1, 0, -1)))

    compressed_dims = tuple(np.where(dims_max <= D, dims_max, D))

    return compressed_dims

def full_contract(tensors: List[jnp.array]):
    """
    WARNING: memory expensive!!
    Assumes dim(l1) = dim(rN) = 1

    Not checked for random matrices. Verified through HamSumZ. 
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

@partial(jit, static_argnames = ('D',))
def left_split_lurd_tensor(tensor: jnp.array, D: int):
    """
    checked through check_canon and full_contract.
    """
    tensor_copy = jnp.swapaxes(tensor, -1, -2) # l, u, d, r
    tensor_copy = jnp.reshape(tensor_copy,
    (tensor_copy.shape[0] * tensor_copy.shape[1] * tensor_copy.shape[2],
    tensor_copy.shape[3])) # lud, r
    u, s, vh = jnp.linalg.svd(tensor_copy, full_matrices = False)

    # u.shape = (lud, D)
    # s.shape = (D,)
    # vh.shape = (D, r)

    # D = jnp.min(jnp.array((D, K))) #JAX conveniently takes care of this
    # with how an out of bounds index is handled
    s = s[:D]
    u = u[:, :D]
    vh = vh[:D, :]

    # if D is not None:
    #     # D = jnp.min(jnp.array((D, K))) #JAX conveniently takes care of this
    #     # with how an out of bounds index is handled
    #     s = s[:D]
    #     u = u[:, :D]
    #     vh = vh[:D, :]

    u = jnp.reshape(u, (tensor.shape[0], tensor.shape[1], tensor.shape[3], u.shape[1])) # l,u,d,D
    u = jnp.swapaxes(u, -1, -2) # l, u, D, d

    return u, s, vh

@partial(jit, static_argnames = ('compressed_dims',))
def left_canonicalize(tensors: List[jnp.array], compressed_dims:Tuple[int]):
    """
    Left canonicalize (leave last site uncanonicalized)
    and compress if D is specified.
    checked through check_canon and full_contract.
    """

    num_sites = len(tensors)
    # dims_max = (2 * 2) ** np.concatenate((np.arange(1, num_sites//2 + 1),
    #                                  np.arange(num_sites//2 - 1, 0, -1)))
    # num_bonds = num_sites - 1

    # compressed_dims = jnp.where(dims_max <= D, dims_max, D)

    # if D is not None:
    #     compressed_dims = jnp.where(dims_max <= D, dims_max, D)
    # else:
    #     compressed_dims = [None] * num_bonds

    for i in range(num_sites - 1):
        u, s, vh = left_split_lurd_tensor(tensors[i], D = compressed_dims[i])
        svh = jnp.matmul(jnp.diag(s), vh) # D, r
        new_right = jnp.tensordot(svh, tensors[i + 1], axes=((-1,), (0,))) # D, u, r, d

        tensors[i] = u
        tensors[i + 1] = new_right

    return tensors

@partial(jit, static_argnames = ('D',))
def right_split_lurd_tensor(tensor: jnp.array, D: int):
    """
    checked through check_canon and full_contract.
    """
    tensor_copy = jnp.swapaxes(tensor, -1, -2) # l, u, d, r
    tensor_copy = jnp.reshape(tensor_copy,
    (tensor_copy.shape[0],
    tensor_copy.shape[1] * tensor_copy.shape[2] * tensor_copy.shape[3])) # l, udr
    u, s, vh = jnp.linalg.svd(tensor_copy, full_matrices = False)

    # u.shape = (l, D)
    # s.shape = (D,)
    # vh.shape = (D, udr)

    # D = jnp.min(jnp.array((D, K))) #JAX conveniently takes care of this
    # with how an out of bounds index is handled
    s = s[:D]
    u = u[:, :D]
    vh = vh[:D, :]

    # if D is not None:
    #     # D = jnp.min(jnp.array((D, K))) #JAX conveniently takes care of this
    #     # with how an out of bounds index is handled
    #     s = s[:D]
    #     u = u[:, :D]
    #     vh = vh[:D, :]

    vh = jnp.reshape(vh, (vh.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[2])) # D, u, d, r
    vh = jnp.swapaxes(vh, -1, -2) # D, u, r, d

    return u, s, vh

@partial(jit, static_argnames = ('compressed_dims',))
def right_canonicalize(tensors: List[jnp.array], compressed_dims:Tuple[int]):
    """
    Right canonicalize (leave first site uncanonicalized)
    and compress if D is specified.
    checked through check_canon and full_contract.
    """

    num_sites = len(tensors)
    # dims_max = (2 * 2) ** np.concatenate((np.arange(1, num_sites//2 + 1),
    #                                  np.arange(num_sites//2 - 1, 0, -1)))
    # num_bonds = num_sites - 1

    # if D is not None:
    #     compressed_dims = jnp.where(dims_max <= D, dims_max, D)
    # else:
    #     compressed_dims = [None] * num_bonds

    for i in range(num_sites - 1, 0, -1):
        u, s, vh = right_split_lurd_tensor(tensors[i], D = compressed_dims[i - 1])
        us = jnp.matmul(u, jnp.diag(s)) # l, D
        new_left = jnp.tensordot(us, tensors[i - 1], axes=((0,), (2,))) #  D l u d
        new_left = jnp.transpose(new_left, [1, 2, 0, 3])

        tensors[i] = vh
        tensors[i - 1] = new_left

    return tensors

@partial(jit, static_argnames = ('canon',))
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

    checked through full_contract.
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
    checked through full_contract.
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

@jit
def trace_two_MPOs(A_tensors: List[jnp.array], B_tensors: List[jnp.array]):
    """
    checked through full_contract.
    """
    # bond_dims = [tensor.shape[2] for tensor in tensors[:-1]]
    num_sites = len(B_tensors)
    tooth = B_tensors[0]

    for i in range(0, num_sites - 1):
        # Di = bond_dims[i]
        # zip = jnp.zeros((Di, Di), dtype = complex)
        zip = jnp.tensordot(tooth, A_tensors[i], axes = ((0,1,3), (0,3,1)))
        # Di (tooth), Di (tensor)
        tooth = jnp.tensordot(zip, B_tensors[i+1], axes = ((0,),(0,)))
        # Di (tooth), Di (tensor) tdot l,u,r,d -> Di (tensor), u,r,d

    res = jnp.tensordot(tooth, A_tensors[-1], axes = ((0,1,2,3), (0,3,2,1)))

    return res

def conjugate_mpo(mpo_tensors: List[jnp.array]):
    conj_mpo_tensors = []
    num_sites = len(mpo_tensors)

    for i in range(num_sites):
        tensor = mpo_tensors[i]
        tensor = jnp.conjugate(tensor)
        tensor = jnp.swapaxes(tensor, 1, 3)
        conj_mpo_tensors.append(tensor)

    return conj_mpo_tensors

def negate_mpo(mpo_tensors: List[jnp.array]):
    return [-tensor for tensor in mpo_tensors] 

def scale_mpo(mpo_tensors: List[jnp.array], s: float):
    scaled_tensors = [tensor for tensor in mpo_tensors] 
    scaled_tensors[0] = s * scaled_tensors[0]
    return scaled_tensors

def hermitize_mpo(mpo_tensors: List[jnp.array]):
    minus_conj_mpo_tensors = negate_mpo(conjugate_mpo(mpo_tensors))
    herm_mpo_tensors = subtract_MPO(mpo_tensors, minus_conj_mpo_tensors)
    herm_mpo_tensors = scale_mpo(herm_mpo_tensors, 0.5)

    return herm_mpo_tensors

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
    """
    this is just general tensor decomposition into MPO. 
    should be the inverse of full_contract. 

    checked through full_contract. 
    """
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
def HaarSQ(key: jnp.array):
    N = 2

    key, subkey = jax.random.split(key)
    A = jax.random.normal(subkey, (N, N))
    key, subkey = jax.random.split(key)
    B = jax.random.normal(subkey, (N, N))
    Z = A + 1j * B

    Q, R = jnp.linalg.qr(Z)
    Lambda = jnp.diag(jnp.diag(R)/jnp.abs(jnp.diag(R)))

    return jnp.matmul(Q, Lambda), key

@jit
def RX(theta:float):
    X = jnp.array([[0, 1],[1, 0]], dtype = complex)
    I = jnp.array([[1, 0],[0, 1]], dtype = complex)

    c = jnp.cos(theta/2)
    s = jnp.sin(theta/2)

    return c * I - 1j * s * X

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

    tensors, _ = gate_to_MPO(U, num_sites = 2, D = 2)

    return U, tensors

@jit
def CNOT():
    U = jnp.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype = complex)

    tensors, _ = gate_to_MPO(U, num_sites = 2, D = 2)

    return U, tensors

#------------------------------------------------------------------------------#
# Circuits and noise
#------------------------------------------------------------------------------#
@jit
def singleq_gate(gate: jnp.array, tensor: jnp.array):
    """checked."""

    res = jnp.tensordot(gate, tensor, axes = ((1), (1)))
    # u,d tdot l,u,r,d -> u,l,r,d
    res = jnp.swapaxes(res, 0, 1) # l,u,r,d
    res = jnp.tensordot(gate.conj().T, res, axes = ((0), (3)))
    # u,d tdot l,u,r,d -> d,l,u,r
    res = jnp.transpose(res, (1, 2, 3, 0))

    return res

@jit
def twoq_gate(gates: List[jnp.array], tensors: List[jnp.array]):
    """checked."""
    res_tensors = []

    num_sites = len(gates)

    for i in range(num_sites):
        gate = gates[i]
        tensor = tensors[i]

        res = jnp.tensordot(gate, tensor, axes = ((3), (1)))
        # gl,gu,gr,gd tdot l,u,r,d -> gl, gu, gr, l, r, d
        res = jnp.transpose(res, (0, 3, 1, 2, 4, 5)) # gl, l, gu, gr, r, d
        # gl, l, gu, gr, r, d -> (gl, l), gu, (gr, r), d 
        res = jnp.reshape(res, (res.shape[0] * res.shape[1], res.shape[2], res.shape[3] * res.shape[4], res.shape[5]))

        gate_herm_conj = jnp.transpose(gate.conj(), (0, 3, 2, 1))
        # gl,gu,gr,gd -> gl,gd,gr,gu*
        res = jnp.tensordot(gate_herm_conj, res, axes = ((1), (3)))
        # gl,gd,gr,gu* tdot l,u,r,d -> gl,gr,gu,l,u,r
        res = jnp.transpose(res, (0, 3, 4, 1, 5, 2)) # gl, l, u, gr, r, gu
        res = jnp.reshape(res, (res.shape[0] * res.shape[1], res.shape[2], res.shape[3] * res.shape[4], res.shape[5]))

        res_tensors.append(res)

    return res_tensors

@jit
def noise_layer(tensors: List[jnp.array], p: float):
    X = jnp.array([[0,1],[1,0]], dtype = complex) # u, d
    Y = jnp.array([[0,-1j],[1j,0]], dtype = complex) # u, d
    Z = jnp.array([[1,0],[0,-1]], dtype = complex) # u, d

    num_sites = len(tensors)

    for i in range(num_sites):
        tmpX = singleq_gate(X, tensors[i])
        tmpY = singleq_gate(Y, tensors[i])
        tmpZ = singleq_gate(Z, tensors[i])

        tensors[i] = (1 - 3 * p/4) * tensors[i] + \
                     (p/4) * (tmpX + tmpY + tmpZ)

    return tensors

#------------------------------------------------------------------------------#
# Circuits, initialization
#------------------------------------------------------------------------------#

class SumZ_RXX():
    def __init__(self, N: int, d: int, p: float, key):
        self.N = N
        self.d = d
        self.p = p
        self.H, self.H_tensors = HamSumZ(self.N)

        self.site_tuple_list = list(zip(range(0, self.N, 2), range(1, self.N, 2))) + list(zip(range(1, self.N, 2), range(2, self.N, 2)))
        self.site_tuple_list_inverted = list(zip(range(1, self.N, 2), range(2, self.N, 2))) + list(zip(range(0, self.N, 2), range(1, self.N, 2)))

        theta_half = jax.random.normal(key, shape = (len(self.site_tuple_list), self.d//2))
        self.theta = jnp.column_stack((theta_half, -jnp.roll(theta_half[:, ::-1], (self.N - 2)//2, axis = 0)))   

        self.psi_init = jnp.zeros((2 ** self.N,), dtype = complex)
        self.psi_init = self.psi_init.at[-1].set(1.0)

        psi_init_local_tensor = jnp.array([[0, 0],[0, 1]], dtype = complex)
        psi_init_local_tensor = jnp.expand_dims(psi_init_local_tensor, axis = (0,2))
        self.psi_init_tensors = [psi_init_local_tensor] * self.N

        self.X = jnp.array([[0,1],[1,0]], dtype = complex) # u, d
        self.Y = jnp.array([[0,-1j],[1j,0]], dtype = complex) # u, d
        self.Z = jnp.array([[1,0],[0,-1]], dtype = complex) # u, d

    def primal_noisy(self):
        rho_init = jnp.outer(self.psi_init, self.psi_init.conj().T)

        rho_after_step = rho_init

        for layer_num in range(0, self.d):
            if layer_num < self.d//2:
                gate_tuples = self.site_tuple_list
            else:
                gate_tuples = self.site_tuple_list_inverted

            # nn unitaries
            for i_tuple, site_tuple in enumerate(gate_tuples):
                theta = self.theta.at[i_tuple, layer_num].get()

                U_2site, _ = RXX(theta)

                dim_left = 2 ** site_tuple[0]
                dim_right = 2 ** (self.N - site_tuple[-1] - 1)

                identity_left = jnp.identity(dim_left)
                identity_right = jnp.identity(dim_right)

                U_2site_full = jnp.kron(identity_left, jnp.kron(U_2site, identity_right))

                rho_after_step = jnp.matmul(U_2site_full, jnp.matmul(rho_after_step, U_2site_full.conj().T)) 

            # noise
            for site in range(self.N):
                dim_left = 2 ** site
                dim_right = 2 ** (self.N - site - 1)

                identity_left = jnp.identity(dim_left)
                identity_right = jnp.identity(dim_right)

                X_full = jnp.kron(identity_left, jnp.kron(self.X, identity_right))
                Y_full = jnp.kron(identity_left, jnp.kron(self.Y, identity_right))
                Z_full = jnp.kron(identity_left, jnp.kron(self.Z, identity_right))

                rho_after_step = (1 - 3 * self.p/4) * rho_after_step \
                + (self.p/4) * (jnp.matmul(X_full, jnp.matmul(rho_after_step, X_full)) + \
                                jnp.matmul(Y_full, jnp.matmul(rho_after_step, Y_full)) + \
                                jnp.matmul(Z_full, jnp.matmul(rho_after_step, Z_full)))

        return jnp.trace(jnp.matmul(self.H.full_ham(), rho_after_step))

    def dual_unitary_layer_on_mpo(self, layer_num: int, mpo_tensors: List[jnp.array]):
        if layer_num < self.d//2:
            gate_tuples = self.site_tuple_list
        else:
            gate_tuples = self.site_tuple_list_inverted

        for i_tuple, gate_tuple in enumerate(gate_tuples):
            theta = self.theta.at[i_tuple, layer_num].get() 
            _, gate_tensors = RXX(-theta) #conjugate the gate

            res_tensors = twoq_gate(gate_tensors, [mpo_tensors[site] for site in gate_tuple])
            mpo_tensors[gate_tuple[0]] = res_tensors[0]
            mpo_tensors[gate_tuple[1]] = res_tensors[1]
        
        return mpo_tensors
    
    def noisy_dual_layer_on_mpo(self, layer_num: int, mpo_tensors: List[jnp.array]):
        mpo_tensors = noise_layer(mpo_tensors, self.p)
        mpo_tensors = self.dual_unitary_layer_on_mpo(layer_num, mpo_tensors)

        return mpo_tensors
    
    def init_mpo(self, D: int):
        mpo_tensors = self.H_tensors

        for i in range(self.d-1, -1, -1):
            mpo_tensors = self.noisy_dual_layer_on_mpo(i, mpo_tensors)
            
            # canonicalize
            compressed_dims = tuple(bond_dims(mpo_tensors)[1:-1])
            mpo_tensors = right_canonicalize(tensors = mpo_tensors, compressed_dims = compressed_dims)

            # compress
            compressed_dims = gen_compression_dims(D, self.N)
            mpo_tensors = left_canonicalize(tensors = mpo_tensors, compressed_dims = compressed_dims)

        return mpo_tensors
    

# #------------------------------------------------------------------------------#
# # Vectorization and unvectorization
# #------------------------------------------------------------------------------#

# def tensor_shape_to_vec_shape(t_shape: Tuple[int]):
#     l, u, r, d = t_shape
#     s = u # assuming u = d = s

#     vec_shape = (l*r*s*s,)
#     return vec_shape

# def vec_to_herm_tensor_bodyfun(i, args):
#     vec, tensor, upper_diag_indices, diag_indices, ones_l, ones_r, zeros_s = args

#     l = ones_l.shape[0]
#     r = ones_r.shape[0]
#     s = zeros_s.shape[0]

#     diag_vec = jax.lax.dynamic_slice_in_dim(vec, i*s*s, s)
#     real_vec = jax.lax.dynamic_slice_in_dim(vec, i*s*s + s, s * (s - 1)//2)
#     imag_vec = jax.lax.dynamic_slice_in_dim(vec, i*s*s + s * (s + 1)//2, s * (s - 1)//2)
    
#     tensor_i = zeros_s
#     tensor_i = tensor_i.at[upper_diag_indices].set(real_vec + 1j * imag_vec)
#     tensor_i = tensor_i + tensor_i.conj().T
#     tensor_i = tensor_i.at[diag_indices].set(diag_vec)

#     tensor = tensor.at[i, :, :].set(tensor_i)

#     return (vec, tensor, upper_diag_indices, diag_indices, ones_l, ones_r, zeros_s)

# @partial(jit, static_argnames = ('shape',))
# def vec_to_herm_tensor(vec: jnp.array, shape: Tuple[int]):
#     l, s, r, _ = shape # u, d = s 

#     # vec = jnp.reshape(vec, (l * r, s * s))

#     tensor = jnp.zeros((l * r, s, s), dtype = complex)
#     zeros_s = jnp.zeros((s,s), dtype = complex)

#     ones_l = jnp.ones((l,))
#     ones_r = jnp.ones((l,))

#     upper_diag_indices = jnp.triu_indices(s, 1)
#     diag_indices = jnp.diag_indices(s)

#     init_args = (vec, tensor, upper_diag_indices, diag_indices, ones_l, ones_r, zeros_s)

#     _, tensor, _, _, _, _, _ = jax.lax.fori_loop(0, l*r, vec_to_herm_tensor_bodyfun, init_args)

#     # for i in range(l * r):
#     #     diag_vec = vec[i, :s]
#     #     real_vec = vec[i, s : s*(s+1)//2]
#     #     imag_vec = vec[i, s*(s+1)//2:]

#     #     tensor_i = zeros_s
#     #     tensor_i = tensor_i.at[upper_diag_indices].set(real_vec + 1j * imag_vec)
#     #     tensor_i = tensor_i + tensor_i.conj().T
#     #     tensor_i = tensor_i.at[diag_indices].set(diag_vec)

#     #     tensor = tensor.at[i, :, :].set(tensor_i)
    
#     tensor = jnp.reshape(tensor, (l, r, s, s))
#     tensor = jnp.transpose(tensor, (0, 2, 1, 3))

#     return tensor

# @partial(jit, static_argnames = ('shape',))
# def vec_to_herm_mpo(vec: jnp.array, shape: Tuple[Tuple[int]]):
#     t_vec_lengths = [s[0] * s[1] * s[2] * s[3] for s in shape]
#     t_vec_slice_indices = [0] + list(np.cumsum(t_vec_lengths))

#     mpo_tensors = []

#     for i, t_shape in enumerate(shape):
#         t_vec = vec[t_vec_slice_indices[i]:t_vec_slice_indices[i + 1]]
#         mpo_tensors.append(vec_to_herm_tensor(t_vec, t_shape))

#     return mpo_tensors

# #------------------------------------------------------------------------------#
# # Methods for SOCP
# #------------------------------------------------------------------------------#

# def gen_overlap_ham(mpo_tensors: List[jnp.array], site_idx: int):

#     """
#     !! edge cases
#     !! check
#     """
#     num_sites = len(mpo_tensors)

#     tooth = mpo_tensors[0]
#     for i in range(0, site_idx - 1):
#         zip = jnp.tensordot(tooth, mpo_tensors[i], axes = ((0,1,3), (0,3,1)))
#         # Di (tooth), Di (tensor)
#         tooth = jnp.tensordot(zip, mpo_tensors[i+1], axes = ((0,),(0,)))
#         # Di (tooth), Di (tensor) tdot l,u,r,d -> Di (tensor), u,r,d

#     left_mat = jnp.tensordot(tooth, mpo_tensors[site_idx - 1], axes = ((0,1,3), (0,3,1)))
#     # Di (tooth), Di (tensor) = l l'

#     tooth = mpo_tensors[-1]
#     for i in range(num_sites - 1, site_idx + 1, -1):
#         zip = jnp.tensordot(tooth, mpo_tensors[i], axes = ((2,1,3), (2,3,1)))
#         # Di (tooth), Di (tensor)
#         tooth = jnp.tensordot(zip, mpo_tensors[i-1], axes = ((0,),(2,)))
#         # Di (tooth), Di (tensor) tdot l,u,r,d -> l, u, Di (tensor),d

#     right_mat = jnp.tensordot(tooth, mpo_tensors[site_idx + 1], axes = ((2,1,3), (2,3,1)))
#     # Di (tooth), Di (tensor) = r r'

    






# def gen_overlap_ham_layer(mpo_tensors: List[jnp.array], layer_tensors: List[jnp.array], site: int):
    




# managing bond dims?
    # optimization will need fixed sizes for MPOs
# dual circuit
# vec to MPO
    # vec to operators first
    # sum with conj tr to get Herm
# local Ham to MPO
    # ??