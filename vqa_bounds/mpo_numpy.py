"""
V1:
    sigma_d = -H.
    lambda_d = 0.
    OBC.

    TODO:
        1. test canonicalization with D < actual dim
        2. lax fori_loops?
"""

from typing import List, Tuple, Callable, Dict
import copy

import numpy as np

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

            identity_left = np.identity(dim_left)
            identity_right = np.identity(dim_right)

            full_mat += np.kron(identity_left, np.kron(term, identity_right))

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

def bond_dims(tensors: List[np.array]):
    bdims = [tensors[0].shape[0]]

    for i in range(len(tensors) - 1):
        bdims.append(tensors[i].shape[2])
    
    bdims.append(tensors[-1].shape[2])

    return bdims

def gen_compression_dims(D: int, N: int):
    """
    only implemented for N = even
    """
    dims_max = (2 * 2) ** np.concatenate((np.arange(1, N//2 + 1),
                                     np.arange(N//2 - 1, 0, -1)))

    compressed_dims = tuple(np.where(dims_max <= D, dims_max, D))

    return compressed_dims


def full_contract(tensors: List[np.array]):
    """
    WARNING: memory expensive!!
    Assumes dim(l1) = dim(rN) = 1

    Not checked for random matrices. Verified through HamSumZ. 
    """
    num_sites = len(tensors)

    res = tensors[0]

    for i in range(num_sites-1):
        res = np.tensordot(res, tensors[i + 1], axes=((-2,), (0,))) # (l) (u) (d) (u') (r') (d')

    res = np.swapaxes(res, -1, -2) # l, u1, d1, u2, d2, ..., dN, rN
    res = np.squeeze(res) # u1, d1, u2, d2, ..., dN

    # (u1) (d1) (u2) (d2) ... (uN) (dN) -> (u1) (u2) ... (uN) (d1) (d2) ... (dN)
    # [0, 2, 4, ..., 2N - 2, 1, 3, ..., 2N - 1]
    i1 = np.arange(0, 2 * num_sites - 1, 2)
    i2 = np.arange(1, 2 * num_sites, 2)
    i = np.concatenate((i1, i2))
    res = np.transpose(res, tuple(i))

    res = np.reshape(res, (2**num_sites, 2**num_sites))

    return res

def left_split_lurd_tensor(tensor: np.array, D: int):
    """
    checked through check_canon and full_contract.
    """
    tensor_copy = np.swapaxes(tensor, -1, -2) # l, u, d, r
    tensor_copy = np.reshape(tensor_copy,
    (tensor_copy.shape[0] * tensor_copy.shape[1] * tensor_copy.shape[2],
    tensor_copy.shape[3])) # lud, r
    u, s, vh = np.linalg.svd(tensor_copy, full_matrices = False)

    # u.shape = (lud, D)
    # s.shape = (D,)
    # vh.shape = (D, r)

    D = np.min(np.array((D, len(s)))) 
    # #JAX conveniently takes care of this
    # with how an out of bounds index is handled
    s = s[:D]
    u = u[:, :D]
    vh = vh[:D, :]

    # if D is not None:
    #     # D = np.min(np.array((D, K))) #JAX conveniently takes care of this
    #     # with how an out of bounds index is handled
    #     s = s[:D]
    #     u = u[:, :D]
    #     vh = vh[:D, :]

    u = np.reshape(u, (tensor.shape[0], tensor.shape[1], tensor.shape[3], u.shape[1])) # l,u,d,D
    u = np.swapaxes(u, -1, -2) # l, u, D, d

    return u, s, vh

# @partial(jit, static_argnames = ('compressed_dims',))

def left_canonicalize(tensors: List[np.array], compressed_dims:Tuple[int]):
    """
    Left canonicalize (leave last site uncanonicalized)
    and compress if D is specified.
    checked through check_canon and full_contract.
    """

    num_sites = len(tensors)
    # dims_max = (2 * 2) ** np.concatenate((np.arange(1, num_sites//2 + 1),
    #                                  np.arange(num_sites//2 - 1, 0, -1)))
    # num_bonds = num_sites - 1

    # compressed_dims = np.where(dims_max <= D, dims_max, D)

    # if D is not None:
    #     compressed_dims = np.where(dims_max <= D, dims_max, D)
    # else:
    #     compressed_dims = [None] * num_bonds

    for i in range(num_sites - 1):
        u, s, vh = left_split_lurd_tensor(tensors[i], D = compressed_dims[i])
        svh = np.diag(s) @ vh # D, r
        new_right = np.tensordot(svh, tensors[i + 1], axes=((-1,), (0,))) # D, u, r, d

        tensors[i] = u
        tensors[i + 1] = new_right

    return tensors


def right_split_lurd_tensor(tensor: np.array, D: int):
    """
    checked through check_canon and full_contract.
    """
    tensor_copy = np.swapaxes(tensor, -1, -2) # l, u, d, r
    tensor_copy = np.reshape(tensor_copy,
    (tensor_copy.shape[0],
    tensor_copy.shape[1] * tensor_copy.shape[2] * tensor_copy.shape[3])) # l, udr
    u, s, vh = np.linalg.svd(tensor_copy, full_matrices = False)

    # u.shape = (l, D)
    # s.shape = (D,)
    # vh.shape = (D, udr)

    D = np.min(np.array((D, len(s)))) 
    # #JAX conveniently takes care of this
    # with how an out of bounds index is handled
    s = s[:D]
    u = u[:, :D]
    vh = vh[:D, :]

    # if D is not None:
    #     # D = np.min(np.array((D, K))) #JAX conveniently takes care of this
    #     # with how an out of bounds index is handled
    #     s = s[:D]
    #     u = u[:, :D]
    #     vh = vh[:D, :]

    vh = np.reshape(vh, (vh.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[2])) # D, u, d, r
    vh = np.swapaxes(vh, -1, -2) # D, u, r, d

    return u, s, vh


def right_canonicalize(tensors: List[np.array], compressed_dims:Tuple[int]):
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
    #     compressed_dims = np.where(dims_max <= D, dims_max, D)
    # else:
    #     compressed_dims = [None] * num_bonds

    for i in range(num_sites - 1, 0, -1):
        u, s, vh = right_split_lurd_tensor(tensors[i], D = compressed_dims[i - 1])
        us = u @ np.diag(s) # l, D
        new_left = np.tensordot(us, tensors[i - 1], axes=((0,), (2,))) #  D l u d
        new_left = np.transpose(new_left, [1, 2, 0, 3])

        tensors[i] = vh
        tensors[i - 1] = new_left

    return tensors

# @partial(jit, static_argnames = ('canon',))
def check_canon(tensors: List[np.array], canon = "left"):
    num_sites = len(tensors)
    norm_list = []

    if canon == "left":
        for i in range(num_sites - 1):
            check = np.tensordot(tensors[i], tensors[i].conj(),
                    axes = ((0, 1, 3), (0, 1, 3)))
            norm_list.append(np.linalg.norm(check - np.identity(check.shape[0])))
    elif canon == "right":
        for i in range(num_sites - 1, 0, -1):
            check = np.tensordot(tensors[i], tensors[i].conj(),
                    axes = ((2, 1, 3), (2, 1, 3)))
            norm_list.append(np.linalg.norm(check - np.identity(check.shape[0])))
    else:
        raise ValueError
    return norm_list


def trace_MPO_squared(tensors: List[np.array]):
    """
    checked through full_contract.
    """
    # bond_dims = [tensor.shape[2] for tensor in tensors[:-1]]
    num_sites = len(tensors)
    tooth = tensors[0]

    for i in range(0, num_sites - 1):
        # Di = bond_dims[i]
        # zip = np.zeros((Di, Di), dtype = complex)
        zip = np.tensordot(tooth, tensors[i], axes = ((0,1,3), (0,3,1)))
        # Di (tooth), Di (tensor)
        tooth = np.tensordot(zip, tensors[i+1], axes = ((0,),(0,)))
        # Di (tooth), Di (tensor) tdot l,u,r,d -> Di (tensor), u,r,d

    res = np.tensordot(tooth, tensors[-1], axes = ((0,1,2,3), (0,3,2,1)))

    return res

def trace_two_MPOs(A_tensors: List[np.array], B_tensors: List[np.array]):
    """
    checked through full_contract.
    """
    # bond_dims = [tensor.shape[2] for tensor in tensors[:-1]]
    num_sites = len(B_tensors)
    tooth = B_tensors[0]

    for i in range(0, num_sites - 1):
        # Di = bond_dims[i]
        # zip = np.zeros((Di, Di), dtype = complex)
        zip = np.tensordot(tooth, A_tensors[i], axes = ((0,1,3), (0,3,1)))
        # Di (tooth), Di (tensor)
        tooth = np.tensordot(zip, B_tensors[i+1], axes = ((0,),(0,)))
        # Di (tooth), Di (tensor) tdot l,u,r,d -> Di (tensor), u,r,d

    res = np.tensordot(tooth, A_tensors[-1], axes = ((0,1,2,3), (0,3,2,1)))

    return res

def conjugate_mpo(mpo_tensors: List[np.array]):
    conj_mpo_tensors = []
    num_sites = len(mpo_tensors)

    for i in range(num_sites):
        tensor = mpo_tensors[i]
        tensor = np.conjugate(tensor)
        tensor = np.swapaxes(tensor, 1, 3)
        conj_mpo_tensors.append(tensor)

    return conj_mpo_tensors


def scale_mpo(mpo_tensors: List[np.array], s: float):
    scaled_tensors = [tensor for tensor in mpo_tensors] 
    scaled_tensors[0] = s * scaled_tensors[0]
    return scaled_tensors

#------------------------------------------------------------------------------#
# Models and circuits
#------------------------------------------------------------------------------#

def HamSumZ(num_sites: int):
    """
    Returns Hamiltonian and MPO representations of \sum Z.
    """
    H = Hamiltonian(num_sites)

    I = np.array([[1, 0],[0, 1]], dtype = complex)
    Z = np.array([[1, 0],[0, -1]], dtype = complex)

    for site_tuple in H.site_tuple_list:
        H.terms[site_tuple] = np.kron(I, Z)

    H.terms[(0,1)] += np.kron(Z, I)

    arrs = []

    op = np.block([Z, I]) # shape = (bs) (b' s')) # b is bond, s is physical
    op = np.reshape(op, (1, 2, 2, 2)) # shape = (b) (s) (b') (s')
    arrs.append(op)

    op = np.block([[I, np.zeros((2,2))], [Z, I]]) # shape = (bs) (b' s'))
    op = np.reshape(op, (2, 2, 2, 2)) # shape = (b) (s) (b') (s')
    arrs += [op] * (num_sites - 2)

    op = np.row_stack((I, Z))
    op = np.reshape(op, (2, 2, 1, 2)) # shape = (b) (s) (b') (s')
    arrs.append(op)

    return H, arrs


def gate_to_MPO(gate: np.array, num_sites: int, D: int = None):
    """
    this is just general tensor decomposition into MPO. 
    should be the inverse of full_contract. 

    checked through full_contract. 
    """
    # gate.shape = s1s2s3..., s1's2's3'...
    # [2] * (2 * num_sites)
    gate = np.reshape(gate, [2] * (2 * num_sites)) # s1,s2,...,s1',s2',...

    # (s1) (s2) ... (sN) (s1') (s2') ... (sN') -> (s1) (s1') ... (sN) (sN')
    # [0, N, 1, N + 1, ..., N - 1, 2N - 1]
    i1 = np.arange(num_sites)
    i2 = num_sites + i1
    i  = np.zeros(2 * num_sites, dtype = int)
    i[::2] = i1
    i[1::2] = i2
    gate = np.transpose(gate, tuple(i))

    gate = np.expand_dims(gate, 0) # (l) (s1) (s1') ... (sN) (sN')

    tensors = []

    for i in range(num_sites - 1):
        lshape = gate.shape[0] # (l)
        rshape = gate.shape[3:] # (s2) (s2')...

        newshape = (gate.shape[0] * gate.shape[1] * gate.shape[2], np.prod(gate.shape[3:]))
        gate = np.reshape(gate, newshape)
        # (l s1 s1') (s2 s2' ...)
        u, s, vh = np.linalg.svd(gate, full_matrices = False)
        # u -> (l s1 s1')(D)
        # s -> (D)
        # vh -> (D)(s2 s2' ...)
        if D is not None:
            D = np.min(np.array((D, len(s)))) 
            s = s[:D]
            u = u[:, :D]
            vh = vh[:D, :]

        u = np.reshape(u, (lshape, 2, 2, s.shape[0]))
        u = np.swapaxes(u, -1, -2)

        gate = np.tensordot(np.diag(s), vh, axes = ((1), (0)))
        # gate -> (D) (s2 s2'...)
        newshape = (gate.shape[0],) + rshape
        gate = np.reshape(gate, newshape) # l, s2, s2'...

        tensors.append(u)

    gate = np.expand_dims(gate, 2)
    tensors.append(gate)

    return tensors, s

#------------------------------------------------------------------------------#
# Common gates
#------------------------------------------------------------------------------#

 
def HaarSQ():
    N = 2

    A = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            A[i,j] = np.random.normal()
    
    B = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            B[i,j] = np.random.normal()

    Z = A + 1j * B

    Q, R = np.linalg.qr(Z)
    Lambda = np.diag(np.diag(R)/np.abs(np.diag(R)))

    return Q @ Lambda
 
def SWAP():
    U = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]], dtype = complex)

    tensors, _ = gate_to_MPO(U, num_sites = 2)

    return U, tensors


def RX(theta:float):
    X = np.array([[0, 1],[1, 0]], dtype = complex)
    I = np.array([[1, 0],[0, 1]], dtype = complex)

    c = np.cos(theta/2)
    s = np.sin(theta/2)

    return c * I - 1j * s * X


def RXX(theta: float):
    # not clifford
    # D = 2
    # period = 2 \pi
    c = np.cos(theta)
    s = np.sin(theta)

    U = np.array([[c, 0.0, 0.0, -1j*s],
                   [0.0, c, -1j*s, 0.0],
                   [0.0, -1j*s, c, 0.0],
                   [-1j*s, 0.0, 0.0, c]], dtype = complex)

    tensors, _ = gate_to_MPO(U, num_sites = 2, D = 2)

    return U, tensors


def CNOT():
    U = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype = complex)

    tensors, _ = gate_to_MPO(U, num_sites = 2, D = 2)

    return U, tensors

#------------------------------------------------------------------------------#
# Circuits and noise
#------------------------------------------------------------------------------#

def singleq_gate(gate: np.array, tensor: np.array):
    """checked.
    U \rho U^{\dagger}
    """

    res = np.tensordot(gate, tensor, axes = ((1), (1)))
    # u,d tdot l,u,r,d -> u,l,r,d
    res = np.swapaxes(res, 0, 1) # l,u,r,d
    res = np.tensordot(gate.conj().T, res, axes = ((0), (3)))
    # u,d tdot l,u,r,d -> d,l,u,r
    res = np.transpose(res, (1, 2, 3, 0))

    return res


def twoq_gate(gates: List[np.array], tensors: List[np.array]):
    """checked.
    U \rho U^{\dagger}
    """
    res_tensors = []

    num_sites = len(gates)

    for i in range(num_sites):
        gate = gates[i]
        tensor = tensors[i]

        res = np.tensordot(gate, tensor, axes = ((3), (1)))
        # gl,gu,gr,gd tdot l,u,r,d -> gl, gu, gr, l, r, d
        res = np.transpose(res, (0, 3, 1, 2, 4, 5)) # gl, l, gu, gr, r, d
        # gl, l, gu, gr, r, d -> (gl, l), gu, (gr, r), d 
        res = np.reshape(res, (res.shape[0] * res.shape[1], res.shape[2], res.shape[3] * res.shape[4], res.shape[5]))

        gate_herm_conj = np.transpose(gate.conj(), (0, 3, 2, 1))
        # gl,gu,gr,gd -> gl,gd,gr,gu*
        res = np.tensordot(gate_herm_conj, res, axes = ((1), (3)))
        # gl,gd,gr,gu* tdot l,u,r,d -> gl,gr,gu,l,u,r
        res = np.transpose(res, (0, 3, 4, 1, 5, 2)) # gl, l, u, gr, r, gu
        res = np.reshape(res, (res.shape[0] * res.shape[1], res.shape[2], res.shape[3] * res.shape[4], res.shape[5]))

        res_tensors.append(res)

    return res_tensors


def noise_layer(tensors: List[np.array], p: float):
    X = np.array([[0,1],[1,0]], dtype = complex) # u, d
    Y = np.array([[0,-1j],[1j,0]], dtype = complex) # u, d
    Z = np.array([[1,0],[0,-1]], dtype = complex) # u, d

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
    def __init__(self, N: int, d: int, p: float, theta: float, seed: int):
        self.N = N
        self.d = d
        self.d_compute = 1
        self.depth = self.d_compute + 2 * self.d 
        self.p = p
        self.H, self.H_tensors = HamSumZ(self.N)

        np.random.seed(seed)

        self.theta = theta

        self.site_tuple_list = list(zip(range(0, self.N, 2), range(1, self.N, 2))) + list(zip(range(1, self.N, 2), range(2, self.N, 2)))
        self.site_tuple_list_inverted = list(zip(range(1, self.N, 2), range(2, self.N, 2))) + list(zip(range(0, self.N, 2), range(1, self.N, 2)))

        if N <= 6:
            self.psi_init = np.zeros((2 ** self.N,), dtype = complex)
            self.psi_init[-1] = 1.0

        psi_init_local_tensor = np.array([[0, 0],[0, 1]], dtype = complex)
        psi_init_local_tensor = np.expand_dims(psi_init_local_tensor, axis = (0,2))
        self.psi_init_tensors = [psi_init_local_tensor] * self.N

        id_local = np.array([[1, 0],[0, 1]], dtype = complex)
        id_local = np.expand_dims(id_local, axis = (0,2))
        self.identity_mpo = [id_local] * self.N

        self.X = np.array([[0,1],[1,0]], dtype = complex) # u, d
        self.Y = np.array([[0,-1j],[1j,0]], dtype = complex) # u, d
        self.Z = np.array([[1,0],[0,-1]], dtype = complex) # u, d

        self.sq_gates = {}

        for i_d in range(self.d_compute):
            for i in range(self.N):
                self.sq_gates[i_d, i] = HaarSQ()

        for i_d in range(self.d_compute, self.d_compute + self.d):
            for i in range(N):
                U = HaarSQ()
                self.sq_gates[i_d, i] = U
                self.sq_gates[self.d_compute + 2 * self.d - 1 - (i_d - self.d_compute), i] = U.conj().T

        self.U2q, self.U2q_tensors = RXX(self.theta)
        self.U2q_dagger, self.U2q_dagger_tensors = RXX(-self.theta)
        
        self.target_H()

        q = 1 - self.p
        q_powers = np.array([q**i for i in range(self.depth)])

        self.entropy_bounds = self.N * self.p * np.log(2) * \
                np.array([np.sum(q_powers[:i+1]) for i in range(self.depth)])
        self.purity_bounds = np.exp(-self.entropy_bounds)

    def primal_noisy(self):
        rho_init = np.outer(self.psi_init, self.psi_init.conj().T)

        rho_after_step = rho_init

        for layer_num in range(self.depth):
            # two qubit gates
            if layer_num < self.d_compute + self.d: 
                gate_tuples = self.site_tuple_list
                U_2site = self.U2q

                for i_tuple, site_tuple in enumerate(gate_tuples):
                    dim_left = 2 ** site_tuple[0]
                    dim_right = 2 ** (self.N - site_tuple[-1] - 1)

                    identity_left = np.identity(dim_left)
                    identity_right = np.identity(dim_right)

                    U_2site_full = np.kron(identity_left, np.kron(U_2site, identity_right))
                    rho_after_step = U_2site_full @ rho_after_step @ U_2site_full.conj().T

                # single qubit gates            
                for i in range(self.N):
                    dim_left = 2 ** i
                    dim_right = 2 ** (self.N - i - 1)
                    identity_left = np.identity(dim_left)
                    identity_right = np.identity(dim_right)

                    U_1site_full = np.kron(identity_left, np.kron(self.sq_gates[layer_num, i], identity_right))
                    rho_after_step = U_1site_full @ rho_after_step @ U_1site_full.conj().T
            else:
                # single qubit gates            
                for i in range(self.N):
                    dim_left = 2 ** i
                    dim_right = 2 ** (self.N - i - 1)
                    identity_left = np.identity(dim_left)
                    identity_right = np.identity(dim_right)

                    U_1site_full = np.kron(identity_left, np.kron(self.sq_gates[layer_num, i], identity_right))
                    rho_after_step = U_1site_full @ rho_after_step @ U_1site_full.conj().T

                gate_tuples = self.site_tuple_list_inverted
                U_2site = self.U2q_dagger

                for i_tuple, site_tuple in enumerate(gate_tuples):
                    dim_left = 2 ** site_tuple[0]
                    dim_right = 2 ** (self.N - site_tuple[-1] - 1)

                    identity_left = np.identity(dim_left)
                    identity_right = np.identity(dim_right)

                    U_2site_full = np.kron(identity_left, np.kron(U_2site, identity_right))
                    rho_after_step =  U_2site_full @ rho_after_step @ U_2site_full.conj().T

            # noise            
            for site in range(self.N):
                dim_left = 2 ** site
                dim_right = 2 ** (self.N - site - 1)

                identity_left = np.identity(dim_left)
                identity_right = np.identity(dim_right)

                X_full = np.kron(identity_left, np.kron(self.X, identity_right))
                Y_full = np.kron(identity_left, np.kron(self.Y, identity_right))
                Z_full = np.kron(identity_left, np.kron(self.Z, identity_right))

                rho_after_step = (1 - 3 * self.p/4) * rho_after_step \
                + (self.p/4) * (X_full @ rho_after_step @ X_full + \
                                Y_full @ rho_after_step @ Y_full + \
                                Z_full @ rho_after_step @ Z_full)
        
        target_H = self.H.full_ham()

        for layer_num in range(self.d_compute):
            # two qubit gates
            gate_tuples = self.site_tuple_list
            U_2site = self.U2q

            for i_tuple, site_tuple in enumerate(gate_tuples):
                dim_left = 2 ** site_tuple[0]
                dim_right = 2 ** (self.N - site_tuple[-1] - 1)

                identity_left = np.identity(dim_left)
                identity_right = np.identity(dim_right)

                U_2site_full = np.kron(identity_left, np.kron(U_2site, identity_right))
                target_H = U_2site_full @ target_H @ U_2site_full.conj().T

            # single qubit gates            
            for i in range(self.N):
                dim_left = 2 ** i
                dim_right = 2 ** (self.N - i - 1)
                identity_left = np.identity(dim_left)
                identity_right = np.identity(dim_right)

                U_1site_full = np.kron(identity_left, np.kron(self.sq_gates[layer_num, i], identity_right))
                target_H = U_1site_full @ target_H @ U_1site_full.conj().T

        self.target_H_exact = target_H

        return np.trace(target_H @ rho_after_step)
    
    # @partial(jit, static_argnums=(0,1))
    def dual_unitary_layer_on_mpo(self, layer_num: int, mpo_tensors: List[np.array]):
        if layer_num < self.d_compute + self.d:
            for i in range(self.N):
                res = singleq_gate(self.sq_gates[layer_num, i].conj().T, mpo_tensors[i])
                mpo_tensors[i] = res

            gate_tuples = self.site_tuple_list_inverted
            gate_tensors = self.U2q_dagger_tensors

            for i_tuple, gate_tuple in enumerate(gate_tuples):
                res_tensors = twoq_gate(gate_tensors, [mpo_tensors[site] for site in gate_tuple])
                mpo_tensors[gate_tuple[0]] = res_tensors[0]
                mpo_tensors[gate_tuple[1]] = res_tensors[1]
        else:
            gate_tuples = self.site_tuple_list
            gate_tensors = self.U2q_tensors

            for i_tuple, gate_tuple in enumerate(gate_tuples):
                res_tensors = twoq_gate(gate_tensors, [mpo_tensors[site] for site in gate_tuple])
                mpo_tensors[gate_tuple[0]] = res_tensors[0]
                mpo_tensors[gate_tuple[1]] = res_tensors[1]
            
            for i in range(self.N):
                res = singleq_gate(self.sq_gates[layer_num, i].conj().T, mpo_tensors[i])
                mpo_tensors[i] = res

        return mpo_tensors
    
    # @partial(jit, static_argnums=(0,1))
    def noisy_dual_layer_on_mpo(self, layer_num: int, mpo_tensors: List[np.array]):
        mpo_tensors = noise_layer(mpo_tensors, self.p)
        mpo_tensors = self.dual_unitary_layer_on_mpo(layer_num, mpo_tensors)

        return mpo_tensors
    
    def unitary_layer_on_mpo(self, layer_num: int, mpo_tensors: List[np.array]):
        if layer_num < self.d_compute + self.d:
            gate_tuples = self.site_tuple_list
            gate_tensors = self.U2q_tensors

            for i_tuple, gate_tuple in enumerate(gate_tuples):
                res_tensors = twoq_gate(gate_tensors, [mpo_tensors[site] for site in gate_tuple])
                mpo_tensors[gate_tuple[0]] = res_tensors[0]
                mpo_tensors[gate_tuple[1]] = res_tensors[1]

            for i in range(self.N):
                res = singleq_gate(self.sq_gates[layer_num, i], mpo_tensors[i])
                mpo_tensors[i] = res

        else:
            gate_tuples = self.site_tuple_list_inverted
            gate_tensors = self.U2q_dagger_tensors

            for i in range(self.N):
                res = singleq_gate(self.sq_gates[layer_num, i], mpo_tensors[i])
                mpo_tensors[i] = res

            for i_tuple, gate_tuple in enumerate(gate_tuples):
                res_tensors = twoq_gate(gate_tensors, [mpo_tensors[site] for site in gate_tuple])
                mpo_tensors[gate_tuple[0]] = res_tensors[0]
                mpo_tensors[gate_tuple[1]] = res_tensors[1]
            
        return mpo_tensors

    def noisy_layer_on_mpo(self, layer_num: int, mpo_tensors: List[np.array]):
        mpo_tensors = self.unitary_layer_on_mpo(layer_num, mpo_tensors)
        mpo_tensors = noise_layer(mpo_tensors, self.p)

        return mpo_tensors

    def target_H(self):
        _, target_H_tensors = HamSumZ(self.N)
        for layer_num in range(self.d_compute):
            gate_tuples = self.site_tuple_list
            gate_tensors = self.U2q_tensors

            for i_tuple, gate_tuple in enumerate(gate_tuples):
                res_tensors = twoq_gate(gate_tensors, [target_H_tensors[site] for site in gate_tuple])
                target_H_tensors[gate_tuple[0]] = res_tensors[0]
                target_H_tensors[gate_tuple[1]] = res_tensors[1]
            
            for i in range(self.N):
                res = singleq_gate(self.sq_gates[layer_num, i], target_H_tensors[i])
                target_H_tensors[i] = res

        return target_H_tensors
    
    # def error_dynamics(self, D: int):
    #     error_list = []
        
    #     mpo_tensors_exact = self.target_H()
    #     mpo_tensors = self.target_H()

    #     for i in range(self.depth-1, -1, -1):
    #         mpo_tensors = self.noisy_dual_layer_on_mpo(i, mpo_tensors)
    #         # canonicalize
    #         compressed_dims = tuple(bond_dims(mpo_tensors)[1:-1])
    #         mpo_tensors = right_canonicalize(tensors = mpo_tensors, compressed_dims = compressed_dims)

    #         # compress
    #         compressed_dims = gen_compression_dims(D, self.N)
    #         mpo_tensors = left_canonicalize(tensors = mpo_tensors, compressed_dims = compressed_dims)

    #         mpo_tensors_exact = self.noisy_dual_layer_on_mpo(i, mpo_tensors_exact)
    #         # canonicalize
    #         compressed_dims = tuple(bond_dims(mpo_tensors_exact)[1:-1])
    #         mpo_tensors_exact = right_canonicalize(tensors = mpo_tensors_exact, compressed_dims = compressed_dims)

    #         # compress
    #         compressed_dims = gen_compression_dims(2 ** self.N, self.N)
    #         mpo_tensors_exact = left_canonicalize(tensors = mpo_tensors_exact, compressed_dims = compressed_dims)

    #         diff_tensors = subtract_MPO(mpo_tensors, mpo_tensors_exact)
    #         error = trace_MPO_squared(diff_tensors)
    #         error_list.append(error)

    #         print(error)

        # return error_list

    def schrod_bound(self, D: int):
        rho = self.psi_init_tensors

        errors = [0.0,]

        for i in range(0, self.depth, 1):
            print(i)
            rho_new = self.noisy_layer_on_mpo(i, rho)
            rho_new_proj = copy.deepcopy(rho_new)
            
            # canonicalize
            compressed_dims = tuple(bond_dims(rho_new_proj)[1:-1])
            rho_new_proj = right_canonicalize(tensors = rho_new_proj, compressed_dims = compressed_dims)

            # compress
            compressed_dims = gen_compression_dims(D, self.N)
            rho_new_proj = left_canonicalize(tensors = rho_new_proj, compressed_dims = compressed_dims)

            error_sq = trace_two_MPOs(rho_new, rho_new) + trace_two_MPOs(rho_new_proj, rho_new_proj) - \
                2 * trace_two_MPOs(rho_new, rho_new_proj)

            # Ht = subtract_MPO(sigma_new_proj, sigma_new)
            error = np.sqrt(error_sq)
            errors.append(error)

            rho = rho_new_proj
        
        energy = trace_two_MPOs(self.target_H(), rho)

        schrod_bound = energy - (2 ** self.N) * np.sqrt(self.N) * np.sum(errors)

        return energy, schrod_bound

    def bounds(self, D: int):
        sigma = self.target_H()
        sigma = scale_mpo(sigma, -1.0)

        Ht_norms = [0.0,]

        for i in range(self.depth-1, -1, -1):
            print(i)
            sigma_new = self.noisy_dual_layer_on_mpo(i, sigma)
            sigma_new_proj = copy.deepcopy(sigma_new)
            
            # canonicalize
            compressed_dims = tuple(bond_dims(sigma_new_proj)[1:-1])
            sigma_new_proj = right_canonicalize(tensors = sigma_new_proj, compressed_dims = compressed_dims)

            # compress
            compressed_dims = gen_compression_dims(D, self.N)
            sigma_new_proj = left_canonicalize(tensors = sigma_new_proj, compressed_dims = compressed_dims)

            Ht_norm_sq = trace_two_MPOs(sigma_new, sigma_new) + trace_two_MPOs(sigma_new_proj, sigma_new_proj) - 2 * trace_two_MPOs(sigma_new, sigma_new_proj)

            # Ht = subtract_MPO(sigma_new_proj, sigma_new)
            Ht_norm = np.sqrt(Ht_norm_sq)
            Ht_norms.append(Ht_norm)

            sigma = sigma_new_proj
        
        init_state_term = -trace_two_MPOs(self.psi_init_tensors, sigma)

        # print('init_state_term = ', init_state_term)

        Ht_norms = np.array(Ht_norms[:-1][::-1])

        # print('Ht_norms = ', Ht_norms)

        heis_bound = init_state_term - np.sum(Ht_norms)
        dual_bound = init_state_term - np.dot(np.sqrt(self.purity_bounds), Ht_norms)

        return init_state_term, heis_bound, dual_bound

    def bounds_tr1(self, D: int):
        sigma = self.target_H()
        sigma = scale_mpo(sigma, -1.0)

        Ht_norms = [0.0,]
        Ht_traces = [0.0,]

        e_inv = 2 ** (-self.N)

        for i in range(self.depth-1, -1, -1):
            print(i)
            sigma_new = self.noisy_dual_layer_on_mpo(i, sigma)
            sigma_new_proj = copy.deepcopy(sigma_new)
            
            # canonicalize
            compressed_dims = tuple(bond_dims(sigma_new_proj)[1:-1])
            sigma_new_proj = right_canonicalize(tensors = sigma_new_proj, compressed_dims = compressed_dims)

            # compress
            compressed_dims = gen_compression_dims(D, self.N)
            sigma_new_proj = left_canonicalize(tensors = sigma_new_proj, compressed_dims = compressed_dims)

            # h2
            Ht_norm_sq = trace_two_MPOs(sigma_new, sigma_new) + trace_two_MPOs(sigma_new_proj, sigma_new_proj) - 2 * trace_two_MPOs(sigma_new, sigma_new_proj)

            # h
            Ht_trace = trace_two_MPOs(sigma_new_proj, self.identity_mpo) -  trace_two_MPOs(sigma_new, self.identity_mpo)

            Ht_norm = np.sqrt(Ht_norm_sq)
            Ht_norms.append(Ht_norm)
            Ht_traces.append(Ht_trace)

            sigma = sigma_new_proj
        
        init_state_term = -trace_two_MPOs(self.psi_init_tensors, sigma)

        # print('init_state_term = ', init_state_term)

        Ht_norms = np.array(Ht_norms[:-1][::-1])
        Ht_traces = np.array(Ht_traces[:-1][::-1])

        # print('Ht_norms = ', Ht_norms)

        heis_bound = init_state_term - np.sum(Ht_norms)
        dual_bound = init_state_term - np.dot(np.sqrt(self.purity_bounds), Ht_norms)
        dual_bound_tr1 = init_state_term  - np.sqrt(np.sum((self.purity_bounds - e_inv) * (Ht_norms * Ht_norms - Ht_traces * Ht_traces * e_inv))) + Ht_traces * e_inv

        return init_state_term, heis_bound, dual_bound, dual_bound_tr1

    def init_mpo(self, D: int):
        mpo_tensors = self.target_H()

        for i in range(self.depth-1, -1, -1):
            mpo_tensors = self.noisy_dual_layer_on_mpo(i, mpo_tensors)
            
            # canonicalize
            compressed_dims = tuple(bond_dims(mpo_tensors)[1:-1])
            mpo_tensors = right_canonicalize(tensors = mpo_tensors, compressed_dims = compressed_dims)

            # compress
            compressed_dims = gen_compression_dims(D, self.N)
            mpo_tensors = left_canonicalize(tensors = mpo_tensors, compressed_dims = compressed_dims)

        return mpo_tensors

# class SumZ_RXX():
#     def __init__(self, N: int, d: int, p: float, key):
#         self.N = N
#         self.d = d
#         self.p = p
#         self.H, self.H_tensors = HamSumZ(self.N)

#         self.site_tuple_list = list(zip(range(0, self.N, 2), range(1, self.N, 2))) + list(zip(range(1, self.N, 2), range(2, self.N, 2)))
#         self.site_tuple_list_inverted = list(zip(range(1, self.N, 2), range(2, self.N, 2))) + list(zip(range(0, self.N, 2), range(1, self.N, 2)))

#         theta_half = jax.random.normal(key, shape = (len(self.site_tuple_list), self.d//2))
#         self.theta = np.column_stack((theta_half, -np.roll(theta_half[:, ::-1], (self.N - 2)//2, axis = 0)))   

#         self.psi_init = np.zeros((2 ** self.N,), dtype = complex)
#         self.psi_init = self.psi_init.at[-1].set(1.0)

#         psi_init_local_tensor = np.array([[0, 0],[0, 1]], dtype = complex)
#         psi_init_local_tensor = np.expand_dims(psi_init_local_tensor, axis = (0,2))
#         self.psi_init_tensors = [psi_init_local_tensor] * self.N

#         self.X = np.array([[0,1],[1,0]], dtype = complex) # u, d
#         self.Y = np.array([[0,-1j],[1j,0]], dtype = complex) # u, d
#         self.Z = np.array([[1,0],[0,-1]], dtype = complex) # u, d

#     def primal_noisy(self):
#         rho_init = np.outer(self.psi_init, self.psi_init.conj().T)

#         rho_after_step = rho_init

#         for layer_num in range(0, self.d):
#             if layer_num < self.d//2:
#                 gate_tuples = self.site_tuple_list
#             else:
#                 gate_tuples = self.site_tuple_list_inverted

#             # nn unitaries
#             for i_tuple, site_tuple in enumerate(gate_tuples):
#                 theta = self.theta.at[i_tuple, layer_num].get()

#                 U_2site, _ = RXX(theta)

#                 dim_left = 2 ** site_tuple[0]
#                 dim_right = 2 ** (self.N - site_tuple[-1] - 1)

#                 identity_left = np.identity(dim_left)
#                 identity_right = np.identity(dim_right)

#                 U_2site_full = np.kron(identity_left, np.kron(U_2site, identity_right))

#                 rho_after_step = np.matmul(U_2site_full, np.matmul(rho_after_step, U_2site_full.conj().T)) 

#             # noise
#             for site in range(self.N):
#                 dim_left = 2 ** site
#                 dim_right = 2 ** (self.N - site - 1)

#                 identity_left = np.identity(dim_left)
#                 identity_right = np.identity(dim_right)

#                 X_full = np.kron(identity_left, np.kron(self.X, identity_right))
#                 Y_full = np.kron(identity_left, np.kron(self.Y, identity_right))
#                 Z_full = np.kron(identity_left, np.kron(self.Z, identity_right))

#                 rho_after_step = (1 - 3 * self.p/4) * rho_after_step \
#                 + (self.p/4) * (np.matmul(X_full, np.matmul(rho_after_step, X_full)) + \
#                                 np.matmul(Y_full, np.matmul(rho_after_step, Y_full)) + \
#                                 np.matmul(Z_full, np.matmul(rho_after_step, Z_full)))

#         return np.trace(np.matmul(self.H.full_ham(), rho_after_step))

#     def dual_unitary_layer_on_mpo(self, layer_num: int, mpo_tensors: List[np.array]):
#         if layer_num < self.d//2:
#             gate_tuples = self.site_tuple_list
#         else:
#             gate_tuples = self.site_tuple_list_inverted

#         for i_tuple, gate_tuple in enumerate(gate_tuples):
#             theta = self.theta.at[i_tuple, layer_num].get() 
#             _, gate_tensors = RXX(-theta) #conjugate the gate

#             res_tensors = twoq_gate(gate_tensors, [mpo_tensors[site] for site in gate_tuple])
#             mpo_tensors[gate_tuple[0]] = res_tensors[0]
#             mpo_tensors[gate_tuple[1]] = res_tensors[1]
        
#         return mpo_tensors
    
#     def noisy_dual_layer_on_mpo(self, layer_num: int, mpo_tensors: List[np.array]):
#         mpo_tensors = noise_layer(mpo_tensors, self.p)
#         mpo_tensors = self.dual_unitary_layer_on_mpo(layer_num, mpo_tensors)

#         return mpo_tensors
    
#     def init_mpo(self, D: int):
#         mpo_tensors = self.H_tensors

#         for i in range(self.d-1, -1, -1):
#             mpo_tensors = self.noisy_dual_layer_on_mpo(i, mpo_tensors)
            
#             # canonicalize
#             compressed_dims = tuple(bond_dims(mpo_tensors)[1:-1])
#             mpo_tensors = right_canonicalize(tensors = mpo_tensors, compressed_dims = compressed_dims)

#             # compress
#             compressed_dims = gen_compression_dims(D, self.N)
#             mpo_tensors = left_canonicalize(tensors = mpo_tensors, compressed_dims = compressed_dims)

#         return mpo_tensors
    

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
# def vec_to_herm_tensor(vec: np.array, shape: Tuple[int]):
#     l, s, r, _ = shape # u, d = s 

#     # vec = np.reshape(vec, (l * r, s * s))

#     tensor = np.zeros((l * r, s, s), dtype = complex)
#     zeros_s = np.zeros((s,s), dtype = complex)

#     ones_l = np.ones((l,))
#     ones_r = np.ones((l,))

#     upper_diag_indices = np.triu_indices(s, 1)
#     diag_indices = np.diag_indices(s)

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
    
#     tensor = np.reshape(tensor, (l, r, s, s))
#     tensor = np.transpose(tensor, (0, 2, 1, 3))

#     return tensor

# @partial(jit, static_argnames = ('shape',))
# def vec_to_herm_mpo(vec: np.array, shape: Tuple[Tuple[int]]):
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

# def gen_overlap_ham(mpo_tensors: List[np.array], site_idx: int):

#     """
#     !! edge cases
#     !! check
#     """
#     num_sites = len(mpo_tensors)

#     tooth = mpo_tensors[0]
#     for i in range(0, site_idx - 1):
#         zip = np.tensordot(tooth, mpo_tensors[i], axes = ((0,1,3), (0,3,1)))
#         # Di (tooth), Di (tensor)
#         tooth = np.tensordot(zip, mpo_tensors[i+1], axes = ((0,),(0,)))
#         # Di (tooth), Di (tensor) tdot l,u,r,d -> Di (tensor), u,r,d

#     left_mat = np.tensordot(tooth, mpo_tensors[site_idx - 1], axes = ((0,1,3), (0,3,1)))
#     # Di (tooth), Di (tensor) = l l'

#     tooth = mpo_tensors[-1]
#     for i in range(num_sites - 1, site_idx + 1, -1):
#         zip = np.tensordot(tooth, mpo_tensors[i], axes = ((2,1,3), (2,3,1)))
#         # Di (tooth), Di (tensor)
#         tooth = np.tensordot(zip, mpo_tensors[i-1], axes = ((0,),(2,)))
#         # Di (tooth), Di (tensor) tdot l,u,r,d -> l, u, Di (tensor),d

#     right_mat = np.tensordot(tooth, mpo_tensors[site_idx + 1], axes = ((2,1,3), (2,3,1)))
#     # Di (tooth), Di (tensor) = r r'

    






# def gen_overlap_ham_layer(mpo_tensors: List[np.array], layer_tensors: List[np.array], site: int):
    




# managing bond dims?
    # optimization will need fixed sizes for MPOs
# dual circuit
# vec to MPO
    # vec to operators first
    # sum with conj tr to get Herm
# local Ham to MPO
    # ??