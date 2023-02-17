from typing import List, Tuple, Callable, Dict

import numpy as np

class Hamiltonian():
    """ Hamiltonians with k-body (nearest neighbor) interaction terms (OBC). """

    def __init__(self, num_sites: int, k: int):
        self.num_sites = num_sites
        self.site_tuple_list = [(i , i + 1) for i in range(num_sites - 1)]
        self.terms = {} # dictionary to hold the terms in the Ham. 

    def full_ham(self):
        full_mat = 0
        for site_tuple, term in self.terms.items():
            dim_left = 2 ** site_tuple[0]
            dim_right = 2 ** (self.num_sites - site_tuple[-1] - 1)

            identity_left = np.identity(dim_left)
            identity_right = np.identity(dim_right)

            full_mat += np.kron(identity_left, np.kron(term, identity_right))
        return full_mat
    
def HamSumZ(num_sites: int):
    """
    Returns Hamiltonian representation of \sum Z.
    """
    H = Hamiltonian(num_sites)

    I = np.array([[1, 0],[0, 1]], dtype = complex)
    Z = np.array([[1, 0],[0, -1]], dtype = complex)

    for site_tuple in H.site_tuple_list:
        H.terms[site_tuple] = np.kron(I, Z)

    H.terms[(0,1)] += np.kron(Z, I)

    return H

#------------------------------------------------------------------------------#
# Models and circuits
#------------------------------------------------------------------------------#
def gate_to_MPO(gate: np.array, num_sites: int, D: int = None):
    # gate.shape = s1s2s3..., s1's2's3'...
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
def RXX(theta: float):
    # not Clifford
    # D = 2
    c = np.cos(theta)
    s = np.sin(theta)

    U = np.array([[c, 0.0, 0.0, -1j*s],
                   [0.0, c, -1j*s, 0.0],
                   [0.0, -1j*s, c, 0.0],
                   [-1j*s, 0.0, 0.0, c]], dtype = complex)

    tensors = gate_to_MPO(U, num_sites = 2, D = 2)
    return U, tensors

def CNOT():
    # Clifford
    # D = 2
    U = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype = complex)

    tensors = gate_to_MPO(U, num_sites = 2, D = 2)
    return U, tensors

#------------------------------------------------------------------------------#
# MPS methods
#------------------------------------------------------------------------------#
def full_contract(tensors: List[np.array]):
    """WARNING: memory expensive!!"""

    num_sites = len(tensors)

    res = tensors[0]

    for i in range(num_sites-1):
        res = np.tensordot(res, tensors[i + 1], axes=((-1,), (0,))) # (l) (u) (u') (r')
        
    res = np.squeeze(res) # (u1) (u2) ... (uN)
    res = np.reshape(res, (2**num_sites,))
    return res

def parent_ham_of_MPS(tensors: List[np.array]):
    num_sites = len(tensors)
    parent = Hamiltonian(num_sites)

    for site_tuple in parent.site_tuple_list:
        A2 = np.tensordot(tensors[site_tuple[0]], tensors[site_tuple[1]], 
                          axes = ((-1), (0))) # l,u,u',r'
        A2 = np.transpose(A2, axes = [1, 2, 0, 3]) # u,u',l,r'
        A2 = np.reshape(A2, (A2.shape[0] * A2.shape[1], 
                            A2.shape[2] * A2.shape[3])) # uu',lr'
        Z = scipy.linalg.null_space(A2.conj().T) # uu'

        


#------------------------------------------------------------------------------#
# Circuits and noise
#------------------------------------------------------------------------------#
def singleq_gate_on_MPS_tensor(gate: np.array, tensor: np.array):
    ### WARNING: INCORRECT> CONJUGATE
    res = np.tensordot(gate, tensor, axes = ((1), (1)))
    # u,d tdot l,u,r,d -> u,l,r,d
    res = np.swapaxes(res, 0, 1) # l,u,r,d
    res = np.tensordot(gate.conj().T, res, axes = ((0), (3)))
    # u,d tdot l,u,r,d -> d, u, l, r
    res = np.transpose(res, (2, 1, 3, 0))

    return res

def twoq_gate_on_MPS_tensor(gates: List[np.array], tensors: List[np.array]):
    ### WARNING: INCORRECT> CONJUGATE
    res = []

    num_sites = len(gates)

    for i in range(num_sites):
        res = np.tensordot(gate, tensor, axes = ((3), (1)))
        # gl,gu,gr,gd tdot l,u,r,d -> gl, gu, gr, l, r, d
        res = np.transpose(res, (0, 3, 1, 2, 4, 5)) # gl, l, gu, gr, r, d

        gate_herm_conj = np.transpose(gate)
        res = np.tensordot(gate, res, axes = ((0), (3)))
        # u,d tdot l,u,r,d -> d, u, l, r
        res = np.transpose(res, (2, 1, 3, 0))

    return res