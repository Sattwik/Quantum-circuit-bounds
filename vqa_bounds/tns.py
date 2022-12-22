import abc
from typing import List, Tuple, Callable, Dict
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.config import config
import tensornetwork as tn
tn.set_default_backend("jax")
config.update("jax_enable_x64", True)

class Operator():
    def __init__(self, d: int, DL: int, DR: int,
                 init_with_array = False, arr = None):
        """
        d: physical dim
        DL: left bond dim
        DR: right bond dim
        """
        self.d = d
        self.DL = DL
        self.DR = DR

        if init_with_array:
            self.op = tn.Node(arr, axis_names = ["up","down", "left", "right"])
        else:
            id = jnp.eye(d, dtype = complex)
            id = id.reshape((d, d, DL, DR))
            self.op = tn.Node(id, axis_names = ["up","down", "left", "right"])

class MPO():
    def __init__(self, num_sites: int, D: List, d: int,
                 init_from_vec = False, vec: jnp.array = None,
                 ops: List):
        self.d = d
        self.D = D # D is the max bond dim
        self.num_sites = num_sites
        self.L = num_sites
        self.ops = ops

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
