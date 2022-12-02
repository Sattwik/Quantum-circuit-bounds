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
