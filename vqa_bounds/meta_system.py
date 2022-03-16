import abc
from typing import List, Tuple, Callable, Dict

import numpy as np

class System(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def circ_obj(self, circ_params: np.array):
        """ Returns the objective that needs to be minimized by choosing
        the quantum circuit params.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def circ_grad(self, circ_params: np.array):
        """ Returns the gradient of circ_obj.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def dual_obj(self, dual_vars: np.array):
        """ Returns the dual function that needs to be optimized to obtain
        bounds on the circuit performance in the presence of noise.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def dual_grad(self, dual_vars: np.array):
        """ Returns the gradient of dual_obj.
        """
        raise NotImplementedError()
