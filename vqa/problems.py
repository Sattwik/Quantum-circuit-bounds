import abc
from typing import List, Tuple, Callable, Dict
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import qutip

class Problem(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def Up(self) -> qutip.Qobj:
        """ Returns the unitary generated by the Hamiltonian
         to be minimized for solving the problem. """

        raise NotImplementedError()

    @abc.abstractmethod
    def Um(self) -> qutip.Qobj:
        """ Returns the mixing unitary. """

        raise NotImplementedError()

class MaxCut(Problem):

    def __init__(self, graph, lattice):

        self.graph = graph
        self.lattice = lattice

        # preparing the Z operators
        self.site_z_ops = {}
        self.site_nums = {}
        op_num = 0

        self.num_sites_in_lattice = self.lattice.number_of_nodes()

        self.Z = qutip.sigmaz()
        self.I = qutip.qeye(2)

        self.I_tot = qutip.tensor([self.I] * self.num_sites_in_lattice)

        for site in self.lattice:

            self.site_z_ops[site] = qutip.tensor([self.I] * op_num + [self.Z] + \
                                            [self.I] * (self.num_sites_in_lattice - op_num - 1))

            self.site_nums[site] = op_num
            op_num += 1

        # the problem Hamiltonian
        self.H = 0

        for edge in self.graph.edges:

            Zj = self.site_z_ops[edge[0]]
            Zk = self.site_z_ops[edge[1]]
            wjk = -1

            local_op = wjk/2 * (self.I_tot - Zj * Zk)

            self.H += local_op

    def init_state(self):

        psi0 = qutip.basis([2] * self.num_sites_in_lattice)
        psi0 = qutip.hadamard_transform(N = self.num_sites_in_lattice) * psi0

        return psi0

    def Up(self, gamma):

        """
        Returns the unitary generated by the Hamiltonian whose ground
        state gives the solution to the MaxCut
        problem on the graph. All edges have weight -1.
        """
        # NB - H is diagonal in the computational basis

        diag_entries_H = self.H.diag()
        Up = np.diag(np.exp(-1j * gamma * diag_entries_H))
        Up = qutip.Qobj(Up, dims = self.H.dims, shape = self.H.shape)

        return Up

    def Um(self, beta):

        """
        Returns the mixing unitary.
        """

        Um = self.I_tot
        sx = qutip.sigmax()
        Ux = (-1j * beta * sx).expm()

        for site in self.graph:

            site_num = self.site_nums[site]

            Um *= qutip.tensor([self.I] * site_num + [Ux] + \
                               [self.I] * (self.num_sites_in_lattice - site_num - 1))

        return Um

    def Unoise(self, mc_probs, p_noise):

        """
        Returns the unitary corresponding to a Monte Carlo sampling of the
        noise channel (depol. noise on individual qubits).
        """

        Un = self.I_tot
        sx = qutip.sigmax()
        sy = qutip.sigmay()
        sz = qutip.sigmaz()

        for site in self.graph:

            site_num = self.site_nums[site]

            r = mc_probs[site_num]

            if r < p_noise/3:
                Un *= qutip.tensor([self.I] * site_num + [sx] + \
                                   [self.I] * (self.num_sites_in_lattice - site_num - 1))

            elif p_noise/3 <= r < 2 * p_noise/3:
                Un *= qutip.tensor([self.I] * site_num + [sy] + \
                                   [self.I] * (self.num_sites_in_lattice - site_num - 1))

            elif 2 * p_noise/3 <= r < p_noise:
                Un *= qutip.tensor([self.I] * site_num + [sz] + \
                                   [self.I] * (self.num_sites_in_lattice - site_num - 1))

        return Un
