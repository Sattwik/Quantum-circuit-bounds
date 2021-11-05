import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import qutip
import scipy

from vqa import problems

class QAOA():

    def __init__(self, problem_obj: problems.Problem, p):

        """
        problem_obj: problem to be solved. Supplies the circuit unitaries.
        p: number of circuit steps
        """

        self.p = p
        self.psi_init = problem_obj.init_state()
        self.problem_obj = problem_obj

    def objective(self, gamma: np.array, beta: np.array):

        """
        Note: for uniform weight MaxCut the objective has periodicity of 2 pi in
        every gamma and pi in every beta.
        """

        psi_after_step = self.psi_init

        for step_num in range(self.p):

            psi_after_step = self.problem_obj.Ufull(gamma[step_num], beta[step_num], psi_after_step)

        return qutip.expect(self.problem_obj.H, psi_after_step)

    def gradient(self, gamma: np.array, beta: np.array):

        delta = 1e-3

        gradient_gamma = np.zeros(self.p)
        gradient_beta = np.zeros(self.p)

        for step_num in range(self.p):

            gamma_plus = np.copy(gamma)
            gamma_plus[step_num] += delta

            gamma_minus = np.copy(gamma)
            gamma_minus[step_num] -= delta

            objective_plus = self.objective(gamma_plus, beta)
            objective_minus = self.objective(gamma_minus, beta)

            gradient_gamma[step_num] = (objective_plus - objective_minus)/(2 * delta)

            beta_plus = np.copy(beta)
            beta_plus[step_num] += delta

            beta_minus = np.copy(beta)
            beta_minus[step_num] -= delta

            objective_plus = self.objective(gamma, beta_plus)
            objective_minus = self.objective(gamma, beta_minus)

            gradient_beta[step_num] = (objective_plus - objective_minus)/(2 * delta)

        gradient = np.concatenate((gradient_gamma, gradient_beta))

        return gradient

def objective_external_QAOA(gamma_beta: np.array, QAOA_obj: QAOA,):

    gamma = gamma_beta[:QAOA_obj.p]
    beta = gamma_beta[QAOA_obj.p:]

    return QAOA_obj.objective(gamma, beta)

def gradient_external_QAOA(gamma_beta: np.array, QAOA_obj: QAOA):

    gamma = gamma_beta[:QAOA_obj.p]
    beta = gamma_beta[QAOA_obj.p:]

    return QAOA_obj.gradient(gamma, beta)

def optimize_QAOA(gamma_beta_init: np.array, QAOA_obj: QAOA):

    """
    Main optimization function.

    Args:
        gamma_init: init value of gamma parameter. Size = p
        beta_init: init value of beta parameter. Size = p
        QAOA_obj: Algo that produces objective, gradient.

    Returns:
        np.array(obj_over_opti): Objective value over the optimization run.
        opt_result: Result of the optimization run.
    """

    opt_args = (QAOA_obj,)

    obj_over_opti = []

    def callback_func(x):

        obj_eval = objective_external_QAOA(x, opt_args[0])

        obj_over_opti.append(obj_eval)

        print('Iteration ' + str(len(obj_over_opti)) + '. Objective = ' + str(obj_eval))

    ub_array = np.concatenate((np.ones(QAOA_obj.p) * 2 * np.pi, np.ones(QAOA_obj.p) * np.pi))

    bnds = scipy.optimize.Bounds(lb = 0, ub = ub_array)

    opt_result = scipy.optimize.minimize(objective_external_QAOA, gamma_beta_init, args = opt_args,
                                         method = 'L-BFGS-B', jac = gradient_external_QAOA, bounds = bnds,
                                         options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09,
                                         'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 20, 'iprint': 5, 'maxls': 20},
                                         callback = callback_func)

    return np.array(obj_over_opti), opt_result
