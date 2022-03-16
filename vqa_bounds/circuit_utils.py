import abc
from typing import List, Tuple, Callable, Dict

import numpy as np
import scipy

from vqa_bounds import meta_system

def circ_obj_ext(circ_params: np.array, sys_obj: meta_system.System):

    return sys_obj.circ_obj(circ_params)

def circ_grad_ext(circ_params: np.array, sys_obj: meta_system.System):

    return sys_obj.circ_grad(circ_params)

def optimize_circuit(circ_params_init: np.array,
                     bounds: np.array,
                     sys_obj: meta_system.System):

    """
    Main optimization function.

    Args:
        circ_params_init: init value of circuit parameters.

    Returns:
        np.array(obj_over_opti): Objective value over the optimization run.
        opt_result: Result of the optimization run.

        Important: updates the sys_obj with the optimized circuit parameters
    """

    opt_args = (sys_obj,)

    obj_over_opti = []

    def callback_func(x):

        obj_eval = circ_obj_ext(x, opt_args[0])

        obj_over_opti.append(obj_eval)

        print('Iteration ' + str(len(obj_over_opti)) +
              '. Objective = ' + str(obj_eval))

    # ub_array = np.concatenate((np.ones(QAOA_obj.p) * 2 * np.pi, np.ones(QAOA_obj.p) * np.pi))
    # bnds = scipy.optimize.Bounds(lb = 0, ub = ub_array)

    bnds = bounds

    opt_result = scipy.optimize.minimize(
                    circ_obj_ext, circ_params_init,
                    args = opt_args,
                    method = 'L-BFGS-B', jac = circ_grad_ext, bounds = bnds,
                    options={'disp': None,
                    'maxcor': 10,
                    'ftol': 2.220446049250313e-09,
                    'gtol': 1e-05,
                    'eps': 1e-08,
                    'maxfun': 15000,
                    'maxiter': 50, 'iprint': 10, 'maxls': 20},
                    callback = callback_func)

    sys_obj.update_opt_circ_params(opt_result.x)

    return np.array(obj_over_opti), opt_result
