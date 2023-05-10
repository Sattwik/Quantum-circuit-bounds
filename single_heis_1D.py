from datetime import datetime
from datetime import date
import argparse
import pickle
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
from vqa_bounds import mpo_numpy

parser = argparse.ArgumentParser(description='System size(N), noise probability(p), seed')
parser.add_argument('--N', type = str)
parser.add_argument('--d', type = str)
parser.add_argument('--seed', type = str)
parser.add_argument('--p', type = str)
parser.add_argument('--theta', type = str)
parser.add_argument('--D', type = str)
parser.add_argument('--result_save_path', type = str)
cliargs = parser.parse_args()

N = int(cliargs.N)
d = int(cliargs.d)
D = int(cliargs.D)
p = float(cliargs.p)
theta = float(cliargs.theta)
seed = int(cliargs.seed)

print('N = ', N)
print('d = ', d)
print('p = ', p)
print('theta = ', theta)

circ = mpo_numpy.SumZ_RXX(N, d, p, theta, seed)

hval, hb, db = circ.bounds(D = D)

# TODO: check if imaginary parts zero
heis_val = complex(hval)
heis_bound = complex(hb)
dual_bound = complex(db)

print("heis_val = ", heis_val)
print("heis_bound = ", heis_bound)
print("dual_bound = ", dual_bound)

if N <= 6:
    output = complex(circ.primal_noisy())
    print("output = ", output)
    data_list = [output, heis_val, heis_bound, dual_bound]
else:
    data_list = [heis_val, heis_bound, dual_bound]

data_file_name = "heis1D-N-" + str(N) + "-d-" + str(d) + "-seed-" + \
                cliargs.seed + "-theta-" + f'{theta:.4f}' + \
                "-p-" + str(p) + "-D-" + str(D) + ".pkl"

with open(os.path.join(cliargs.result_save_path, data_file_name), "wb") as f_for_pkl:
    pickle.dump(data_list, f_for_pkl)