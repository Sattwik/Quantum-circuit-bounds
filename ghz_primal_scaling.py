import numpy as np
import stim
import matplotlib.pyplot as plt
from jax import jit, grad, vmap
import jax.numpy as jnp
import jax

# def avg_energy_from_samples(samples):
#
#     mult = np.zeros((samples.shape[0], samples.shape[1] - 1))
#
#     for i in range(mult.shape[1]):
#         mult[:, i] = -1 * samples[:, i] * samples[:, i + 1]
#
#     energies = np.sum(mult, axis = 1)
#     avg_energy = np.sum(energies)/samples.shape[0]
#
#     return avg_energy
#
#
# energy_list = []
# N_list = range(3, 31)
# p = 0.5
# num_samples = int(1e4)
#
# for N in N_list:
#     print(N)
#     circuit = stim.Circuit()
#     circuit.append("H", [0])
#
#     for i in range(N - 1):
#         circuit.append("CNOT", [i, i + 1])
#         circuit.append("DEPOLARIZE1", [i for i in range(N)], p)
#
#     circuit.append("M", [i for i in range(N)])
#
#     # print(circuit.diagram())
#
#     sampler = circuit.compile_sampler()
#     samples = sampler.sample(shots = int(N * N * 100/p)) * 1
#     samples = (0.5 - samples) * 2
#
#     energy_list.append(avg_energy_from_samples(samples))
#
#     # print(samples)
#     #
#     # print(avg_energy_from_samples(samples))
#
# plt.plot(N_list, energy_list)
# # plt.yscale('log')
# plt.show()

def xor_cond(args):
    i, _ = args
    return i >= 0

def xor_fun(args):
    i, s_new = args
    s_new = s_new.at[i].set(jnp.logical_xor(s_new.at[i].get(), s_new.at[i + 1].get()))
    i = i - 1

    return (i, s_new)

def step_stab(i, args: jnp.array):
    s, num_z = args
    s_new = s
    N = s.shape[0]

    init_args = (N - 2, s_new)

    _, s_new = jax.lax.while_loop(xor_cond, xor_fun, init_args)
    s_new = s_new.at[N - 1].set(jnp.logical_xor(s_new.at[0].get(), s_new.at[N - 1].get()))

    return s_new, num_z + jnp.sum(s_new)

# @jit
def propagate_stab(s: jnp.array, num_steps: int):
    args = (s, 1)
    s_end, num_z = jax.lax.fori_loop(0, num_steps, step_stab, args)
    return s_end, num_z

# N = 4
# s = jnp.eye(N)[:, -1]
# s_end, num_z = propagate_stab(s, 2)
# print(s_end)
# print(num_z)

N_list = np.arange(4, 202, 2)
energy_list = []

p = 0.01
q = 1-p

for N in N_list:
    print(N)
    s = jnp.eye(N)
    s_end, num_z = vmap(propagate_stab, in_axes = [1, None])(s, N)

    energy = np.sum(q ** num_z)
    energy_list.append(energy)

    # print(s_end)
    # print(num_z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(N_list, energy_list)
ax.set_yscale('log')
plt.show()
