import numpy as np
import stim
import matplotlib.pyplot as plt

def avg_energy_from_samples(samples):

    mult = np.zeros((samples.shape[0], samples.shape[1] - 1))

    for i in range(mult.shape[1]):
        mult[:, i] = -1 * samples[:, i] * samples[:, i + 1]

    energies = np.sum(mult, axis = 1)
    avg_energy = np.sum(energies)/samples.shape[0]

    return avg_energy


energy_list = []
N_list = range(3, 31)
p = 0.5
num_samples = int(1e4)

for N in N_list:
    print(N)
    circuit = stim.Circuit()
    circuit.append("H", [0])

    for i in range(N - 1):
        circuit.append("CNOT", [i, i + 1])
        circuit.append("DEPOLARIZE1", [i for i in range(N)], p)

    circuit.append("M", [i for i in range(N)])

    # print(circuit.diagram())

    sampler = circuit.compile_sampler()
    samples = sampler.sample(shots = int(N * N * 100/p)) * 1
    samples = (0.5 - samples) * 2

    energy_list.append(avg_energy_from_samples(samples))

    # print(samples)
    #
    # print(avg_energy_from_samples(samples))

plt.plot(N_list, energy_list)
# plt.yscale('log')
plt.show()
