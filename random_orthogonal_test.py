import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

num_samples = int(1e4)
d = 4

D = np.diag(np.arange(0, d))
Gamma = np.random.normal(size = (d,d))
Gamma = Gamma - Gamma.T

E_analytical = (np.trace(D)**2/(d**2 - 1)) * Gamma + \
               (2/(d * (d + 1))) * D @ Gamma @ D \
               - (2/(d**2 - 1)) * np.trace(D) * (Gamma @ D + D @ Gamma)

norm_list = []

E = 0

for i in range(num_samples):

    # M = np.random.normal(size = (d,d))
    # O, R = np.linalg.qr(M)

    O = stats.special_ortho_group.rvs(d)

    V = O @ D @ O.T @ Gamma @ O.T @ D @ O

    E = E * i/(i + 1) + V/(i + 1)
    norm_list.append(np.linalg.norm(E-E_analytical))

plt.plot(norm_list)
plt.show()
