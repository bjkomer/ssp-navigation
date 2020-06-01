import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fname = 'output/results_integ_noise0.25.npz'

returns = np.load(fname)['returns']

n_mazes = returns.shape[0]
n_episodes = returns.shape[1]

fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)

# ax.plot(returns[0, :])
# ax.plot(returns[1, :])
# ax.plot(returns.mean(axis=1))
ax.scatter(np.arange(n_mazes), returns.mean(axis=1))

plt.show()
