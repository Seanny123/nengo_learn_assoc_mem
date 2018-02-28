import nengo
from nengo.utils.matplotlib import rasterplot

import numpy as np
import matplotlib.pyplot as plt

from nengo_learn_assoc_mem.learning_rules.rec_bcm import RecBCM


def rec_legend(root: str, li):
    return [f"{root}$\\rightarrow${ll}" for ll in li]


def enable_func(t):
    if t > 1.:
        return 1.
    else:
        return 0.


def stim_func(t):
    if (t % 1) > 0.5:
        return 0.3
    else:
        return -0.3


dt = 0.001
seed = 0
sim_len = 10
lr = 5e-6
pre_nrn = 4

ens_params = dict(encoders=[[1], [1], [-1], [-1]], intercepts=[-0.5, -0.1, -0.1, -0.5], max_rates=[250, 300, 300, 250])

rec_inhib = (-1*np.ones(pre_nrn) + np.eye(pre_nrn)) / 1e4

with nengo.Network() as model:
    stim = nengo.Node(stim_func)
    enabled = nengo.Node(enable_func)

    ens = nengo.Ensemble(pre_nrn, 1, **ens_params)

    rec_bcm = RecBCM(pre_nrn, rec_inhib, learning_rate=5e-7, threshold=120, max_inhib=-0.02)

    nengo.Connection(stim, ens, synapse=None)

    nengo.Connection(ens.neurons, rec_bcm.in_neurons, synapse=0.01)
    nengo.Connection(ens.neurons, rec_bcm.out_neurons, synapse=0.01)
    nengo.Connection(enabled, rec_bcm.enable, synapse=None)
    nengo.Connection(rec_bcm.output, ens.neurons, synapse=None)

    p_in = nengo.Probe(stim)
    p_spikes = nengo.Probe(ens.neurons)
    p_out = nengo.Probe(ens, synapse=0.01)


with nengo.Simulator(model) as sim:
    sim.run(sim_len)


w_hist = np.array(rec_bcm.weight_history)
w_hist_trange = np.concatenate(([0], sim.trange(dt=0.1),))

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sim.trange(), sim.data[p_out], label="Post")
plt.ylabel("Decoded value")
plt.ylim(-1.6, 1.6)
plt.legend(loc="lower left")

plt.subplot(3, 1, 2)
plt.plot(w_hist_trange, w_hist[:, 0], alpha=0.6)
plt.ylabel("Connection weights\nfrom first neuron")
plt.legend(rec_legend("A", ("A", "B", "C", "D")))

plt.subplot(3, 1, 3)
plt.plot(w_hist_trange, w_hist[:, -1], alpha=0.6)
plt.ylabel("Connection weight\nfrom last neuron")
plt.legend(rec_legend("D", ("A", "B", "C", "D")))

plt.show()

plt.figure(figsize=(12, 8))
win = int(2e3)

ax = plt.subplot(2, 1, 1)
rasterplot(sim.trange()[:win], sim.data[p_spikes][:win], ax)
ax.set_ylabel('Neuron')
ax.set_yticklabels(("A", "B", "C", "D"))
ax.set_xlabel('Time (s)')
ax.set_title('Before learning')

ax = plt.subplot(2, 1, 2)
rasterplot(sim.trange()[-win:], sim.data[p_spikes][-win:], ax)
ax.set_ylabel('Neuron')
ax.set_yticklabels(("A", "B", "C", "D"))
ax.set_xlabel('Time (s)')
ax.set_title('After learning')

plt.tight_layout()
plt.show()
