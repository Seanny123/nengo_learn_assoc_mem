import nengo
from nengo.utils.matplotlib import rasterplot

import numpy as np
import matplotlib.pyplot as plt

from nengo_learn_assoc_mem.learning_rules.fake_voja import FakeVoja
from nengo_learn_assoc_mem.utils import cycle_array


dims = 2
n_neurons = 20
seed = 2
intercepts = [0.8]*n_neurons

rad_comp = 1 / np.sqrt(2)

with nengo.Network() as model:
    ens = nengo.Ensemble(n_neurons, dims, intercepts=intercepts, seed=seed)

with nengo.Simulator(model) as sim:
    pass

enc = sim.data[ens].encoders.copy()

with nengo.Network() as model:
    in_nd = nengo.Node(cycle_array([[-rad_comp, -rad_comp], [rad_comp, rad_comp]], 0.1))
    enabled = nengo.Node(lambda t: 0. if t < 0.1 else 1.)

    fake_voja = FakeVoja(enc, learning_rate=-1e-3)
    ens = nengo.Ensemble(n_neurons, dims, intercepts=intercepts, seed=seed)

    nengo.Connection(in_nd, fake_voja.input_signal, synapse=None)
    nengo.Connection(ens.neurons, fake_voja.input_activities, synapse=0)
    nengo.Connection(enabled, fake_voja.enable, synapse=None)
    nengo.Connection(fake_voja.output, ens.neurons, synapse=None)

    p_in = nengo.Probe(in_nd)
    p_spikes = nengo.Probe(ens.neurons)

with nengo.Simulator(model) as sim:
    sim.run(1)

# basic plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(sim.trange(), sim.data[p_in])
rasterplot(sim.trange(), sim.data[p_spikes], ax=ax2)
plt.show()

x_val = np.linspace(0, 2*np.pi, 50)
circ = np.array([np.cos(x_val), np.sin(x_val)]).T
enc_h = fake_voja.encoder_hist

# first iteration plot
plt.figure()

win_pre = 10
win_a = 100

plt.scatter(circ[:, 0], circ[:, 1], color='k', alpha=0.2)
plt.scatter(sim.data[p_in][win_pre][0], sim.data[p_in][win_pre][1], label="stim", s=200)

plt.scatter(enc_h[0][:, 0], enc_h[0][:, 1], label="orig", s=100)
plt.scatter(enc_h[win_pre][:, 0], enc_h[win_pre][:, 1], label="pre", s=50)
plt.scatter(enc_h[win_a][:, 0], enc_h[win_a][:, 1], label="a", s=50)

plt.legend()
plt.show()

# second iteration plot
plt.figure()

win_pre = 210
win_a = 300

plt.scatter(circ[:, 0], circ[:, 1], color='k', alpha=0.2)
plt.scatter(sim.data[p_in][win_pre][0], sim.data[p_in][win_pre][1], label="stim", s=200)

plt.scatter(enc_h[0][:, 0], enc_h[0][:, 1], label="orig", s=100)
plt.scatter(enc_h[win_pre][:, 0], enc_h[win_pre][:, 1], label="pre", s=50)
plt.scatter(enc_h[win_a][:, 0], enc_h[win_a][:, 1], label="a", s=50)

plt.legend()
plt.show()
