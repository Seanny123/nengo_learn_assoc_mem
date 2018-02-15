"""Generating encoders and assigning them doesn't work for some reason."""

import nengo

import numpy as np
import matplotlib.pyplot as plt

n_neurons = 300
seed = 0

with nengo.Network() as model:
    ens = nengo.Ensemble(n_neurons, 1, seed=seed)

with nengo.Simulator(model) as sim:
    pass

enc = sim.data[ens].encoders

with nengo.Network() as model:
    in_nd = nengo.Node(lambda t: np.sin(2*np.pi*t*5))
    fake_enc = nengo.Node(lambda t, x: np.dot(enc, x), size_in=1)
    ens = nengo.Ensemble(n_neurons, 1, seed=seed)

    nengo.Connection(in_nd, fake_enc, synapse=None)
    nengo.Connection(fake_enc, ens.neurons, synapse=None)

    p_in = nengo.Probe(in_nd, synapse=0.01)
    p_out = nengo.Probe(ens, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(1)

plt.plot(sim.trange(), sim.data[p_in])
plt.plot(sim.trange(), sim.data[p_out])
plt.show()
