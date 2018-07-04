import nengo
import numpy as np

from nengo_learn_assoc_mem.learning_rules.mixed_voja import MeanMixed

seed = 8
dimensions = 2
n_neurons = 50

x_val = np.linspace(0, 2*np.pi, n_neurons)
enc = np.array([np.cos(x_val), np.sin(x_val)]).T

with nengo.Network() as model:
    in_nd = nengo.Node([1, 1])
    nvoja_learn = nengo.Node(1)

    ens = nengo.Ensemble(n_neurons, dimensions, seed=seed)

    neg_voja = MeanMixed(enc)

    nengo.Connection(in_nd, neg_voja.input_signal, synapse=None)
    nengo.Connection(ens.neurons, neg_voja.input_activities, synapse=0)
    nengo.Connection(nvoja_learn, neg_voja.enable, synapse=None)
    nengo.Connection(neg_voja.output, ens.neurons, synapse=None)

with nengo.Simulator(model, progress_bar=None) as sim:
    sim.run(1.)
