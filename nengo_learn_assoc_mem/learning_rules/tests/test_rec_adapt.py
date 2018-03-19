import nengo
import numpy as np
import matplotlib.pyplot as plt

from nengo_learn_assoc_mem.learning_rules.rec_adapt import RecAdapt


def stim_func(t):
    if t < 0.5:
        return 0.3
    elif 1.0 < t < 1.5:
        return 0.3
    else:
        return -0.3


ens_params = dict(encoders=[[1], [1], [-1], [-1]], intercepts=[-0.5, -0.1, -0.1, -0.5], max_rates=[250, 300, 300, 250])
adapt_lr = -5e-6

n_neurons = 4
sim_len = 8

with nengo.Network() as model:
    stim = nengo.Node(stim_func)
    enable = nengo.Node(1)

    ens = nengo.Ensemble(n_neurons, 1, **ens_params)
    rec_learn = RecAdapt(n_neurons, np.zeros(n_neurons), learning_rate=adapt_lr,
                         max_inhib=-0.002)

    nengo.Connection(stim, ens, synapse=None)
    nengo.Connection(enable, rec_learn.enable, synapse=None)
    nengo.Connection(ens.neurons, rec_learn.in_neurons, synapse=0.1)
    nengo.Connection(rec_learn.output, ens.neurons, synapse=None)

    p_spikes = nengo.Probe(ens.neurons)
    p_out = nengo.Probe(ens.neurons, synapse=0.05)

with nengo.Simulator(model) as sim:
    sim.run(sim_len)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(sim.trange(), sim.data[p_out])

w_hist = np.array(rec_learn.weight_history)
ax2.plot(sim.trange(dt=0.1), w_hist[:-2])
plt.show()
