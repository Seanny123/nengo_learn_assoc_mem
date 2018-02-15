import nengo

import matplotlib.pyplot as plt

from nengo_learn_assoc_mem.learning_rules.fake_voja import FakeVoja


def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period / dt))
    if i_every != period / dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))

    def f(t):
        i = int(round((t - dt) / dt))  # t starts at dt
        return x[int(i / i_every) % len(x)]

    return f


dims = 2
n_neurons = 10
seed = 0
intercepts = [0.5]*n_neurons

with nengo.Network() as model:
    ens = nengo.Ensemble(n_neurons, dims, intercepts=intercepts, seed=seed)

with nengo.Simulator(model) as sim:
    pass

enc = sim.data[ens].encoders
fake_voja = FakeVoja(enc)

with nengo.Network() as model:
    in_nd = nengo.Node(cycle_array([[1, 1], [-1, -1]], 0.1))

    voja_nd = nengo.Node(fake_voja.encode, size_in=dims+n_neurons)
    ens = nengo.Ensemble(n_neurons, dims, intercepts=intercepts, seed=seed)

    nengo.Connection(in_nd, voja_nd[:dims], synapse=None)
    nengo.Connection(ens.neurons, voja_nd[dims:])
    nengo.Connection(voja_nd, ens.neurons, synapse=None)

    p_in = nengo.Probe(in_nd, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(1)

plt.plot(sim.trange(), sim.data[p_in])
plt.show()
