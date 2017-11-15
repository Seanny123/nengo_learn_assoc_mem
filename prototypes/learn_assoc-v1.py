import nengo
import nengo.spa as spa
import numpy as np

D = 32

items = ['A', 'B', 'C', 'D', 'E', 'F']
vocab = spa.Vocabulary(D)
isi = 0.5

model = nengo.Network()
with model:

    def stim(t):
        index = int(t / isi)
        rng = np.random.RandomState(seed=index)
        scale = rng.uniform(0.5, 1.0)
        return vocab.parse('%g*%s' % (scale, items[index % len(items)])).v
    stim = nengo.Node(stim)

    intercepts = 0.0

    ens = nengo.Ensemble(n_neurons=10, dimensions=D,
                         intercepts=nengo.dists.Choice([intercepts]))
    nengo.Connection(stim, ens,
                     learning_rule_type=nengo.Voja(post_tau=0.005,
                                                   learning_rate=1e-3))

    output = nengo.Node(None, size_in=D)

    conn_out = nengo.Connection(ens, output,
                                learning_rule_type=nengo.PES(learning_rate=1e-3),
                                function=lambda x: np.zeros(D))

    nengo.Connection(output, conn_out.learning_rule)
    nengo.Connection(stim, conn_out.learning_rule, transform=-1)

    display = spa.State(D, vocab=vocab)
    for e in display.all_ensembles:
        e.neuron_type = nengo.Direct()
    nengo.Connection(output, display.input)

    nengo.Connection(ens.neurons, ens.neurons,
                     transform=0.0005*(np.eye(ens.n_neurons)-1),
                     synapse=0.01)
