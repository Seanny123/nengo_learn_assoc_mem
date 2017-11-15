import nengo
import nengo.spa as spa
import numpy as np

D = 32

items = ['A', 'B', 'C', 'D', 'E', 'F']
vocab = spa.Vocabulary(D)
isi = 0.5


class LearningAssocMem(nengo.Network):

    def __init__(self, n_neurons, dimensions,
                 intercepts=nengo.dists.Uniform(0,0),
                 voja_tau=0.005,
                 voja_rate=1e-3,
                 pes_rate=1e-3,
                 label=None,
                 inhibit_synapse=0.01,
                 inhibit_strength=0.0005,
                 ):
        super(LearningAssocMem, self).__init__(label=label)

        with self:
            self.mem = nengo.Ensemble(n_neurons=n_neurons,
                                      dimensions=dimensions,
                                      intercepts=intercepts,
                                      )
            self.input = nengo.Node(None, size_in=dimensions)
            self.output = nengo.Node(None, size_in=dimensions)

            nengo.Connection(self.input, self.mem,
                             learning_rule_type=nengo.Voja(post_tau=voja_tau,
                                                           learning_rate=voja_rate))

            conn_out = nengo.Connection(self.mem, self.output,
                                        learning_rule_type=nengo.PES(learning_rate=pes_rate),
                                        function=lambda x: np.zeros(D))

            nengo.Connection(self.output, conn_out.learning_rule)
            nengo.Connection(self.input, conn_out.learning_rule, transform=-1)

            nengo.Connection(self.mem.neurons, self.mem.neurons,
                             transform=inhibit_strength*(np.eye(self.mem.n_neurons)-1),
                             synapse=inhibit_synapse)


model = nengo.Network()
with model:

    def stim(t):
        index = int(t / isi)
        rng = np.random.RandomState(seed=index)
        scale = rng.uniform(0.5, 1.0)
        return vocab.parse('%g*%s' %(scale, items[index % len(items)])).v
    stim = nengo.Node(stim)

    mem = LearningAssocMem(n_neurons=10, dimensions=D)

    nengo.Connection(stim, mem.input)

    display = spa.State(D, vocab=vocab)
    for e in display.all_ensembles:
        e.neuron_type = nengo.Direct()
    nengo.Connection(mem.output, display.input)
