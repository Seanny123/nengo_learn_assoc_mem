import nengo
import numpy as np


class LearningAssocMem(nengo.Network):
    def __init__(self, n_neurons, dimensions,
                 intercepts=nengo.dists.Uniform(0, 0),
                 voja_tau=0.005,
                 voja_rate=1e-3,
                 pes_rate=1e-3,
                 label=None,
                 inhibit_synapse=0.01,
                 inhibit_strength=0.0005,
                 seed=None,
                 load_from=None,
                 inhibit_all=False,
                 ):
        super(LearningAssocMem, self).__init__(label=label)
        if load_from is not None:
            data = np.load(load_from)
            encoders = data['enc']
            decoders = data['dec']
            if seed is None:
                seed = int(data['seed'])
            else:
                assert seed == int(data['seed'])
        else:
            encoders = nengo.Default
            decoders = np.zeros((dimensions, n_neurons), dtype=float)

        self.seed = seed

        with self:
            self.mem = nengo.Ensemble(n_neurons=n_neurons,
                                      dimensions=dimensions,
                                      intercepts=intercepts,
                                      # max_rates=nengo.dists.Uniform(150,150),
                                      encoders=encoders,
                                      seed=seed,
                                      )
            self.input = nengo.Node(None, size_in=dimensions)
            self.output = nengo.Node(None, size_in=dimensions)
            self.correct = nengo.Node(None, size_in=dimensions)

            if voja_rate > 0:
                learning_rule_type = nengo.Voja(post_tau=voja_tau,
                                                learning_rate=voja_rate)
            else:
                learning_rule_type = None
            self.conn_in = nengo.Connection(self.input, self.mem,
                                            learning_rule_type=learning_rule_type)

            if pes_rate > 0:
                learning_rule_type = nengo.PES(learning_rate=pes_rate)
            else:
                learning_rule_type = None

            self.conn_out = nengo.Connection(self.mem.neurons, self.output,
                                             transform=decoders,
                                             learning_rule_type=learning_rule_type,
                                             )

            if pes_rate > 0:
                self.learn_control = nengo.Node(
                    lambda t, x: x[:-1] if x[-1] < 0.5 else x[:-1] * 0,
                    size_in=dimensions + 1)
                nengo.Connection(self.learn_control,
                                 self.conn_out.learning_rule,
                                 )
                nengo.Connection(self.output, self.learn_control[:-1],
                                 synapse=None)
                nengo.Connection(self.correct, self.learn_control[:-1],
                                 transform=-1, synapse=None)
                self.stop_pes = nengo.Node(None, size_in=1)
                nengo.Connection(self.stop_pes, self.learn_control[-1],
                                 synapse=None)

            if inhibit_all:
                inhibit = nengo.Node(None, size_in=1)
                nengo.Connection(self.mem.neurons, inhibit,
                                 transform=inhibit_strength * np.ones((1, n_neurons)),
                                 synapse=None)
                nengo.Connection(inhibit, self.mem.neurons,
                                 transform=-np.ones((self.mem.n_neurons, 1)),
                                 synapse=inhibit_synapse)

            else:
                nengo.Connection(self.mem.neurons, self.mem.neurons,
                                 transform=inhibit_strength * (np.eye(self.mem.n_neurons) - 1),
                                 synapse=inhibit_synapse)

    def create_weight_probes(self):
        with self:
            self.probe_encoder = nengo.Probe(self.mem, 'scaled_encoders',
                                             sample_every=0.1)
            self.probe_decoder = nengo.Probe(self.conn_out, 'weights',
                                             sample_every=0.1)

    def save(self, filename, sim):
        enc = sim.data[self.probe_encoder][-1]
        dec = sim.data[self.probe_decoder][-1]
        np.savez(filename, enc=enc, dec=dec, seed=self.seed)

    def save_all_weights(self, filename, sim):
        enc = sim.data[self.probe_encoder]
        dec = sim.data[self.probe_decoder]
        np.savez(filename, enc=enc, dec=dec)
