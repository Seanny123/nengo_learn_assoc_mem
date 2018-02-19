import nengo

import numpy as np


class FakeVoja(nengo.Network):

    def __init__(self, encoders: np.ndarray, post_tau=0.005, learning_rate=1e-3,
                 label=None, seed=None, add_to_container=None):
        super().__init__(label, seed, add_to_container)

        self.dims = encoders.shape[1]
        self.encoders = encoders
        self.post_tau = post_tau
        self.learning_rate = learning_rate
        self.encoder_hist = []

        self.acts = np.zeros(encoders.shape[0])
        self.in_sig = np.zeros(self.dims)
        self.enabled = 0.

        self.input_signal = nengo.Node(self.set_sig, size_in=self.dims)
        self.input_activities = nengo.Node(self.set_acts, size_in=encoders.shape[0])
        self.enable = nengo.Node(self.set_enabled, size_in=1)
        self.output = nengo.Node(self.encode, size_out=encoders.shape[0])

    def set_acts(self, t, x):
        self.acts = x

    def set_sig(self, t, x):
        self.in_sig = x

    def set_enabled(self, t, x):
        self.enabled = x

    def encode(self, t):
        self.encoder_hist.append(self.encoders.copy())
        self.encoders += self.enabled * self.learning_rate * self.acts[:, None] * (self.encoders - self.in_sig)

        return np.dot(self.encoders, self.in_sig)
