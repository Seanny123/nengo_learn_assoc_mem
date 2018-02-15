import numpy as np


class FakeVoja(object):

    def __init__(self, encoders: np.ndarray, post_tau=0.005, learning_rate=1e-3):
        self.dims = encoders.shape[1]
        self.encoders = encoders
        self.post_tau = post_tau
        self.learning_rate = learning_rate
        self.encoder_hist = []

    def encode(self, t, x):
        in_sig = x[:self.dims]
        acts = x[self.dims:]

        self.encoder_hist.append(self.encoders.copy())
        self.encoders += self.learning_rate * acts[:, None] * (self.encoders - in_sig)

        return np.dot(self.encoders, in_sig)
