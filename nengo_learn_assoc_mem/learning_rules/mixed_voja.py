import numpy as np

from nengo_learn_assoc_mem.learning_rules.fake_voja import FakeVoja

import ipdb


class MaxMixed(FakeVoja):

    def __init__(self, encoders: np.ndarray, bias: float, post_tau=0.005, learning_rate=1e-3, radius=1.):
        super().__init__(encoders, post_tau, learning_rate)
        self.radius = radius
        self.bias = bias

    def encode(self, t):
        threshold = self.bias * np.max(self.acts)

        lr = self.enabled * self.learning_rate
        delta = lr * (self.acts[:, None] - threshold) * (self.encoders - self.in_sig)
        mod_enc = self.encoders + delta
        mag = np.linalg.norm(mod_enc, axis=1)
        self.encoders = self.radius / mag[:, None] * mod_enc

        if (t / self.dt % self.period) < 1:
            self.encoder_hist.append(self.encoders.copy())

        return np.dot(self.encoders, self.in_sig)


class MeanMixed(FakeVoja):

    def __init__(self, encoders: np.ndarray, bias=1., max_dist=0.2, post_tau=0.005, learning_rate=1e-3, radius=1.):
        super().__init__(encoders, post_tau, learning_rate)
        self.radius = radius
        self.bias = bias
        self.max_dist = max_dist

    def encode(self, t):
        threshold = self.bias * np.mean(self.acts)
        lr = self.enabled * self.learning_rate

        dist = (self.encoders - self.in_sig)
        dist_mag = np.linalg.norm(dist, axis=1)
        dist[dist_mag > self.max_dist] = 0.

        delta = lr * (self.acts[:, None] - threshold) * dist

        mod_enc = self.encoders + delta
        mag = np.linalg.norm(mod_enc, axis=1)
        self.encoders = self.radius / mag[:, None] * mod_enc

        if (t / self.dt % self.period) < 1:
            self.encoder_hist.append(self.encoders.copy())

        return np.dot(self.encoders, self.in_sig)
