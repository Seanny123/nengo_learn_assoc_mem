import numpy as np

from nengo_learn_assoc_mem.learning_rules.fake_voja import FakeVoja


class MaxMixed(FakeVoja):

    def __init__(self, encoders: np.ndarray, bias: float, post_tau=0.005, learning_rate=1e-3, radius=1.):
        super().__init__(encoders, post_tau, learning_rate)
        self.radius = radius
        self.bias = bias

    def encode(self, t):
        # highly reacting neurons: 1000-500 = 500
        # low reacting neurons: 1-500 = -499, ie switches learning around
        # problem is tons of zeros, so let's assume it knows the max

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

    def __init__(self, encoders: np.ndarray, bias=1., post_tau=0.005, learning_rate=1e-3, radius=1.):
        super().__init__(encoders, post_tau, learning_rate)
        self.radius = radius
        self.bias = bias

    def encode(self, t):
        threshold = self.bias * np.mean(self.acts)

        lr = self.enabled * self.learning_rate
        delta = lr * (self.acts[:, None] - threshold) * (self.encoders - self.in_sig)
        mod_enc = self.encoders + delta
        mag = np.linalg.norm(mod_enc, axis=1)
        self.encoders = self.radius / mag[:, None] * mod_enc

        if (t / self.dt % self.period) < 1:
            self.encoder_hist.append(self.encoders.copy())

        return np.dot(self.encoders, self.in_sig)
