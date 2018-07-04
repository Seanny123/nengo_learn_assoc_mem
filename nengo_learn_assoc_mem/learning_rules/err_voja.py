import nengo
import numpy as np

from nengo_learn_assoc_mem.learning_rules.fake_voja import FakeVoja


class ErrVoja(FakeVoja):

    def __init__(self, encoders: np.ndarray, bias: float, post_tau=0.005, learning_rate=1e-3, radius=1.):
        super().__init__(encoders, post_tau, learning_rate)
        self.radius = radius
        self.bias = np.ones(self.dims) * bias
        self.err = np.zeros(self.dims)

        self.error = nengo.Node(self.set_err, size_in=self.dims)

    def set_err(self, t, x):
        self.err = x

    def encode(self, t):
        lr = self.enabled * self.learning_rate * (self.bias - np.abs(self.err))
        delta = lr * self.acts[:, None] * (self.encoders - self.in_sig)
        mod_enc = self.encoders + delta
        mag = np.linalg.norm(mod_enc, axis=1)
        self.encoders = self.radius / mag[:, None] * mod_enc

        if (t / self.dt % self.period) < 1:
            self.encoder_hist.append(self.encoders.copy())

        return np.dot(self.encoders, self.in_sig)


class ErrFiringRateVoja(ErrVoja):

    def __init__(self, encoders: np.ndarray, bias: float, post_tau=0.005, learning_rate=1e-3, radius=1.):
        super().__init__(encoders, bias, post_tau, learning_rate, radius)
        self.bias = bias

    def encode(self, t):
        threshold = (self.bias - np.sum(np.abs(self.err))) * np.max(self.acts)

        lr = self.enabled * self.learning_rate
        delta = lr * (self.acts[:, None] - threshold) * (self.encoders - self.in_sig)
        mod_enc = self.encoders + delta
        mag = np.linalg.norm(mod_enc, axis=1)
        self.encoders = self.radius / mag[:, None] * mod_enc

        if (t / self.dt % self.period) < 1:
            self.encoder_hist.append(self.encoders.copy())

        return np.dot(self.encoders, self.in_sig)
