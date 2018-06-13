import numpy as np

from nengo_learn_assoc_mem.learning_rules.fake_voja import FakeVoja


class NegVoja(FakeVoja):

    def __init__(self, encoders: np.ndarray, post_tau=0.005, learning_rate=1e-3, radius=1.):
        super().__init__(encoders, post_tau, learning_rate)
        self.radius = radius

    def encode(self, t):
        delta = self.enabled * self.learning_rate * self.acts[:, None] * (self.encoders - self.in_sig)
        mod_enc = self.encoders + delta
        mag = np.linalg.norm(mod_enc, axis=1)
        self.encoders = self.radius / mag[:, None] * mod_enc

        if (t / self.dt % self.period) < 1:
            self.encoder_hist.append(self.encoders.copy())

        return np.dot(self.encoders, self.in_sig)
