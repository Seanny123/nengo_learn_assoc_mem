import numpy as np

from nengo_learn_assoc_mem.learning_rules.fake_voja import FakeVoja


class NegVoja(FakeVoja):

    def __init__(self, encoders: np.ndarray, post_tau=0.005, learning_rate=1e-3, radius=1.0):
        super().__init__(encoders, post_tau, learning_rate)
        self.radius = radius

    def encode(self, t, x):
        in_sig = x[:self.dims]
        acts = x[self.dims:]

        self.encoder_hist.append(self.encoders.copy())
        delta = self.learning_rate * acts[:, None] * (self.encoders - in_sig)
        mod_enc = self.encoders + delta
        mag = np.linalg.norm(mod_enc, axis=1)
        self.encoders = self.radius / mag[:, None] * mod_enc

        return np.dot(self.encoders, in_sig)
