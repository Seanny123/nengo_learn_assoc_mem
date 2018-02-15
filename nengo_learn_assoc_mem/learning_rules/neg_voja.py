import numpy as np

from nengo_learn_assoc_mem.learning_rules.fake_voja import FakeVoja


class NegVoja(FakeVoja):

    def __init__(self, encoders: np.ndarray, post_tau=0.005, learning_rate=1e-3):
        super().__init__(encoders, post_tau, learning_rate)

    def encode(self, t, x):
        in_sig = x[:self.dims]
        acts = x[self.dims:]

        self.encoder_hist.append(self.encoders.copy())
        self.encoders += self.learning_rate * acts[:, None] * (self.encoders - in_sig)

        return np.dot(self.encoders, in_sig)
