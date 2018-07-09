import numpy as np

from nengo_learn_assoc_mem.learning_rules.fake_voja import FakeVoja


class StaticMixed(FakeVoja):

    def __init__(self, encoders: np.ndarray, max_rates: np.ndarray, thresh: float, max_dist=0.2,
                 post_tau=0.005, learning_rate=1e-3, radius=1., sample_every=0.1):
        super().__init__(encoders, post_tau, learning_rate, sample_every)
        self.radius = radius
        assert 0. < thresh < 1.
        self.thresh = thresh
        self.max_dist = max_dist
        self.max_rates = max_rates

    def encode(self, t):
        firing_ratio = self.acts / self.max_rates
        assert np.all(firing_ratio <= 1.1)

        lr = self.enabled * self.learning_rate

        dist = (self.encoders - self.in_sig)
        dist_mag = np.linalg.norm(dist, axis=1)
        dist[dist_mag > self.max_dist] = 0.

        delta = lr * (firing_ratio[:, None] - self.thresh) * dist

        mod_enc = self.encoders + delta
        mag = np.linalg.norm(mod_enc, axis=1)
        self.encoders = self.radius / mag[:, None] * mod_enc

        if (t / self.dt % self.period) < 1:
            self.encoder_hist.append(self.encoders.copy())

        return np.dot(self.encoders, self.in_sig)


class MeanMixed(FakeVoja):

    def __init__(self, encoders: np.ndarray, max_rates: np.ndarray, bias=1., max_dist=0.2,
                 post_tau=0.005, learning_rate=1e-3, radius=1., sample_every=0.1):
        super().__init__(encoders, post_tau, learning_rate, sample_every)
        self.radius = radius
        self.bias = bias
        self.max_dist = max_dist
        self.max_rates = max_rates

    def encode(self, t):
        firing_ratio = self.acts / self.max_rates
        assert np.all(firing_ratio <= 1.1)

        threshold = self.bias * np.mean(firing_ratio)
        lr = self.enabled * self.learning_rate

        dist = (self.encoders - self.in_sig)
        dist_mag = np.linalg.norm(dist, axis=1)
        dist[dist_mag > self.max_dist] = 0.

        delta = lr * (firing_ratio[:, None] - threshold) * dist

        mod_enc = self.encoders + delta
        mag = np.linalg.norm(mod_enc, axis=1)
        self.encoders = self.radius / mag[:, None] * mod_enc

        if (t / self.dt % self.period) < 1:
            self.encoder_hist.append(self.encoders.copy())

        return np.dot(self.encoders, self.in_sig)
