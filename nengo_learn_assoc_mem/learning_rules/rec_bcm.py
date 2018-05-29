import nengo

import numpy as np


class RecBCM(nengo.Network):

    def __init__(self, n_neurons: int, start_weights: np.ndarray,
                 learning_rate=1e-9, threshold=150, theta_tau=1.0, max_inhib=-1.,
                 sample_every=0.1, dt=0.001,
                 label=None, seed=None, add_to_container=None):
        super().__init__(label, seed, add_to_container)
        self.kappa = learning_rate * dt
        self.weights = start_weights.copy()
        self.n_nrns = n_neurons
        self.thresh = threshold
        self.max_inhib = max_inhib

        self.dt = dt
        self.period = sample_every / dt
        self.weight_history = []
        self.lowpass = nengo.Lowpass(theta_tau).make_step(n_neurons, n_neurons, dt, None)

        self.in_rates = np.zeros(n_neurons)
        self.out_rates = np.zeros(n_neurons)
        self.enabled = 0.

        self.in_neurons = nengo.Node(self.set_in_rates, size_in=n_neurons)
        self.out_neurons = nengo.Node(self.set_out_rates, size_in=n_neurons)
        self.enable = nengo.Node(self.set_enabled, size_in=1)
        self.output = nengo.Node(self.bcm_func, size_out=n_neurons)

    def set_in_rates(self, t, x):
        self.in_rates = x

    def set_out_rates(self, t, x):
        self.out_rates = x

    def set_enabled(self, t, x):
        self.enabled = x

    def bcm_func(self, t):
        theta = self.lowpass(t, self.out_rates)

        self.in_rates[self.in_rates < self.thresh] = 0.
        self.weights += np.outer(self.kappa * (self.out_rates - theta), self.in_rates) * self.enabled
        self.weights[self.weights > 0.] = 0.
        self.weights[self.weights < self.max_inhib] = self.max_inhib

        np.fill_diagonal(self.weights, 0)

        if ((t - self.dt) / self.dt % self.period) < 1:
            self.weight_history.append(self.weights.copy())

        return np.dot(self.weights, self.in_rates)


def pos_rec_bcm(activities: np.ndarray, base_inhib=-1e-4, max_excite=1e-3) -> np.ndarray:
    n_items = activities.shape[0]
    n_neurons = activities.shape[1]

    act_corr = np.zeros((n_neurons, n_neurons), dtype=np.float)

    for item in range(n_items):
        act_corr += np.outer(activities[item], activities[item])
    np.fill_diagonal(act_corr, 0)

    pos_corr = act_corr[act_corr > 0.]
    min_pos_corr = np.min(pos_corr)

    max_corr = np.max(act_corr)

    rec_w = np.zeros((n_neurons, n_neurons), dtype=np.float)
    rec_w[act_corr > 0.] = np.interp(pos_corr,
                                     (min_pos_corr, max_corr),
                                     (base_inhib, max_excite))
    np.fill_diagonal(rec_w, 0)

    return rec_w


def mean_rec_bcm(activities: np.ndarray,
                 base_inhib=-1e-4, max_excite=1e-3, max_inhib=-1e-3) -> np.ndarray:
    n_items = activities.shape[0]
    n_neurons = activities.shape[1]

    act_corr = np.zeros((n_neurons, n_neurons), dtype=np.float)
    mean_act = activities - np.mean(activities, axis=0)

    for item in range(n_items):
        act_corr += np.outer(mean_act[item], mean_act[item])
    np.fill_diagonal(act_corr, 0)

    max_corr = np.max(act_corr)
    min_corr = np.min(act_corr)

    pos_corr = act_corr[act_corr > 0.]
    min_pos_corr = np.min(pos_corr)

    neg_corr = act_corr[act_corr < 0.]
    max_neg_corr = np.max(neg_corr)

    rec_w = np.ones((n_neurons, n_neurons), dtype=np.float) * base_inhib
    rec_w[act_corr > 0.] = np.interp(pos_corr, (min_pos_corr, max_corr), (base_inhib, max_excite))
    rec_w[act_corr < 0.] = np.interp(neg_corr, (min_corr, max_neg_corr), (max_inhib, base_inhib))
    np.fill_diagonal(rec_w, 0)

    return rec_w
