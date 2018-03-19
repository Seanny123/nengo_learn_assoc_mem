import nengo

import numpy as np


class RecAdapt(nengo.Network):

    def __init__(self, n_neurons: int, start_weights: np.ndarray,
                 learning_rate=1e-9, max_inhib=-1., theta_tau=1.0,
                 sample_every=0.1, dt=0.001,
                 label=None, seed=None, add_to_container=None):
        super().__init__(label, seed, add_to_container)

        self.kappa = learning_rate * dt
        self.weights = start_weights.copy()
        self.n_nrns = n_neurons
        self.max_inhib = max_inhib

        self.dt = dt
        self.period = sample_every / dt
        self.weight_history = []
        self.lowpass = nengo.Lowpass(theta_tau).make_step(n_neurons, n_neurons, dt, None)

        self.in_rates = np.zeros(n_neurons)
        self.enabled = 0.

        self.in_neurons = nengo.Node(self.set_in_rates, size_in=n_neurons)
        self.enable = nengo.Node(self.set_enabled, size_in=1)
        self.output = nengo.Node(self.bcm_func, size_out=n_neurons)

    def set_in_rates(self, t, x):
        self.in_rates = x

    def set_enabled(self, t, x):
        self.enabled = x

    def bcm_func(self, t):
        theta = self.lowpass(t, self.in_rates)

        self.weights += self.kappa * theta * self.enabled
        self.weights[self.weights < self.max_inhib] = self.max_inhib

        if ((t - self.dt) / self.dt % self.period) < 1:
            self.weight_history.append(self.weights.copy())

        return self.weights * self.in_rates
