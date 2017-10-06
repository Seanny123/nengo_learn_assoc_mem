import nengo

import numpy as np


class FakeBCM(object):

    def __init__(self, learning_rate=1e-9, in_neurons=4, out_neurons=2, theta_tau=1.0,
                 sample_every=0.1, start_weights=None, dt=0.001):
        self.kappa = learning_rate * dt
        assert start_weights is not None
        self.omega = start_weights.copy()
        self.in_nrns = in_neurons
        self.lowpass = nengo.Lowpass(theta_tau).make_step(out_neurons, out_neurons, dt, None)
        self.weight_history = []
        self.period = sample_every / dt
        self.dt = dt

    def bcm_func(self, t, x):
        in_rates = x[:self.in_nrns]
        out_rates = x[self.in_nrns:]
        theta = self.lowpass(t, out_rates)

        self.omega += np.outer(self.kappa * out_rates * (out_rates - theta), in_rates)

        if (t / self.dt % self.period) < 1:
            self.weight_history.append(self.omega.copy())

        return np.dot(self.omega, in_rates)
