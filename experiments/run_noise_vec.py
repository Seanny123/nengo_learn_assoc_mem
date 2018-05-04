import string
from typing import List, Dict

import nengo
from nengo.utils.ensemble import tuning_curves
import nengolib
import nengo_spa as spa

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nengo_learn_assoc_mem.utils import BasicVecFeed

seed = 8
D = 8
n_neurons = 200
n_items = 10
intercepts = np.ones(n_neurons) * 0.35
rng = np.random.RandomState(seed)

vocab = spa.Vocabulary(D, max_similarity=0.35, rng=rng)
vocab.populate(";".join([string.ascii_uppercase[i] for i in range(n_items)]))

dt = 0.001
t_present = 0.3
t_pause = 0.0
n_repeats = 5
t_each = t_present + t_pause


def conf_metric(ss_data: np.ndarray) -> Dict:
    correct = False

    smoothed = np.mean(ss_data, axis=0)
    winner = np.argmax(smoothed)
    mask = np.ones(n_items, dtype=bool)
    mask[winner] = False
    runnerup = np.argmax(smoothed[mask])
    runnerup_dist = smoothed[winner] - smoothed[runnerup]

    if runnerup_dist > 0:
        correct = True

    return dict(correct=correct, top_mag=smoothed[winner], runnerup_dist=runnerup_dist)


def train_mem(vecs: List[np.ndarray], voja_learn_rate=1e-5, pes_learn_rate=1e-3):
    feed = BasicVecFeed(vecs, vecs, t_present, D, len(vecs), t_pause)

    with nengolib.Network(seed=seed) as train_model:
        in_nd = nengo.Node(feed.feed)
        learning = nengo.Node(lambda t: -feed.paused)
        correct_answer = nengo.Node(feed.get_answer)
        output = nengo.Node(size_in=D)

        ens = nengo.Ensemble(n_neurons, D, intercepts=intercepts, neuron_type=nengo.LIF())

        in_conn = nengo.Connection(in_nd, ens,
                                   learning_rule_type=nengo.Voja(voja_learn_rate),
                                   synapse=None)
        nengo.Connection(learning, in_conn.learning_rule, synapse=None)
        conn_out = nengo.Connection(ens, output,
                                    learning_rule_type=nengo.PES(pes_learn_rate))

        # Error flow node
        pes_learn_control = nengo.Node(
            lambda t, x: x[:-1] if x[-1] >= 0 else x[:-1] * 0,
            size_in=D + 1)
        nengo.Connection(pes_learn_control,
                         conn_out.learning_rule)

        # Error calculation connections
        nengo.Connection(output, pes_learn_control[:-1],
                         synapse=None)
        nengo.Connection(correct_answer, pes_learn_control[:-1],
                         transform=-1, synapse=None)
        # Control connection
        nengo.Connection(learning, pes_learn_control[-1],
                         synapse=None)

        p_in = nengo.Probe(in_nd)
        p_enc = nengo.Probe(ens, 'scaled_encoders', sample_every=0.05)
        p_dec = nengo.Probe(conn_out, 'weights', sample_every=0.1)
        p_out = nengo.Probe(output, synapse=0.01)

    with nengo.Simulator(train_model) as train_sim:
        train_sim.run(n_repeats * len(vecs) * t_each + t_pause)

    enc = train_sim.data[p_enc][-1]
    dec = train_sim.data[p_dec][-1]

    return enc, dec


def test_mem(enc: np.ndarray, dec: np.ndarray, in_vec: List[np.ndarray],
             noise_mag: float, noise_synapse=None,
             rec_w=None, rec_synapse=0.01) -> np.ndarray:
    feed = BasicVecFeed(in_vec, in_vec, t_present, D, len(in_vec), t_pause)

    with nengolib.Network(seed=seed) as test_model:
        vec_nd = nengo.Node(feed.feed)
        in_nd = nengo.Node(size_in=D)
        output = nengo.Node(size_in=D)
        pause = nengo.Node(lambda t: feed.paused)

        ens = nengo.Ensemble(n_neurons, D, encoders=enc, intercepts=intercepts)

        if noise_mag > 0:
            noise_nd = nengo.Node(nengo.processes.WhiteNoise(
                nengo.dists.Gaussian(.0, 0.1)), size_out=D)
            nengo.Connection(noise_nd, in_nd, transform=noise_mag,
                             synapse=noise_synapse)

        nengo.Connection(vec_nd, in_nd, synapse=None)
        nengo.Connection(in_nd, ens, synapse=None)
        nengo.Connection(pause, ens.neurons, transform=-10 * np.ones((n_neurons, 1)))
        nengo.Connection(ens.neurons, output, transform=dec)

        if rec_w is not None:
            nengo.Connection(ens.neurons, ens.neurons,
                             transform=rec_w, synapse=rec_synapse)

        p_in = nengo.Probe(in_nd, synapse=0.01)
        p_out = nengo.Probe(output, synapse=0.01)

    with nengo.Simulator(test_model) as test_sim:
        test_sim.run(n_repeats * t_each + t_pause)

    return test_sim.data[p_out]


def rec_bcm(vecs: np.ndarray, enc: np.ndarray, base_inhib=-1e-4, max_excite=1e-3) -> np.ndarray:
    with nengolib.Network(seed=seed) as model:
        ens = nengo.Ensemble(n_neurons, D, encoders=enc, intercepts=intercepts)

    with nengo.Simulator(model) as sim:
        pass

    _, activities = tuning_curves(ens, sim, inputs=vecs)

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

    return rec_w


encoders, decoders = train_mem(list(vocab.vectors))
rec_weights = rec_bcm(vocab.vectors, encoders)

test_vec = [vocab['J'].v * 0.8]

# TODO: Run multiple iterations
# TODO: Save results somehow
# TODO: Plot the confidence interval as a function of approaches
for noise_magnitude in (0.0, 0.1, 0.2):

    base_res = test_mem(encoders, decoders, test_vec, noise_magnitude)
    low_syn_res = test_mem(encoders, decoders, test_vec, noise_magnitude,
                           rec_w=rec_weights, rec_synapse=0.001)
    rec_res = test_mem(encoders, decoders, test_vec, noise_magnitude,
                       rec_w=rec_weights, rec_synapse=0.005)
    high_syn_res = test_mem(encoders, decoders, test_vec, noise_magnitude,
                            rec_w=rec_weights, rec_synapse=0.01)

    base_conf = conf_metric(spa.similarity(base_res, vocab))
    rec_conf = conf_metric(spa.similarity(rec_res, vocab))
