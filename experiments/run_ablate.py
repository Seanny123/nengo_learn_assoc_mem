import string
import os
from typing import List

import nengo
import nengolib
import nengo_spa as spa

import numpy as np
import pandas as pd

from nengo_learn_assoc_mem.learning_rules.rec_bcm import pos_rec_bcm
from nengo_learn_assoc_mem.utils import BasicVecFeed, conf_metric, get_activities
from nengo_learn_assoc_mem.paths import data_path

seed = 8
D = 8
n_neurons = 200
n_items = 10
intercepts = np.ones(n_neurons) * 0.35
rng = np.random.RandomState(seed)

vocab = spa.Vocabulary(D, max_similarity=0.35, rng=rng)
vocab.populate(";".join([string.ascii_uppercase[i] for i in range(n_items)]))

t_present = 0.3


def train_mem(vecs: List[np.ndarray], voja_learn_rate=1e-5, pes_learn_rate=1e-3):
    feed = BasicVecFeed(vecs, vecs, t_present, D, len(vecs), 0.)

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
        train_sim.run(5 * len(vecs) * t_present)

    enc = train_sim.data[p_enc][-1]
    dec = train_sim.data[p_dec][-1]

    return enc, dec


def test_mem(enc: np.ndarray, dec: np.ndarray, in_vec: List[np.ndarray],
             noise_mag: float, ablate_idx: np.array, noise_synapse=None,
             rec_w=None, rec_synapse=0.01) -> np.ndarray:

    feed = BasicVecFeed(in_vec, in_vec, t_present, D, len(in_vec), 0.)

    with nengolib.Network(seed=seed) as test_model:
        vec_nd = nengo.Node(feed.feed)
        in_nd = nengo.Node(size_in=D)
        output = nengo.Node(size_in=D)
        pause = nengo.Node(lambda t: feed.paused)

        ens = nengo.Ensemble(n_neurons, D,
                             encoders=enc, intercepts=intercepts)

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
            ablate_rec_w = rec_w.copy()
            ablate_rec_w[ablate_idx] = 0.
            ablate_rec_w[:, ablate_idx] = 0.
            nengo.Connection(ens.neurons, ens.neurons,
                             transform=ablate_rec_w, synapse=rec_synapse)

        p_in = nengo.Probe(in_nd, synapse=0.01)
        p_out = nengo.Probe(output, synapse=0.01)

    with nengo.Simulator(test_model) as test_sim:
        encoder_sig = test_sim.signals[test_sim.model.sig[ens]['encoders']]
        encoder_sig.setflags(write=True)
        encoder_sig[ablate_idx] = 0.
        encoder_sig.setflags(write=False)
        test_sim.run(t_present)

    return test_sim.data[p_out]


encoders, decoders = train_mem(list(vocab.vectors))
activities = get_activities(vocab.vectors, n_neurons, D, encoders, intercepts, seed)
rec_weights = pos_rec_bcm(activities)

df_cols = ("cor", "mag", "rn_dist", "noise_mag", "dead_num", "rec_w", "rec_syn", "letter")
all_res = []

test_cases = {
    "base": (None, 0.),
    "mid": (rec_weights, 0.01),
}

noise_mags = (0.0, 0.05, 0.1,)
ablate_nums = (5, 10, 20, 50, 100)


for nm, (rec_weights, rec_syn) in test_cases.items():

    print(nm)

    if rec_weights is None:
        save_weights = False
    else:
        save_weights = True

    for repeats in range(10):
        for l_i in range(n_items):
            letter = string.ascii_uppercase[l_i]
            test_vec = [vocab[letter].v * 0.8]

            for noise_magnitude in noise_mags:
                for ablate_num in ablate_nums:
                    ablate_index = rng.choice(np.arange(n_neurons),
                                              replace=False, size=ablate_num)
                    res = test_mem(encoders, decoders, test_vec,
                                   noise_magnitude, ablate_index,
                                   rec_w=rec_weights, rec_synapse=rec_syn)
                    conf = conf_metric(spa.similarity(res, vocab), l_i)

                    all_res.append((conf["correct"],
                                    conf["top_mag"],
                                    conf["runnerup_dist"],
                                    noise_magnitude,
                                    ablate_num,
                                    save_weights,
                                    rec_syn,
                                    letter))

all_df = pd.DataFrame(all_res, columns=df_cols)
all_df.to_hdf(
    os.path.join(data_path, "neg_voja_rec_test", "more_ablate.h5"),
    "conf", mode="w")
