"""Show variability caused by seed"""
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import h5py

import nengo
import nengolib

from nengo_learn_assoc_mem.utils import make_alt_vocab, BasicVecFeed, gen_added_strings, list_as_ascii
from nengo_learn_assoc_mem.paths import data_path
from nengo_learn_assoc_mem.learning_rules.neg_voja import NegVoja


def get_encoders(cepts: np.ndarray, seed: float) -> np.ndarray:
    with nengolib.Network(seed=seed) as model:
        ens = nengo.Ensemble(n_neurons, dimensions, intercepts=cepts, seed=seed)

    with nengo.Simulator(model) as sim:
        pass

    return sim.data[ens].encoders.copy()


def train_encoders(feed_vecs: np.ndarray, enc: np.ndarray, cepts: np.ndarray,
                   nvoja_lr: float, seed: float) -> np.ndarray:
    feed = BasicVecFeed(feed_vecs, feed_vecs,
                        t_present, dimensions, len(feed_vecs), t_pause)

    with nengolib.Network(seed=seed) as model:
        in_nd = nengo.Node(feed.feed)
        paused = nengo.Node(lambda t: 1 - feed.paused)

        neg_voja = NegVoja(enc.copy(), learning_rate=nvoja_lr)
        ens = nengo.Ensemble(n_neurons, dimensions, intercepts=cepts, seed=seed)

        nengo.Connection(in_nd, neg_voja.input_signal, synapse=None)
        nengo.Connection(ens.neurons, neg_voja.input_activities, synapse=0)
        nengo.Connection(paused, neg_voja.enable, synapse=None)
        nengo.Connection(neg_voja.output, ens.neurons, synapse=None)

    t_sim = len(feed_vecs) * t_each + t_pause
    print("Simulating for", t_sim)
    with nengo.Simulator(model) as sim:
        sim.run(t_sim)

    return neg_voja.encoder_hist


def test_response(feed_vecs, vo, encs: np.ndarray, save_file: str, enc_args: Dict, seed: float, plt_res=False):

    feed = BasicVecFeed(feed_vecs, feed_vecs,
                        t_present, dimensions, len(feed_vecs), t_pause)

    with nengolib.Network(seed=seed) as test_model:
        in_nd = nengo.Node(feed.feed)
        pause = nengo.Node(lambda t: feed.paused)
        meg_nd = nengo.Node(lambda t, x: np.sum(x),
                            size_in=n_neurons, size_out=1)

        ens = nengo.Ensemble(n_neurons, dimensions,
                             encoders=encs.copy(),
                             intercepts=intercepts,
                             seed=seed)

        nengo.Connection(in_nd, ens, synapse=None)
        nengo.Connection(pause, ens.neurons, transform=-10 * np.ones((n_neurons, 1)), synapse=None)

        nengo.Connection(ens.neurons, meg_nd, synapse=None)

        p_meg = nengo.Probe(meg_nd, synapse=0.01)

    t_sim = len(feed_vecs) * t_each + t_pause
    with nengo.Simulator(test_model) as test_sim:
        test_sim.run(t_sim)

    test_meg = test_sim.data[p_meg].squeeze()

    fan1_resp = np.mean(test_meg[fan1_slc].reshape((-1, td_each)), axis=0)
    fan2_resp = np.mean(test_meg[fan2_slc].reshape((-1, td_each)), axis=0)
    foil1_resp = np.mean(test_meg[foil1_slc].reshape((-1, td_each)), axis=0)
    foil2_resp = np.mean(test_meg[foil2_slc].reshape((-1, td_each)), axis=0)

    if plt_res:
        plt.figure()
        plt.plot(fan1_resp)
        plt.plot(fan2_resp)
        plt.plot(foil1_resp)
        plt.plot(foil2_resp)

        plt.legend(["fan1 targ", "fan2 targ", "foil1 targ", "foil2 targ"], facecolor=None)
        plt.show()

    # save the resulting activities, weights and vocab
    save_path = os.path.join(data_path, "", save_file)
    with h5py.File(save_path, "w") as fi:
        fi.create_dataset("resp/fan1", data=fan1_resp)
        fi.create_dataset("resp/fan2", data=fan2_resp)
        fi.create_dataset("resp/foil1", data=foil1_resp)
        fi.create_dataset("resp/foil2", data=foil2_resp)

        tm = fi.create_dataset("t_range", data=[0, t_sim])
        tm.attrs["dt"] = float(test_sim.dt)
        tm.attrs["t_pause"] = t_pause
        tm.attrs["t_present"] = t_present

        enc = fi.create_dataset("encoders", data=encs)
        enc.attrs["seed"] = seed
        enc.attrs["intercept"] = enc_args["intercept"]
        enc.attrs["learning_rate"] = enc_args["learning_rate"]
        enc.attrs["bias"] = enc_args["bias"]

        pnt_nms = []
        pnt_vectors = []
        for nm, pnt in vo.items():
            pnt_nms.append(nm)
            pnt_vectors.append(pnt.v)

        fi.create_dataset("vocab_strings", data=list_as_ascii(pnt_nms))
        vec = fi.create_dataset("vocab_vectors", data=pnt_vectors)
        vec.attrs["dimensions"] = dimensions

        for nm, pairs in zip(("fan1", "fan2", "foil1", "foil2"), (fan1, fan2, foil1, foil2)):
            added_str = gen_added_strings(pairs)
            fi.create_dataset(nm, data=list_as_ascii(added_str))


n_items = 16
dimensions = 32
n_neurons = 2000

dt = 0.001

t_pause = 0.1
t_present = 0.3
t_each = t_pause + t_present

td_each = int(t_each / dt)
td_pause = int(t_pause / dt)

fan1_slc = slice(td_pause, td_each*n_items+td_pause)
fan2_slc = slice(fan1_slc.stop, fan1_slc.stop+td_each*n_items)
foil1_slc = slice(fan2_slc.stop, fan2_slc.stop+td_each*n_items)
foil2_slc = slice(foil1_slc.stop, foil1_slc.stop+td_each*n_items)

init_seed = 8
intercept = 0.15
intercepts = nengo.dists.Uniform(0., intercept).sample(n_neurons)
neg_voja_lr = 5e-6
past_encs = np.zeros((n_neurons, dimensions))
start_encs = get_encoders(intercepts, init_seed)

for seed_val in range(10):
    vocab, fan1, fan1_pair_vecs, fan2, fan2_pair_vecs, \
        foil1, foil1_pair_vecs, foil2, foil2_pair_vecs = make_alt_vocab(n_items, n_items, dimensions,
                                                                        seed_val, norm=True)

    learned_encs = train_encoders(fan1_pair_vecs + fan2_pair_vecs,
                                  start_encs, intercepts,
                                  neg_voja_lr, init_seed)

    assert not np.allclose(start_encs, learned_encs)
    assert not np.allclose(learned_encs, past_encs)
    past_encs = learned_encs.copy()

    learning_args = {"intercept": intercepts,
                     "learning_rate": neg_voja_lr}
    test_response(fan1_pair_vecs + fan2_pair_vecs + foil1_pair_vecs + foil2_pair_vecs,
                  vocab,
                  learned_encs[-1].copy(),
                  f"neg_voja_{seed_val}.h5", learning_args, init_seed, False)
