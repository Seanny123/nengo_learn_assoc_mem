"""Show that encoders"""
from collections import namedtuple
import itertools
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py

import nengo
import nengolib

from nengo_learn_assoc_mem.utils import make_alt_vocab, BasicVecFeed, gen_added_strings, list_as_ascii
from nengo_learn_assoc_mem.paths import data_path
from nengo_learn_assoc_mem.learning_rules.neg_voja import NegVoja


n_items = 16
dimensions = 32
n_neurons = 500

dt = 0.001
vocab_seed = 8

vocab, fan1, fan1_pair_vecs, fan2, fan2_pair_vecs,\
    foil1, foil1_pair_vecs, foil2, foil2_pair_vecs = make_alt_vocab(n_items, n_items, dimensions, vocab_seed, norm=True)

Stim = namedtuple("Stim", ['fan_num', 'targ', 'vecs'])

stim_vecs = {"fan1": Stim(1, True, np.array(fan1_pair_vecs)),
             "fan2": Stim(2, True, np.array(fan2_pair_vecs)),
             "foil1": Stim(1, False, np.array(foil1_pair_vecs)),
             "foil2": Stim(2, False, np.array(foil2_pair_vecs))}

feed_vecs = fan1_pair_vecs + fan2_pair_vecs

t_pause = 0.1
t_present = 0.3
t_each = t_pause + t_present

td_pause = t_pause / dt


def get_encoders(cepts: np.ndarray, seed: float) -> np.ndarray:
    with nengolib.Network(seed=seed) as model:
        ens = nengo.Ensemble(n_neurons, dimensions, intercepts=cepts, seed=seed)

    with nengo.Simulator(model) as sim:
        pass

    return sim.data[ens].encoders.copy()


def train_encoders(enc: np.ndarray, cepts: np.ndarray, n_voja_lr: float, sim_len: float, seed: float) -> np.ndarray:
    feed = BasicVecFeed(feed_vecs, feed_vecs, t_present, dimensions, len(feed_vecs), t_pause)

    with nengolib.Network(seed=seed) as model:
        in_nd = nengo.Node(feed.feed)
        paused = nengo.Node(lambda t: 1-feed.paused)

        neg_voja = NegVoja(enc.copy(), learning_rate=n_voja_lr)
        ens = nengo.Ensemble(n_neurons, dimensions, intercepts=cepts, seed=seed)

        nengo.Connection(in_nd, neg_voja.input_signal, synapse=None)
        nengo.Connection(ens.neurons, neg_voja.input_activities, synapse=0)
        nengo.Connection(paused, neg_voja.enable, synapse=None)
        nengo.Connection(neg_voja.output, ens.neurons, synapse=None)

    print("Simulating for", sim_len)
    with nengo.Simulator(model) as sim:
        sim.run(sim_len)

    return neg_voja.encoder_hist


def test_response(encs: np.ndarray, save_file: str, enc_args: Dict, seed: float, plt_res=False):
    fin_enc = encs[-1].copy()
    # verify result with plots
    with nengo.Network() as model:
        ens = nengo.Ensemble(n_neurons, dimensions, encoders=fin_enc, intercepts=intercepts, seed=seed)
    sim = nengo.Simulator(model)

    df_list = []

    for key, obj in stim_vecs.items():
        _, act = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=obj.vecs)

        flat_act = list(act.flatten())
        resp = [obj.fan_num] * len(flat_act)
        targ = [obj.targ] * len(flat_act)
        df_list.append(pd.DataFrame({"act": flat_act, "resp": resp, "targ": targ}))

    act_df = pd.concat(df_list)

    if plt_res:
        plt.figure()
        ax = sns.barplot(x="resp", y="act", data=act_df[act_df.targ == True])

        ax.set_title("Difference between FAN1 and FAN2 firing rates")
        ax.set_ylabel("Mean Firing Rate")
        ax.set_xlabel("FAN type")
        plt.show()

        plt.figure()
        ax = sns.barplot(x="resp", y="act", hue="targ", hue_order=(True, False), data=act_df)

        ax.set_title("Difference between FAN1 and FAN2 firing rates")
        ax.set_ylabel("Mean Firing Rate")
        ax.set_xlabel("FAN type")

        ax.legend_.set_title("Target vs. Foil")
        new_labels = ("Target", "Foil")
        for t, l in zip(ax.legend_.texts, new_labels):
            t.set_text(l)
        plt.show()

    # save the resulting activities, weights and vocab
    save_path = os.path.join(data_path, save_file)
    with h5py.File(save_path, "w") as fi:
        tm = fi.create_dataset("t_range", data=[0, enc_args["t_sim"]])
        tm.attrs["dt"] = float(sim.dt)
        tm.attrs["t_pause"] = t_pause
        tm.attrs["t_present"] = t_present

        enc = fi.create_dataset("encoders", data=encs)
        enc.attrs["seed"] = seed
        enc.attrs["intercept"] = enc_args["intercept"]
        enc.attrs["learning_rate"] = enc_args["learning_rate"]

        pnt_nms = []
        pnt_vectors = []
        for nm, pnt in vocab.items():
            pnt_nms.append(nm)
            pnt_vectors.append(pnt.v)

        fi.create_dataset("vocab_strings", data=list_as_ascii(pnt_nms))
        vec = fi.create_dataset("vocab_vectors", data=pnt_vectors)
        vec.attrs["dimensions"] = dimensions

        for nm, pairs in zip(("fan1", "fan2", "foil1", "foil2"), (fan1, fan2, foil1, foil2)):
            added_str = gen_added_strings(pairs)
            fi.create_dataset(nm, data=list_as_ascii(added_str))

    act_df.to_hdf(save_path, "response", mode="r+", format="fixed")


intercept = 0.15
lr = 8e-6
n_repeats = 4
past_encs = np.zeros((n_neurons, dimensions))

for seed_val in range(10):
    intercepts = np.ones(n_neurons) * intercept
    neg_voja_lr = lr / n_repeats
    t_sim = n_repeats * len(feed_vecs) * t_each + t_pause

    start_encs = get_encoders(intercepts, seed_val)
    learned_encs = train_encoders(start_encs, intercepts, neg_voja_lr, t_sim, seed_val)
    
    assert not np.allclose(start_encs, learned_encs)
    assert not np.allclose(learned_encs, past_encs)
    past_encs = learned_encs.copy()

    learning_args = {"intercept": intercept, "learning_rate": neg_voja_lr, "t_sim": t_sim}
    test_response(learned_encs, f"neg_voja_enc_{n_repeats}_{intercept}_{lr}_{seed_val}.h5", learning_args, seed_val)
