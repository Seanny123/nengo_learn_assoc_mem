"""Show that encoders"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py

import nengo

from nengo_learn_assoc_mem.utils import make_alt_vocab, BasicVecFeed, gen_added_strings, list_as_ascii
from nengo_learn_assoc_mem.learning_rules.neg_voja import NegVoja

from collections import namedtuple

n_items = 16
dimensions = 32
n_neurons = 500

dt = 0.001
seed = 8

vocab, fan1, fan1_pair_vecs, fan2, fan2_pair_vecs,\
    foil1, foil1_pair_vecs, foil2, foil2_pair_vecs = make_alt_vocab(n_items, n_items, dimensions, seed, norm=True)

feed_vecs = fan1_pair_vecs + fan2_pair_vecs

n_repeats = 5
t_pause = 0.1
t_present = 0.3
t_each = t_pause + t_present

td_pause = t_pause/dt

intercept = 0.2
intercepts = [intercept]*n_neurons

neg_voja_lr = 5e-6 / n_repeats

# generate encoders to be used in Negative Voja

with nengo.Network() as model:
    ens = nengo.Ensemble(n_neurons, dimensions, intercepts=intercepts, seed=seed)

with nengo.Simulator(model) as sim:
    pass

enc = sim.data[ens].encoders.copy()

# learn encoders

feed = BasicVecFeed(feed_vecs, feed_vecs, t_present, dimensions, len(feed_vecs), t_pause)

with nengo.Network() as model:
    in_nd = nengo.Node(feed.feed)
    paused = nengo.Node(lambda t: 1-feed.paused)

    neg_voja = NegVoja(enc.copy(), learning_rate=neg_voja_lr)
    ens = nengo.Ensemble(n_neurons, dimensions, intercepts=intercepts, seed=seed)

    nengo.Connection(in_nd, neg_voja.input_signal, synapse=None)
    nengo.Connection(ens.neurons, neg_voja.input_activities, synapse=0)
    nengo.Connection(paused, neg_voja.enable, synapse=None)
    nengo.Connection(neg_voja.output, ens.neurons, synapse=None)

    p_in = nengo.Probe(in_nd)
    p_spikes = nengo.Probe(ens.neurons)

t_sim = n_repeats*len(feed_vecs)*t_each + t_pause
with nengo.Simulator(model) as sim:
    sim.run(t_sim)

fin_enc = neg_voja.encoder_hist[-1].copy()

# verify result with plots

with nengo.Network() as model:
    ens = nengo.Ensemble(n_neurons, dimensions, encoders=fin_enc, intercepts=intercepts, seed=seed)
sim = nengo.Simulator(model)

Stim = namedtuple("Stim", ['fan_num', 'targ', 'vecs'])

stim_vecs = {"fan1": Stim(1, True, np.array(fan1_pair_vecs)),
             "fan2": Stim(2, True, np.array(fan2_pair_vecs)),
             "foil1": Stim(1, False, np.array(foil1_pair_vecs)),
             "foil2": Stim(2, False, np.array(foil2_pair_vecs))}

df_list = []

for key, obj in stim_vecs.items():
    _, act = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=obj.vecs)

    flat_act = list(act.flatten())
    resp = [obj.fan_num] * len(flat_act)
    targ = [obj.targ] * len(flat_act)
    df_list.append(pd.DataFrame({"act": flat_act, "resp": resp, "targ": targ}))

act_df = pd.concat(df_list)

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

# save the resulting weights and vocab

with h5py.File("data/neg_voja_enc.h5", "w") as fi:
    tm = fi.create_dataset("t_range", data=[0, t_sim])
    tm.attrs["dt"] = float(sim.dt)
    tm.attrs["t_pause"] = t_pause
    tm.attrs["t_present"] = t_present

    enc = fi.create_dataset("encoders", data=fin_enc)
    enc.attrs["seed"] = seed
    enc.attrs["intercept"] = intercept
    enc.attrs["learning_rate"] = neg_voja_lr

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
