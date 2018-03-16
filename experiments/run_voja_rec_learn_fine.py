"""Copying a bunch of code for running voja_rec_learn under different conditions"""

import numpy as np
import h5py

import nengo
import nengo_spa as spa

from nengo_learn_assoc_mem.utils import BasicVecFeed, numpy_bytes_to_str, norm_spa_vecs
from nengo_learn_assoc_mem.learning_rules import rec_bcm

from collections import namedtuple


with h5py.File("../data/neg_voja_enc.h5", "r") as fi:
    print(list(fi.keys()))

    fan1 = numpy_bytes_to_str(fi['fan1'])
    fan2 = numpy_bytes_to_str(fi['fan2'])
    foil1 = numpy_bytes_to_str(fi['foil1'])
    foil2 = numpy_bytes_to_str(fi['foil2'])

    v_strs = numpy_bytes_to_str(fi['vocab_strings'])
    v_vecs = list(fi['vocab_vectors'])
    dimensions = fi['vocab_vectors'].attrs['dimensions']

    fin_enc = np.array(fi['encoders'])
    n_neurons = fin_enc.shape[0]
    intercepts = [fi['encoders'].attrs['intercept']] * n_neurons
    seed = fi['encoders'].attrs['seed']

    dt = fi['t_range'].attrs['dt']

vocab = spa.Vocabulary(dimensions)
for val, vec in zip(v_strs, v_vecs):
    vocab.add(val, vec)

fan1_pair_vecs = norm_spa_vecs(vocab, fan1)
fan2_pair_vecs = norm_spa_vecs(vocab, fan2)
foil1_pair_vecs = norm_spa_vecs(vocab, foil1)
foil2_pair_vecs = norm_spa_vecs(vocab, foil2)

save_file_name = "more_repeats"

t_pause = 0.1
t_present = 0.3

t_each = t_pause + t_present
td_each = int(t_each/dt)
td_pause = int(t_pause/dt)

fan1_slc = slice(td_pause, td_each*len(fan1_pair_vecs)+td_pause)
fan2_slc = slice(fan1_slc.stop, fan1_slc.stop+td_each*len(fan2_pair_vecs))
foil1_slc = slice(fan2_slc.stop, fan2_slc.stop+td_each*len(foil1_pair_vecs))
foil2_slc = slice(foil1_slc.stop, foil1_slc.stop+td_each*len(foil2_pair_vecs))

SlcStim = namedtuple("Stim", ['fan_num', 'targ', 'vecs', 'slc'])

slc_stim_vecs = {"fan1": SlcStim(1, True, np.array(fan1_pair_vecs), fan1_slc),
                 "fan2": SlcStim(2, True, np.array(fan2_pair_vecs), fan2_slc),
                 "foil1": SlcStim(1, False, np.array(foil1_pair_vecs), foil1_slc),
                 "foil2": SlcStim(2, False, np.array(foil2_pair_vecs), foil2_slc)}

feed_vecs = fan1_pair_vecs + fan2_pair_vecs

n_repeats = 6
sample_freq = 10

base_inhib = -2e-3
bcm_thresh = 10
bcm_lr = 1e-6

rec_inhib = base_inhib*(np.ones(n_neurons) - np.eye(n_neurons))

feed = BasicVecFeed(feed_vecs, feed_vecs, t_present, dimensions, len(feed_vecs), t_pause)

with nengo.Network() as model:
    in_nd = nengo.Node(feed.feed)
    paused = nengo.Node(lambda t: 1-feed.paused)

    ens = nengo.Ensemble(n_neurons, dimensions,
                         encoders=fin_enc.copy(), intercepts=intercepts, seed=seed)
    rec_learn = rec_bcm.RecBCM(n_neurons, rec_inhib, learning_rate=bcm_lr, threshold=bcm_thresh,
                               max_inhib=-0.05, sample_every=t_each/sample_freq)

    nengo.Connection(in_nd, ens, synapse=None)
    nengo.Connection(ens.neurons, rec_learn.in_neurons, synapse=0.01)
    nengo.Connection(ens.neurons, rec_learn.out_neurons, synapse=0.01)
    nengo.Connection(paused, rec_learn.enable, synapse=None)
    nengo.Connection(rec_learn.output, ens.neurons, synapse=None)

    p_in = nengo.Probe(in_nd)
    p_spikes = nengo.Probe(ens.neurons)

with nengo.Simulator(model) as sim:
    sim.run(n_repeats*len(feed_vecs)*t_each + t_pause)

w_hist = np.array(rec_learn.weight_history)

with h5py.File(f"../data/neg_voja_rec_learn/{save_file_name}.h5", "w") as sv_fi:
    sv_fi.create_dataset("train_input", data=np.array(sim.data[p_in]))

    sv_fi.create_dataset("train_spikes", data=np.array(sim.data[p_spikes]))

    bcm = sv_fi.create_dataset("w_hist", data=w_hist)

    bcm.attrs["repeats"] = n_repeats
    bcm.attrs["sample_freq"] = sample_freq

    bcm.attrs["t_pause"] = t_pause
    bcm.attrs["t_present"] = t_present

    bcm.attrs["base_inhib"] = base_inhib
    bcm.attrs["thresh"] = bcm_thresh
    bcm.attrs["learning_rate"] = bcm_lr

rec_w = w_hist[-1]

feed = BasicVecFeed(feed_vecs, feed_vecs, t_present, dimensions, len(feed_vecs), t_pause)

with nengo.Network() as learned_model:
    in_nd = nengo.Node(feed.feed)

    ens = nengo.Ensemble(n_neurons, dimensions,
                         encoders=fin_enc.copy(), intercepts=intercepts, seed=seed)

    nengo.Connection(in_nd, ens)
    nengo.Connection(ens.neurons, ens.neurons, transform=rec_w, synapse=0.01)

    p_in = nengo.Probe(in_nd)
    p_spikes = nengo.Probe(ens.neurons)

with nengo.Simulator(learned_model) as learned_sim:
    learned_sim.run(len(feed_vecs) * t_each + t_pause)

with h5py.File(f"../data/neg_voja_rec_learn/{save_file_name}.h5", "a") as sv_fi:
    sv_fi.create_dataset("spike_response", data=np.array(learned_sim.data[p_spikes]))
