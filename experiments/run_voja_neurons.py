"""make a plot showing how the response changes as a function of the voja rate and number of neurons"""

import numpy as np
import h5py

import nengo

from nengo_learn_assoc_mem.utils import BasicVecFeed, meg_from_spikes, make_fan_vocab, spa_parse_norm
from nengo_learn_assoc_mem.learn_assoc import LearningAssocMem

from typing import List


t_present = 0.3
t_pause = 0.1
dimensions = 32
dt = 0.001
seed = 8

t_pres = int(t_present / dt)
t_paus = int(t_pause / dt)
slc_tm = (t_paus+t_pres)

vocab, fan1, fan1_vecs, fan1_labels, fan2, fan2_vecs, fan2_labels = make_fan_vocab(seed, dimensions)

all_vecs = np.concatenate((fan1_vecs, fan2_vecs))
simi = np.dot(all_vecs, all_vecs.T)
np.fill_diagonal(simi, 0.)
intercept = np.ceil(np.max(simi)*100)/100
print(intercept)

all_labels = fan1 + fan2


def is_ans(t, x):
    if np.abs(np.sum(x)) > 0:
        return 1
    else:
        return -1


def get_foil_resp(stim, correct, model, foil_stim: List[str], neuron_num: int, spikes: np.ndarray):
    foil_resp = []

    for foil in foil_stim:
        f_vec = spa_parse_norm(vocab, foil)
        stim.output = lambda t: f_vec
        correct.output = lambda t: f_vec

        with nengo.Simulator(model) as sim:
            sim.run(t_present)

        foil_resp.append(meg_from_spikes(
            np.concatenate(
                (np.zeros((100, neuron_num)), spikes), axis=0))[100:]
                          )

    return np.array(foil_resp).T


def train_net(neuron_num: int, vr: float, net_seed):
    pres_num = 20

    data_feed = BasicVecFeed(all_vecs, all_vecs, t_present, dimensions, len(all_vecs), t_pause)

    with nengo.Network(seed=net_seed) as train_model:
        stim = nengo.Node(data_feed.feed)
        correct = nengo.Node(data_feed.get_answer)
        learn_control = nengo.Node(is_ans, size_in=dimensions)

        # Note: setting the intercept doesn't really change much, because there's so much similarity already
        mem = LearningAssocMem(neuron_num, dimensions, voja_rate=vr, pes_rate=1e-3,
                               intercepts=np.ones(neuron_num) * 0.1,
                               seed=net_seed)

        nengo.Connection(stim, mem.input, synapse=None)
        nengo.Connection(correct, mem.correct, synapse=None)

        nengo.Connection(correct, learn_control, synapse=None)
        nengo.Connection(learn_control, mem.stop_learn)

        train_model.p_enc = nengo.Probe(mem.mem, 'scaled_encoders', sample_every=0.1)
        train_model.p_dec = nengo.Probe(mem.conn_out, 'weights', sample_every=0.1)

    with nengo.Simulator(train_model) as sim:
        sim.run(len(all_labels) * (t_present + t_pause) * pres_num + t_pause)

    load_from = dict()
    load_from["enc"] = sim.data[train_model.p_enc][-1].copy()
    load_from["dec"] = sim.data[train_model.p_dec][-1].copy()
    load_from["seed"] = seed

    return load_from


def run_feed(neuron_num: int, vr: float, net_seed):

    load_from = train_net(neuron_num, vr, net_seed)

    data_feed = BasicVecFeed(all_vecs, all_vecs, t_present, dimensions, len(all_vecs), t_pause)

    with nengo.Network(seed=net_seed) as model:
        stim = nengo.Node(data_feed.feed)
        correct = nengo.Node(data_feed.get_answer)

        mem = LearningAssocMem(neuron_num, dimensions, voja_rate=0, pes_rate=0, load_from=load_from,
                               intercepts=np.ones(neuron_num) * 0.1,
                               seed=net_seed)

        nengo.Connection(stim, mem.input, synapse=None)

        p_spikes = nengo.Probe(mem.mem.neurons, synapse=None)

    with nengo.Simulator(model) as sim:
        sim.run(len(all_labels) * (t_present + t_pause) + t_pause)

    ens_responses = meg_from_spikes(sim.data[p_spikes])

    fan1_resp = []

    for t_p in range(t_paus, len(fan1)*slc_tm, slc_tm):
        fan1_resp.append(ens_responses[t_p:t_p+slc_tm])

    fan1_resp = np.array(fan1_resp).T

    fan2_resp = []

    for t_p in range(t_paus+len(fan1)*slc_tm, (len(fan1)+len(fan2))*slc_tm, slc_tm):
        fan2_resp.append(ens_responses[t_p:t_p+slc_tm])

    fan2_resp = np.array(fan2_resp).T

    foil1_stim = ["FROG+CAT", "TOAD+DOG", "NEWT+DUCK", "ROACH+FISH"]
    foil1_resp = get_foil_resp(stim, correct, model, foil1_stim, neuron_num, sim.data[p_spikes])

    foil2_stim = ["FROG+TOAD", "TOAD+NEWT", "NEWT+ROACH", "ROACH+FROG"]
    foil2_resp = get_foil_resp(stim, correct, model, foil2_stim, neuron_num, sim.data[p_spikes])

    return {"fan1": fan1_resp, "fan2": fan2_resp, "foil1": foil1_resp, "foil2": foil2_resp}


# Show effect of different learning rate
voja_rates = ("1e-4", "5e-4", "1e-5")
vr_dict = {}
for voja_r in voja_rates:
    vr_dict[voja_r] = float(voja_r)
nrn = 50

with h5py.File("data/fixed_foil.h5", "w") as fi:
    for desc, voja_rate in vr_dict.items():
        v_grp = fi.create_group(desc)

        for sd in range(10):
            sd_grp = v_grp.create_group(str(sd))
            sd_grp.attrs["n_neurons"] = nrn
            sd_grp.attrs["learning_rate"] = voja_rate
            sd_grp.attrs["seed"] = sd
            res = run_feed(nrn, voja_rate, sd)

            for key, val in res.items():
                sd_grp.create_dataset(key, data=val)
