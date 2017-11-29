"""make a plot showing how the response changes as a function of the voja rate and number of neurons"""

import numpy as np
import pandas as pd

import nengo

from nengo_learn_assoc_mem.utils import BasicVecFeed, meg_from_spikes, make_fan_vocab
from nengo_learn_assoc_mem.learn_assoc import LearningAssocMem


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


def train_net(neuron_num: int, vr: float, net_seed):
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
        sim.run(len(all_labels) * (t_present + t_pause) * 20 + t_pause)

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

    foil_resp = []

    for foil in ("FROG", "TOAD", "NEWT", "ROACH"):
        f_vec = vocab.parse(foil).v
        stim.output = lambda t: f_vec
        correct.output = lambda t: f_vec

        with nengo.Simulator(model) as sim:
            sim.run(t_present)

        foil_resp.append(meg_from_spikes(
            np.concatenate(
                (np.zeros((100, neuron_num)), sim.data[p_spikes]), axis=0))
        )

    foil_resp = np.array(foil_resp).T

    return fan1_resp, fan2_resp, foil_resp


df_cols = ("t_idx", "value", "metric", "n_neurons", "learning_rate", "seed")
resp_df = pd.DataFrame(columns=df_cols)
mean_df = pd.DataFrame(columns=df_cols)

# Show effect of different number of neurons
n_neurons = (10, 30, 50)

resp_res = []
mean_res = []

for sd in range(10):
    for nrn in n_neurons:
        res = run_feed(nrn, 1e-4, sd)

        t_len = res[0].shape[0]
        resp_res.append((
            np.arange(t_len),
            res[0],
            ["fan1",]*t_len,
            np.ones(t_len)*nrn,
            np.ones(t_len)*1e-4,
            np.ones(t_len)*sd))


# Show effect of different learning rate
voja_rates = (1e-3, 1e-4, 1e-5)

for sd in range(10):
    for voja_rate in voja_rates:
        res = run_feed(10, voja_rate, sd)

        vr_fan1_resp[vr].append(res[0])
        vr_fan2_resp[vr].append(res[1])
        vr_foil_resp[vr].append(res[2])

        vr_fan1_mean[vr].append(np.mean(res[0], axis=0))
        vr_fan2_mean[vr].append(np.mean(res[1], axis=0))
        vr_foil_mean[vr].append(np.mean(res[2], axis=0))
