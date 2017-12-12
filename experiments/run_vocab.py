"""make a plot showing how the response changes as a function of the voja rate and number of neurons"""

import numpy as np
import h5py

import nengo
from nengo import spa

from nengo_learn_assoc_mem.utils import BasicVecFeed, meg_from_spikes, make_fan_vocab, norm_spa_vecs
from nengo_learn_assoc_mem.learn_assoc import LearningAssocMem

from typing import NamedTuple, List


class ModelVocab(NamedTuple):
    vocab: spa.Vocabulary
    all_vecs: np.ndarray
    fan1: List[str]
    fan2: List[str]
    foil: List[str]


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

basic_foil_vocab = ["FROG", "TOAD", "NEWT", "ROACH"]

m_v = ModelVocab(vocab, all_vecs.copy(), fan1.copy(), fan2.copy(), basic_foil_vocab.copy())


def is_ans(t, x):
    if np.abs(np.sum(x)) > 0:
        return 1
    else:
        return -1


def train_net(neuron_num: int, vr: float, model_vocab: ModelVocab, net_seed):
    pres_num = 20
    n_items = len(model_vocab.all_vecs)

    data_feed = BasicVecFeed(model_vocab.all_vecs, model_vocab.all_vecs, t_present, dimensions, n_items, t_pause)

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
        sim.run(n_items * (t_present + t_pause) * pres_num + t_pause)

    load_from = dict()
    load_from["enc"] = sim.data[train_model.p_enc][-1].copy()
    load_from["dec"] = sim.data[train_model.p_dec][-1].copy()
    load_from["seed"] = seed

    return load_from


def run_feed(neuron_num: int, vr: float, model_vocab: ModelVocab, net_seed):
    load_from = train_net(neuron_num, vr, model_vocab, net_seed)
    n_items = len(model_vocab.all_vecs)

    data_feed = BasicVecFeed(model_vocab.all_vecs, model_vocab.all_vecs, t_present, dimensions, n_items, t_pause)

    with nengo.Network(seed=net_seed) as model:
        stim = nengo.Node(data_feed.feed)
        correct = nengo.Node(data_feed.get_answer)

        mem = LearningAssocMem(neuron_num, dimensions, voja_rate=0, pes_rate=0, load_from=load_from,
                               intercepts=np.ones(neuron_num) * 0.1,
                               seed=net_seed)

        nengo.Connection(stim, mem.input, synapse=None)

        p_spikes = nengo.Probe(mem.mem.neurons, synapse=None)

    with nengo.Simulator(model) as sim:
        sim.run(n_items * (t_present + t_pause) + t_pause)

    ens_responses = meg_from_spikes(sim.data[p_spikes])

    fan1_resp = []

    for t_p in range(t_paus, len(model_vocab.fan1)*slc_tm, slc_tm):
        fan1_resp.append(ens_responses[t_p:t_p+slc_tm])

    fan1_resp = np.array(fan1_resp).T

    fan2_resp = []

    for t_p in range(t_paus+len(model_vocab.fan1)*slc_tm, n_items*slc_tm, slc_tm):
        fan2_resp.append(ens_responses[t_p:t_p+slc_tm])

    fan2_resp = np.array(fan2_resp).T

    foil_resp = []

    for foil in model_vocab.foil:
        f_vec = model_vocab.vocab.parse(foil).v
        stim.output = lambda t: f_vec
        correct.output = lambda t: f_vec

        with nengo.Simulator(model) as sim:
            sim.run(t_present)

        foil_resp.append(meg_from_spikes(
            np.concatenate(
                (np.zeros((100, neuron_num)), sim.data[p_spikes]), axis=0))[100:]
        )

    foil_resp = np.array(foil_resp).T

    return {"fan1": fan1_resp, "fan2": fan2_resp, "foil": foil_resp}


# Show effect of different learning rate
nrn = 50
voja_rate = 1e-5

with h5py.File("data/vocab_exp.h5", "w") as fi:
    v_grp = fi.create_group("more_foil")
    m_v.foil += ["BUG", "FLY"]

    for sd in range(10):
        sd_grp = v_grp.create_group(str(sd))
        sd_grp.attrs["train_rep"] = 20
        sd_grp.attrs["n_neurons"] = nrn
        sd_grp.attrs["learning_rate"] = voja_rate
        sd_grp.attrs["seed"] = sd
        res = run_feed(nrn, voja_rate, m_v, sd)

        for key, val in res.items():
            sd_grp.create_dataset(key, data=val)

    v_grp = fi.create_group("more_fan1")

    m_v.foil = basic_foil_vocab.copy()

    new_fan1 = ["CROW+TURKEY", "DOVE+PIGEON"]
    m_v.fan1 += new_fan1
    new_fan1_vecs = norm_spa_vecs(vocab, new_fan1)
    m_v.all_vecs = np.concatenate((m_v.all_vecs, new_fan1_vecs))

    for sd in range(10):
        sd_grp = v_grp.create_group(str(sd))
        sd_grp.attrs["train_rep"] = 20
        sd_grp.attrs["n_neurons"] = nrn
        sd_grp.attrs["learning_rate"] = voja_rate
        sd_grp.attrs["seed"] = sd
        res = run_feed(nrn, voja_rate, m_v, sd)

        for key, val in res.items():
            sd_grp.create_dataset(key, data=val)

    v_grp = fi.create_group("more_fan2")

    m_v.fan1 = fan1.copy()

    new_fan2 = ["MOLE+MOUSE", "MOLE+KOALA"]
    m_v.fan2 += ["MOLE+MOUSE", "MOLE+KOALA"]
    new_fan2_vecs = norm_spa_vecs(vocab, new_fan2)
    m_v.all_vecs = np.concatenate((all_vecs.copy(), new_fan2_vecs))

    for sd in range(10):
        sd_grp = v_grp.create_group(str(sd))
        sd_grp.attrs["train_rep"] = 20
        sd_grp.attrs["n_neurons"] = nrn
        sd_grp.attrs["learning_rate"] = voja_rate
        sd_grp.attrs["seed"] = sd
        res = run_feed(nrn, voja_rate, m_v, sd)

        for key, val in res.items():
            sd_grp.create_dataset(key, data=val)
