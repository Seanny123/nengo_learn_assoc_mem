"""Model from the vision output until the recognition"""
import numpy as np
import matplotlib.pyplot as plt

import nengo
import nengo_spa as spa

from nengo_learn_assoc_mem.utils import make_alt_vocab, VecToScalarFeed


def choose_encoders(n_neurons: int, dimensions: int, encoder_proportion: float, mean_fan1, mean_fan2):
    encoders = np.zeros((n_neurons, dimensions))

    fan1_end = int(n_neurons * encoder_proportion)

    for n_i in range(fan1_end):
        encoders[n_i] = mean_fan1 + np.random.normal(size=dimensions) * 0.1

    for n_i in range(fan1_end, n_neurons):
        encoders[n_i] = mean_fan2 + np.random.normal(size=dimensions) * 0.1

    return encoders


D = 32
de_n_neurons = 1000
ea_n_neurons = 100

seed = 8
p_fan = 0.85

t_present = 0.3
t_pause = 0.1

integ_tau = 0.1

vocab, fan1, fan1_pair_vecs, fan2, fan2_pair_vecs, \
    foil1, foil1_pair_vecs, foil2, foil2_pair_vecs = make_alt_vocab(5, 5, D, seed, norm=True)
mean_fan1_pair = np.mean(fan1_pair_vecs, axis=0)
mean_fan2_pair = np.mean(fan2_pair_vecs, axis=0)

all_fan = fan1 + fan2
all_fan_pairs = [f"{fa1}+{fa2}" for fa1, fa2 in all_fan]

fan_and_foil = fan1 + fan2 + foil1 + foil2
ff_pairs = [f"{f1}+{f2}" for f1, f2 in fan_and_foil]

encs = choose_encoders(de_n_neurons, D, p_fan, mean_fan1_pair, mean_fan2_pair)

all_vecs = fan1_pair_vecs + fan2_pair_vecs + foil1_pair_vecs + foil2_pair_vecs
concat_vecs = [np.concatenate((v1, v2)) for v1, v2 in all_vecs]
# Note: targets = 1, foil = -1
target_ans = [1] * (len(fan1) + len(fan2))
foil_ans = [-1] * (len(foil1) + len(foil2))
all_ans = target_ans + foil_ans

feed = VecToScalarFeed(concat_vecs, all_ans, t_present, t_pause)

with spa.Network("Associative Model", seed=seed) as model:
    model.stim = nengo.Node(feed.feed)
    model.correct = nengo.Node(feed.get_answer)
    model.reset = nengo.Node(lambda t: feed.paused)

    model.famili_a = spa.WTAAssocMem(
        input_vocab=vocab,
        threshold=0.3,
        function=lambda x: x > 0.)

    model.famili_b = spa.WTAAssocMem(
        input_vocab=vocab,
        threshold=0.3,
        function=lambda x: x > 0.)

    model.designed_ensemble = nengo.Ensemble(de_n_neurons, D, encoders=encs)

    model.cleanup = spa.WTAAssocMem(
        threshold=0.3,
        input_vocab=vocab,
        mapping=dict(zip(all_fan_pairs, all_fan_pairs)),
        function=lambda x: x > 0.)

    model.accum_reset = nengo.Node(size_in=1)
    model.accum = nengo.networks.EnsembleArray(ea_n_neurons, D)
    for ea in model.accum.ea_ensembles:
        nengo.Connection(ea, ea, synapse=integ_tau)

    model.decision = spa.Compare(vocab)

    nengo.Connection(model.stim[:D], model.famili_a.input)
    nengo.Connection(model.stim[D:], model.famili_b.input)
    nengo.Connection(model.famili_a.output, model.designed_ensemble[:D])
    nengo.Connection(model.famili_b.output, model.designed_ensemble[D:])
    nengo.Connection(model.designed_ensemble, model.cleanup.input)
    nengo.Connection(model.cleanup.output, model.accum.input,
                     synapse=integ_tau)
    nengo.Connection(model.reset, model.accum_reset,
                     synapse=None)
    nengo.Connection(model.accum_reset, model.accum.add_neuron_input(),
                     transform=np.ones((ea_n_neurons*D, 1)) * -3,
                     synapse=None)
    nengo.Connection(model.accum.output, model.decision.input_a)
    nengo.Connection(model.stim, model.decision.input_b)

    # ens_spikes = nengo.Probe(model.designed_ensemble.neurons)
    # famili_spikes = []
    # for am_ens in model.famili.selection.thresholding.ea_ensembles:
    #     famili_spikes.append(nengo.Probe(am_ens.neurons))

    p_in = nengo.Probe(model.stim, synapse=None)
    p_accum = nengo.Probe(model.accum.output, synapse=0.01)
    p_clean = nengo.Probe(model.cleanup.output, synapse=0.01)
    p_cor = nengo.Probe(model.correct, synapse=None)
    p_out = nengo.Probe(model.decision.output, synapse=0.1)

with nengo.Simulator(model) as sim:
    sim.run(len(all_vecs)*(t_present+t_pause) + t_pause)
