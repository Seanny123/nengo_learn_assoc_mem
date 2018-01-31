"""Model from the familiarity output to the recognition with an IA cleanup"""

import h5py

import nengo
import nengo_spa as spa

from nengo_learn_assoc_mem.utils import make_alt_vocab, VecToScalarFeed, gen_added_strings, list_as_ascii
from nengo_learn_assoc_mem.meg_ia import MegAssociativeMemory

from collections import OrderedDict


D = 32
seed = 7

t_present = 0.3
t_pause = 0.5

integ_tau = 0.1
n_pairs = 16

vocab, fan1, fan1_pair_vecs, fan2, fan2_pair_vecs, \
    foil1, foil1_pair_vecs, foil2, foil2_pair_vecs = make_alt_vocab(n_pairs, n_pairs, D, seed, norm=True)

all_fan = fan1 + fan2
all_fan_pairs = gen_added_strings(all_fan)

fan_and_foil = fan1 + fan2 + foil1 + foil2

all_vecs = fan1_pair_vecs + fan2_pair_vecs + foil1_pair_vecs + foil2_pair_vecs
# Note: targets = 1, foil = -1
target_ans = [1] * (len(fan1) + len(fan2))
foil_ans = [-1] * (len(foil1) + len(foil2))
all_ans = target_ans + foil_ans

feed = VecToScalarFeed(all_vecs, all_ans, t_present, t_pause)

with spa.Network("Associative Model", seed=seed) as model:
    model.famili = nengo.Node(feed.feed)
    model.correct = nengo.Node(feed.get_answer)
    model.reset = nengo.Node(lambda t: feed.paused)

    n_accum_neurons = [45] * len(fan1) + [35] * len(fan2)
    n_thresh_neurons = [15] * len(all_fan)
    model.cleanup = MegAssociativeMemory(n_accum_neurons=n_accum_neurons,
                                         n_thresholding_neurons=n_thresh_neurons,
                                         input_vocab=vocab,
                                         mapping=OrderedDict(zip(all_fan_pairs, all_fan_pairs)))

    nengo.Connection(model.famili, model.cleanup.input)
    nengo.Connection(model.reset, model.cleanup.input_reset, synapse=None)

    p_in = nengo.Probe(model.famili, synapse=None, label="input")
    p_ia_out = nengo.Probe(model.cleanup.selection.accumulators.output,
                           synapse=0.01, label="clean_accum")
    p_clean = nengo.Probe(model.cleanup.output, synapse=0.01, label="clean")
    p_cor = nengo.Probe(model.correct, synapse=None, label="correct")


with nengo.Simulator(model) as sim:
    sim.run(len(all_vecs)*(t_present+t_pause) + t_pause)

with h5py.File("data/meg_ia_full_shuffled.h5py", "w") as fi:
    tm = fi.create_dataset("t_range", data=[0, sim.trange()[-1]])
    tm.attrs["dt"] = float(sim.dt)
    tm.attrs["t_pause"] = t_pause
    tm.attrs["t_present"] = t_present

    pnt_nms = []
    pnt_vectors = []
    for nm, pnt in vocab.pointers.items():
        pnt_nms.append(nm)
        pnt_vectors.append(pnt.v)

    fi.create_dataset("vocab_strings", data=list_as_ascii(pnt_nms))
    vec = fi.create_dataset("vocab_vectors", data=pnt_vectors)
    vec.attrs["dimensions"] = D

    for nm, pairs in zip(("fan1", "fan2", "foil1", "foil2"), (fan1, fan2, foil1, foil2)):
        added_str = gen_added_strings(pairs)
        fi.create_dataset(nm, data=list_as_ascii(added_str))

    for probe in model.all_probes:
        fi.create_dataset(probe.label, data=sim.data[probe])
