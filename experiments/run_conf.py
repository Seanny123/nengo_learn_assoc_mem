from typing import Dict
from collections import OrderedDict
import os

import h5py
import numpy as np
import nengo_spa as spa
import nengo

from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs
from nengo_learn_assoc_mem.paths import data_path


def run_react(in_vec: np.ndarray, n_nrns: int, dims: int,
              enc: np.ndarray, cepts: np.ndarray,
              t_sim: float, thresh=.7) -> Dict[str, np.ndarray]:

    with spa.Network() as model:
        in_nd = nengo.Node(lambda t: in_vec)

        # max_rates should be set deterministically via seed
        ens = nengo.Ensemble(n_nrns, dims,
                             encoders=enc,
                             intercepts=cepts, seed=seed)
        cmp = spa.Compare(dims)
        cmp_res = nengo.Ensemble(200, 1)

        integ_tau = 0.1
        integ_nrns = 100
        pos_integ = nengo.networks.Integrator(integ_tau, integ_nrns, 1)
        neg_integ = nengo.networks.Integrator(integ_tau, integ_nrns, 1)

        nengo.Connection(in_nd, ens, synapse=None)
        nengo.Connection(ens, cmp.input_a, synapse=.01)
        nengo.Connection(in_nd, cmp.input_b, synapse=.01)
        nengo.Connection(cmp.output, cmp_res)
        nengo.Connection(cmp_res, pos_integ.input,
                         function=lambda x: x - thresh)
        nengo.Connection(cmp_res, neg_integ.input,
                         function=lambda x: -x + thresh)

        p_cmp_out = nengo.Probe(cmp.output, synapse=.01)
        p_pos_out = nengo.Probe(pos_integ.output, synapse=.05)
        p_neg_out = nengo.Probe(neg_integ.output, synapse=.05)

    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(t_sim)

    return {"cmp": sim.data[p_cmp_out],
            "pos": sim.data[p_pos_out],
            "neg": sim.data[p_neg_out]}


stim_types = ("fan1", "fan2", "foil1", "foil2")

save_path = os.path.join(data_path, "static_react", "neg_voja_enc.h5")

seed_val = 0
#for seed_val in range(10):
read_path = os.path.join(data_path, "mixed_static", f"static_voja_alt_{seed_val}.h5")
stim_strs = []

with h5py.File(read_path, "r") as fi:

    for s_t in stim_types:
        stim_strs.append((s_t, numpy_bytes_to_str(fi[s_t])))

    v_strs = numpy_bytes_to_str(fi['vocab_strings'])
    v_vecs = list(fi['vocab_vectors'])
    dimensions = fi['vocab_vectors'].attrs['dimensions']

    encoders = np.array(fi['encoders'])
    n_neurons = encoders.shape[0]
    intercepts = np.ones(n_neurons) * fi['encoders'].attrs['intercept']
    seed = fi['encoders'].attrs['seed']

    tm = fi["t_range"]
    dt = tm.attrs["dt"]
    t_pause = tm.attrs["t_pause"]
    t_present = tm.attrs["t_present"]

vocab = spa.Vocabulary(dimensions)
for val, vec in zip(v_strs, v_vecs):
    vocab.add(val, vec)

stim_vecs = OrderedDict((
                ("fan1", None),
                ("fan2", None),
                ("foil1", None),
                ("foil2", None)))
for (nm, strs) in stim_strs:
    stim_vecs[nm] = norm_spa_vecs(vocab, strs)

feed_vecs = []
for vecs in stim_vecs.values():
    feed_vecs += vecs

feed_idx = 0
res = run_react(feed_vecs[feed_idx], n_neurons, dimensions,
                encoders, intercepts, seed)

# save output confidence for each input vector
