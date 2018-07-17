from typing import Dict, List
from collections import OrderedDict
import os

import h5py
import numpy as np
import nengo_spa as spa
import nengo

from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs
from nengo_learn_assoc_mem.paths import data_path


def run_comp(in_vec: np.ndarray, n_nrns: int, dims: int,
             enc: np.ndarray, cepts: np.ndarray, seed: int,
             t_sim: float, thresh=.7) -> np.ndarray:

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

    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(t_sim)

    return sim.data[p_cmp_out]


def run_integ(cmp_out: List, thresh=.7, dt=0.001) -> Dict[str, np.ndarray]:
    t_sim = len(cmp_out) * dt

    with spa.Network() as model:
        in_nd = nengo.Node(lambda t: cmp_out[int((t-dt)/dt)])
        cmp_res = nengo.Ensemble(200, 1)

        integ_tau = 0.1
        integ_nrns = 100
        pos_integ = nengo.networks.Integrator(integ_tau, integ_nrns, 1)
        neg_integ = nengo.networks.Integrator(integ_tau, integ_nrns, 1)

        nengo.Connection(in_nd, cmp_res)
        nengo.Connection(cmp_res, pos_integ.input,
                         function=lambda x: x - thresh)
        nengo.Connection(cmp_res, neg_integ.input,
                         function=lambda x: -x + thresh)

        p_pos_out = nengo.Probe(pos_integ.output, synapse=.05)
        p_neg_out = nengo.Probe(neg_integ.output, synapse=.05)

    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(t_sim)

    return {"pos": sim.data[p_pos_out], "neg": sim.data[p_neg_out]}


stim_types = ("fan1", "fan2", "foil1", "foil2")

save_path = os.path.join(data_path, "static_react", "confidence.h5")

for seed_val in range(10):
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
        intercepts = np.full(n_neurons, fi['encoders'].attrs['intercept'])
        init_seed = fi['encoders'].attrs['seed']

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

    all_comp_res = []
    for f_vec in feed_vecs:
        comp_res = run_comp(f_vec, n_neurons, dimensions,
                            encoders, intercepts, init_seed, 0.2)
        all_comp_res.append(comp_res)

    with h5py.File(save_path, "w") as w_fi:
        w_fi.create_dataset(f"comp_res_{seed_val}", data=all_comp_res)

    neg_integ_res = []
    pos_integ_res = []
    for c_res in all_comp_res:
        integ_res = run_integ(list(c_res))
        neg_integ_res.append(integ_res["neg"])
        pos_integ_res.append(integ_res["pos"])

    # save output confidence for each input vector
    with h5py.File(save_path, "w") as w_fi:
        w_fi.create_dataset(f"integ_res_pos_{seed_val}", data=pos_integ_res)
        w_fi.create_dataset(f"integ_res_neg_{seed_val}", data=neg_integ_res)
