from typing import Dict, List
from collections import OrderedDict
import os

import h5py
import numpy as np
import nengo_spa as spa
import nengo

from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs, BasicVecFeed
from nengo_learn_assoc_mem.paths import data_path


def train_decoders(in_vecs: List[np.ndarray],  n_nrns: int, dims: int,
                   enc: np.ndarray, cepts: np.ndarray, seed: int, t_pres: float, t_paus: float) -> np.ndarray:
    t_each = t_pres + t_paus
    feed = BasicVecFeed(in_vecs, in_vecs, t_pres, dims, len(in_vecs), t_paus)

    with nengo.Network(seed=seed) as learned_model:
        in_nd = nengo.Node(feed.feed)
        correct = nengo.Node(feed.get_answer)
        learning = nengo.Node(lambda t: -feed.paused)
        pause = nengo.Node(lambda t: feed.paused)
        output = nengo.Node(size_in=dims)

        ens = nengo.Ensemble(n_nrns, dims,
                             encoders=enc, intercepts=cepts, seed=seed)

        nengo.Connection(in_nd, ens, synapse=None)
        conn_out = nengo.Connection(ens.neurons, output, transform=np.zeros((dims, n_nrns)),
                                    learning_rule_type=nengo.PES(3e-2))
        nengo.Connection(pause, ens.neurons, transform=-10 * np.ones((n_nrns, 1)), synapse=None)

        # Error flow node
        pes_learn_control = nengo.Node(
            lambda t, x: x[:-1] if x[-1] >= 0 else x[:-1] * 0,
            size_in=dims + 1)
        nengo.Connection(pes_learn_control,
                         conn_out.learning_rule)

        # Error calculation connections
        nengo.Connection(output, pes_learn_control[:-1],
                         synapse=None)
        nengo.Connection(correct, pes_learn_control[:-1],
                         transform=-1, synapse=None)
        # Control connection
        nengo.Connection(learning, pes_learn_control[-1],
                         synapse=None)

        p_dec = nengo.Probe(conn_out, 'weights', sample_every=0.1)

    with nengo.Simulator(learned_model) as learned_sim:
        learned_sim.run(len(in_vecs) * t_each + t_paus)

    return learned_sim.data[p_dec][-1]


def run_comp(in_vec: np.ndarray, dec: np.ndarray, n_nrns: int, dims: int,
             enc: np.ndarray, cepts: np.ndarray, seed: int,
             t_sim: float) -> np.ndarray:

    with spa.Network(seed=seed) as model:
        in_nd = nengo.Node(lambda t: in_vec)

        # max_rates should be set deterministically via seed
        ens = nengo.Ensemble(n_nrns, dims,
                             encoders=enc,
                             intercepts=cepts, seed=seed)
        cmp = spa.Compare(dims)

        nengo.Connection(in_nd, ens, synapse=None)
        nengo.Connection(ens.neurons, cmp.input_a, synapse=.01, transform=dec)
        nengo.Connection(in_nd, cmp.input_b, synapse=.01)

        p_cmp_out = nengo.Probe(cmp.output, synapse=.01)

    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(t_sim)

    cmp_out = sim.data[p_cmp_out]
    print(f"Final cmp val: {cmp_out[-1]}")
    return cmp_out


def run_integ(cmp_out: List, seed: float, dt=0.001, pos_adjust=0.3) -> Dict[str, np.ndarray]:
    t_sim = len(cmp_out) * dt

    with spa.Network(seed=seed) as model:
        in_nd = nengo.Node(lambda t: cmp_out[int((t-dt)/dt)])
        cmp_res = nengo.Ensemble(300, 1)

        integ_tau = 0.1
        integ_nrns = 300
        pos_integ = nengo.networks.Integrator(integ_tau, integ_nrns, 1)
        neg_integ = nengo.networks.Integrator(integ_tau, integ_nrns, 1)

        nengo.Connection(in_nd, cmp_res, synapse=None)
        nengo.Connection(cmp_res, pos_integ.input,
                         function=lambda x: x - pos_adjust)
        nengo.Connection(cmp_res, neg_integ.input,
                         function=lambda x: 1 - x)

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

        tm = fi["t_range"]
        dt_sim = tm.attrs["dt"]
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

    train_vecs = stim_vecs["fan1"] + stim_vecs["fan2"]
    decs = train_decoders(train_vecs, n_neurons, dimensions,
                          encoders, intercepts, init_seed, t_present, t_pause)
    feed_vecs = []
    for vecs in stim_vecs.values():
        feed_vecs += vecs

    print("simulating comparisons")
    all_comp_res = []
    final_vals = []
    for f_i, f_vec in enumerate(feed_vecs):
        comp_res = run_comp(f_vec, decs, n_neurons, dimensions,
                            encoders, intercepts, init_seed, 0.2)
        final_vals.append(comp_res[-1])
        all_comp_res.append(comp_res)

    train_len = len(train_vecs)
    max_targ_comp = np.max(final_vals[:train_len])
    min_targ_comp = np.min(final_vals[:train_len])
    max_foil_comp = np.max(final_vals[train_len:])
    min_foil_comp = np.min(final_vals[train_len:])

    with h5py.File(save_path, "a") as w_fi:
        w_fi.create_dataset(f"comp_res_{seed_val}", data=all_comp_res)

    neg_integ_res = []
    pos_integ_res = []
    print("simulating integration")
    for c_i, c_res in enumerate(all_comp_res):
        integ_res = run_integ(list(c_res)[10:125], init_seed, pos_adjust=min_foil_comp - .05)
        neg_integ_res.append(integ_res["neg"])
        pos_integ_res.append(integ_res["pos"])

    print(f"Done seed {seed_val}")
    # save output confidence for each input vector
    with h5py.File(save_path, "a") as w_fi:
        w_fi.create_dataset(f"integ_res_pos_{seed_val}",
                            data=pos_integ_res)
        w_fi.create_dataset(f"integ_res_neg_{seed_val}",
                            data=neg_integ_res)
