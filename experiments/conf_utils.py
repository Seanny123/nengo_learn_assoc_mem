from typing import List, Dict

import nengo
import nengo_spa as spa
import numpy as np

from nengo_learn_assoc_mem.utils import BasicVecFeed


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
