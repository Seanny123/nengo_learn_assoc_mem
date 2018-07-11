from typing import Tuple, Dict

import h5py
import nengo
import nengolib
import numpy as np
from matplotlib import pyplot as plt

from nengo_learn_assoc_mem.utils import BasicVecFeed, list_as_ascii, gen_added_strings


class Constants(object):

    def __init__(self, n_neurons: int, dimensions: int, n_items: int, t_pause: float, t_present: float, dt=0.001):
        super().__init__()
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.n_items = n_items

        self.t_pause = t_pause
        self.t_present = t_present
        self.t_each = t_pause + t_present
        self.td_each = int(self.t_each / dt)
        self.td_pause = int(self.t_pause / dt)

        self.fan1_slc = slice(self.td_pause,
                              self.td_each * n_items + self.td_pause)
        self.fan2_slc = slice(self.fan1_slc.stop,
                              self.fan1_slc.stop + self.td_each * n_items)
        self.foil1_slc = slice(self.fan2_slc.stop,
                               self.fan2_slc.stop + self.td_each * n_items)
        self.foil2_slc = slice(self.foil1_slc.stop,
                               self.foil1_slc.stop + self.td_each * n_items)


def get_ens_params(cnst: Constants, cepts: np.ndarray, seed: float) -> Tuple[np.ndarray, np.ndarray]:
    with nengolib.Network(seed=seed) as model:
        ens = nengo.Ensemble(cnst.n_neurons, cnst.dimensions,
                             intercepts=cepts, seed=seed)

    with nengo.Simulator(model) as sim:
        pass

    return sim.data[ens].encoders.copy(), sim.data[ens].max_rates.copy()


def train_mixed_encoders(cnst: Constants, learning_rule, feed_vecs: np.ndarray,
                         enc: np.ndarray, rates: np.ndarray, cepts: np.ndarray,
                         learning_kwargs: Dict,
                         act_syn: float, seed: float) -> np.ndarray:
    feed = BasicVecFeed(feed_vecs, feed_vecs,
                        cnst.t_present, cnst.dimensions, len(feed_vecs), cnst.t_pause)

    with nengolib.Network(seed=seed) as model:
        in_nd = nengo.Node(feed.feed)
        paused = nengo.Node(lambda t: 1 - feed.paused)

        neg_voja = learning_rule(enc.copy(), rates, **learning_kwargs)
        ens = nengo.Ensemble(cnst.n_neurons, cnst.dimensions,
                             intercepts=cepts, max_rates=rates, seed=seed)

        nengo.Connection(in_nd, neg_voja.input_signal, synapse=None)
        nengo.Connection(ens.neurons, neg_voja.input_activities,
                         synapse=act_syn)
        nengo.Connection(paused, neg_voja.enable, synapse=None)
        nengo.Connection(neg_voja.output, ens.neurons, synapse=None)

    t_sim = len(feed_vecs) * cnst.t_each + cnst.t_pause
    print("Simulating for", t_sim)
    with nengo.Simulator(model) as sim:
        sim.run(t_sim)

    return neg_voja.encoder_hist


def test_response(cnst: Constants, feed_vecs, vo, vo_strs: Dict,
                  encs: np.ndarray, cepts: np.ndarray, max_rates,
                  save_path: str, enc_args: Dict,
                  seed: float, plt_res=False):

    feed = BasicVecFeed(feed_vecs, feed_vecs,
                        cnst.t_present, cnst.dimensions, len(feed_vecs), cnst.t_pause)

    with nengolib.Network(seed=seed) as test_model:
        in_nd = nengo.Node(feed.feed)
        pause = nengo.Node(lambda t: feed.paused)
        meg_nd = nengo.Node(lambda t, x: np.sum(x),
                            size_in=cnst.n_neurons, size_out=1)

        ens = nengo.Ensemble(cnst.n_neurons, cnst.dimensions,
                             encoders=encs.copy(),
                             intercepts=cepts,
                             max_rates=max_rates,
                             seed=seed)

        nengo.Connection(in_nd, ens, synapse=None)
        nengo.Connection(pause, ens.neurons, transform=-10 * np.ones((cnst.n_neurons, 1)), synapse=None)

        nengo.Connection(ens.neurons, meg_nd, synapse=None)

        p_meg = nengo.Probe(meg_nd, synapse=0.01)

    t_sim = len(feed_vecs) * cnst.t_each + cnst.t_pause
    with nengo.Simulator(test_model) as test_sim:
        test_sim.run(t_sim)

    test_meg = test_sim.data[p_meg].squeeze()

    targ_shape = (-1, cnst.td_each)
    fan1_resp = np.mean(
        test_meg[cnst.fan1_slc].reshape(targ_shape), axis=0)
    fan2_resp = np.mean(
        test_meg[cnst.fan2_slc].reshape(targ_shape), axis=0)
    foil1_resp = np.mean(
        test_meg[cnst.foil1_slc].reshape(targ_shape), axis=0)
    foil2_resp = np.mean(
        test_meg[cnst.foil2_slc].reshape(targ_shape), axis=0)

    if plt_res:
        plt.figure()
        plt.plot(fan1_resp)
        plt.plot(fan2_resp)
        plt.plot(foil1_resp)
        plt.plot(foil2_resp)

        plt.legend(["fan1 targ", "fan2 targ", "foil1 targ", "foil2 targ"], facecolor=None)
        plt.show()

    # save the resulting activities, weights and vocab
    with h5py.File(save_path, "w") as fi:
        fi.create_dataset("resp/fan1", data=fan1_resp)
        fi.create_dataset("resp/fan2", data=fan2_resp)
        fi.create_dataset("resp/foil1", data=foil1_resp)
        fi.create_dataset("resp/foil2", data=foil2_resp)

        tm = fi.create_dataset("t_range", data=[0, t_sim])
        tm.attrs["dt"] = float(test_sim.dt)
        tm.attrs["t_pause"] = cnst.t_pause
        tm.attrs["t_present"] = cnst.t_present

        enc = fi.create_dataset("encoders", data=encs)
        enc.attrs["seed"] = seed
        for key, val in enc_args.items():
            enc.attrs[key] = val

        pnt_nms = []
        pnt_vectors = []
        for nm, pnt in vo.items():
            pnt_nms.append(nm)
            pnt_vectors.append(pnt.v)

        fi.create_dataset("vocab_strings", data=list_as_ascii(pnt_nms))
        vec = fi.create_dataset("vocab_vectors", data=pnt_vectors)
        vec.attrs["dimensions"] = cnst.dimensions

        for nm, pairs in vo_strs.items():
            added_str = gen_added_strings(pairs)
            fi.create_dataset(nm, data=list_as_ascii(added_str))
