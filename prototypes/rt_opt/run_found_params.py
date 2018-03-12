"""Run and plot results of parameters found from optimisation."""

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nengo
import nengo_spa as spa

from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs
from prototypes.rt_opt.opt_utils import decision, f_in

import multiprocessing
import os

from typing import Tuple

dt = 0.001  # monkey patch to just get things running


def sim_init(shared_inp, shared_accum_out, shared_dims, shared_t_range):
    global in_func
    in_func = shared_inp

    global accum_output
    accum_output = shared_accum_out

    global dims
    dims = shared_dims

    global t_range
    t_range = shared_t_range


def sim_iter(n_runs: int, p_vecs, res_lst):
    for rr in range(n_runs):
        yield rr, p_vecs, res_lst


def sim_net(run: int, p_vecs: np.ndarray, res_lst):

    with spa.Network(seed=run) as model:
        in_nd = nengo.Node(lambda t: in_func[int(t / dt)])
        accum_out = nengo.Node(lambda t: accum_output[int(t / dt)])

        clean_cmp = spa.Compare(dims)

        nengo.Connection(accum_out, clean_cmp.input_a,
                         transform=p_vecs.T, synapse=None)

        nengo.Connection(in_nd, clean_cmp.input_b, synapse=None)

        p_clean_out = nengo.Probe(clean_cmp.output, synapse=0.01)

    with nengo.Simulator(model, progress_bar=False, seed=run) as sim:
        sim.run(t_range[-1])

    res_lst.append(sim.data[p_clean_out].copy())


def run_react(args, init_args: Tuple):
    print(args)
    thresh, fan1_noise_mag, fan2_noise_mag, fan1_reduce, fan2_reduce = args

    fan1_len = len(f_strs['fan1'])

    p_vecs = fan_pair_vecs.copy()
    p_vecs[fan1_len:] += np.random.normal(size=p_vecs[fan1_len:].shape) * fan1_noise_mag
    p_vecs[:fan1_len] += np.random.normal(size=p_vecs[:fan1_len].shape) * fan2_noise_mag
    p_vecs[fan1_len:] *= fan1_reduce
    p_vecs[:fan1_len] *= fan2_reduce

    errs = {ff: [] for ff in f_in}
    rts = {ff: [] for ff in f_in}

    manager = multiprocessing.Manager()
    res_lst = manager.list()

    with multiprocessing.Pool(os.cpu_count(), sim_init, init_args) as pool:
        pool.starmap(sim_net, sim_iter(10, p_vecs, res_lst))

    for res in res_lst:
        decs = []
        t_decs = []

        for t_slice in all_time_slices:
            cl_slc = res[t_slice + td_pause:t_slice + td_each]
            tm, dec = decision(cl_slc, thresh=thresh, dec_thresh=1.0)
            decs.append(dec)
            t_decs.append(tm)

        decs = np.array(decs)
        t_decs = np.array(t_decs)

        last_idx = 0
        for lst, lbl in zip((f_strs['fan1'], f_strs['fan2'], f_strs['foil1'], f_strs['foil2']), f_in):
            comp_slc = slice(last_idx, last_idx + len(lst))

            errs[lbl].append(np.sum(decs[comp_slc] != match_correct[comp_slc]) / len(lst))
            rts[lbl].append(np.mean(t_decs[comp_slc]))
            last_idx += len(lst)

    return errs, rts


if __name__ == '__main__':

    with h5py.File("data/meg_ia_full.h5py", "r") as fi:
        print(list(fi.keys()))
        inp = list(np.array(fi['input']))

        fan1 = numpy_bytes_to_str(fi['fan1'])
        fan2 = numpy_bytes_to_str(fi['fan2'])
        foil1 = numpy_bytes_to_str(fi['foil1'])
        foil2 = numpy_bytes_to_str(fi['foil2'])

        v_strs = numpy_bytes_to_str(fi['vocab_strings'])
        v_vecs = list(fi['vocab_vectors'])
        D = fi['vocab_vectors'].attrs['dimensions']

        accum = list(np.array(fi['clean_accum']))

        #dt = fi['t_range'].attrs['dt']
        t_range = np.arange(fi['t_range'][0], fi['t_range'][1], dt)
        t_pause = fi['t_range'].attrs['t_pause']
        t_present = fi['t_range'].attrs['t_present']

    t_each = t_pause + t_present
    td_each = int(t_each/dt)
    td_pause = int(t_pause/dt)

    all_time_slices = list(range(0, int((t_range[-1])/dt), int(td_each)))[:-1]

    vocab = spa.Vocabulary(D)
    for val, vec in zip(v_strs, v_vecs):
        vocab.add(val, vec)

    fan1_pair_vecs = norm_spa_vecs(vocab, fan1)
    fan2_pair_vecs = norm_spa_vecs(vocab, fan2)
    foil1_pair_vecs = norm_spa_vecs(vocab, foil1)
    foil2_pair_vecs = norm_spa_vecs(vocab, foil2)

    match_correct = [1] * (len(fan1) + len(fan2)) + [-1] * (len(fan1) + len(fan2))

    fan_pair_vecs = np.array(fan1_pair_vecs + fan2_pair_vecs)
    f_strs = {"fan1": fan1, "fan2": fan2, "foil1": foil1, "foil2": foil2}

    params = [0.7811007509368267, 0.19625920377375358, 0.011436221080962653, 0.8966996121060862, 0.8760664213580347]
    initial_proc_args = (inp, accum, D, t_range)
    err_res, rt_res = run_react(params, initial_proc_args)

    err_lst = []
    for nm, err in err_res.items():
        err_lst.extend([(nm, ee) for ee in err])

    err_df = pd.DataFrame(err_lst, columns=("trial", "err"))
    sns.barplot(x="trial", y="err", data=err_df)
    plt.show()

    rt_lst = []
    for nm, rt in rt_res.items():
        rt_lst.extend([(nm, rr) for rr in rt])

    rt_df = pd.DataFrame(rt_lst, columns=("trial", "rt"))
    sns.barplot(x="trial", y="rt", data=rt_df)
    plt.show()
