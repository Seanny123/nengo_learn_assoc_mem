import h5py
import numpy as np

import nengo
import nengo_spa as spa

from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs
from prototypes.rt_opt.opt_utils import decision, loss_func, f_in

import multiprocessing
from multiprocessing import dummy
import os
from collections import OrderedDict

from typing import List, Tuple

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


def opt_iter(arg_lst, loss_lst, init_args):
    for a_i, arg in enumerate(arg_lst):
        yield a_i, arg, loss_lst, init_args


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


def opt_run_react(a_idx: int, args, loss_lst: List, init_args: Tuple):
    print(args)
    thresh, fan1_noise_mag, fan2_noise_mag, fan1_reduce, fan2_reduce = args

    fan1_len = len(f_strs['fan1'])

    p_vecs = pair_vecs.copy()
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

        for t_slice in time_slices:
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

    loss_lst[a_idx] = loss_func(errs, rts)


def param_to_arglist(params: OrderedDict, child_num: int):
    arg_list = []
    for c_n in range(child_num):
        arg_list.append([p_vals[c_n] for p_vals in params.values()])

    return arg_list


def init_opt(shared_pair_vecs, shared_time_slices, shared_td_pause, shared_td_each, shared_f_strs):
    global pair_vecs
    pair_vecs = shared_pair_vecs

    global time_slices
    time_slices = shared_time_slices

    global td_pause
    td_pause = shared_td_pause

    global td_each
    td_each = shared_td_each

    global f_strs
    f_strs = shared_f_strs


def genetic_opt(run_func, param_space: OrderedDict, thread_init_args: Tuple, proc_init_args, run_num: int, child_num=4):
    gen_manager = multiprocessing.Manager()
    loss_lst = gen_manager.list([np.inf for _ in range(child_num)])

    # choose initial args
    params = OrderedDict([(nm, []) for nm in param_space.keys()])
    for nm, v_range in param_space.items():
        params[nm] = np.random.uniform(v_range[0], v_range[1], size=child_num)

    arg_list = param_to_arglist(params, child_num)

    for r_n in range(run_num):

        with dummy.Pool(os.cpu_count(), init_opt, thread_init_args) as pool:
            pool.starmap(run_func, opt_iter(arg_list, loss_lst, proc_init_args))

        # given minimum loss, save it and spawn more children
        print(f"Best of batch: {base_args}\n")
        fittest_idx = int(np.argmin(loss_lst))
        base_args = arg_list[fittest_idx]

        # mutate offspring from fittest individual
        params = OrderedDict([(nm, []) for nm in param_space.keys()])
        for s_i, (nm, v_range) in enumerate(param_space.items()):
            std = (v_range[1] - v_range[0]) / 2
            noise = np.random.normal(0, std, size=child_num)
            new_param = base_args[s_i] + noise
            params[nm] = np.clip(new_param, v_range[0], v_range[1])

        arg_list = param_to_arglist(params, child_num)

        loss_lst = gen_manager.list([np.inf for _ in range(child_num)])

    return True


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

    space = OrderedDict((
        ('cls_thresh', (0.5, 1.2)),
        ('f1_noise', (0.0, 0.2)),
        ('f2_noise', (0.0, 0.2)),
        ('f1_reduced', (0.8, 1.0)),
        ('f2_reduced', (0.8, 1.0))
    ))

    initial_global_args = (fan_pair_vecs, all_time_slices, td_pause, td_each, f_strs)
    initial_proc_args = (inp, accum, D, t_range)

    best = genetic_opt(opt_run_react, space, initial_global_args, initial_proc_args, 200, 8)
