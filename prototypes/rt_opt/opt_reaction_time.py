"""Try to match the experimental reaction time by optimising the parameters with hyperopt"""

import h5py
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe

import nengo
import nengo_spa as spa

from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs
from prototypes.rt_opt.opt_utils import decision, loss_func, f_in


def opt_run_react(args):
    print(args)
    thresh, fan1_noise_mag, fan2_noise_mag, fan1_reduce, fan2_reduce = args

    p_vecs = pair_vecs.copy()
    p_vecs[len(fan1):] += np.random.normal(size=p_vecs[len(fan1):].shape) * fan1_noise_mag
    p_vecs[:len(fan1)] += np.random.normal(size=p_vecs[:len(fan1)].shape) * fan2_noise_mag
    p_vecs[len(fan1):] *= fan1_reduce
    p_vecs[:len(fan1)] *= fan2_reduce

    # TODO: how to share this dictionary between multiple processes
    errs = {ff: [] for ff in f_in}
    rts = {ff: [] for ff in f_in}

    for run in range(10):

        with spa.Network(seed=run) as model:
            in_nd = nengo.Node(lambda t: inp[int(t/dt)])
            accum_out = nengo.Node(lambda t: accum[int(t/dt)])

            clean_cmp = spa.Compare(D)

            nengo.Connection(accum_out, clean_cmp.input_a,
                             transform=p_vecs.T, synapse=None)

            nengo.Connection(in_nd, clean_cmp.input_b, synapse=None)

            p_clean_out = nengo.Probe(clean_cmp.output, synapse=0.01)

        with nengo.Simulator(model, progress_bar=False, seed=run) as sim:
            sim.run(t_range[-1])

        clean_out = sim.data[p_clean_out]

        decs = []
        t_decs = []

        for t_slice in all_time_slices:
            cl_slc = clean_out[t_slice + td_pause:t_slice + td_each]
            tm, dec = decision(cl_slc, thresh=thresh, dec_thresh=1.0)
            decs.append(dec)
            t_decs.append(tm)

        decs = np.array(decs)
        t_decs = np.array(t_decs)

        last_idx = 0
        for lst, lbl in zip((fan1, fan2, foil1, foil2), f_in):
            comp_slc = slice(last_idx, last_idx+len(lst))

            errs[lbl].append(np.sum(decs[comp_slc] != match_correct[comp_slc]) / len(lst))
            rts[lbl].append(np.mean(t_decs[comp_slc]))
            last_idx += len(lst)

    return loss_func(errs, rts)


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

    dt = fi['t_range'].attrs['dt']
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

pair_vecs = np.array(fan1_pair_vecs + fan2_pair_vecs)

space = (
    hp.uniform('cls_thresh', 0.5, 1.2),
    hp.uniform('f1_noise', 0.0, 0.2),
    hp.uniform('f2_noise', 0.0, 0.2),
    hp.uniform('f1_reduced', 0.8, 1.0),
    hp.uniform('f2_reduced', 0.8, 1.0),
)

best = fmin(opt_run_react, space=space, algo=tpe.suggest, max_evals=100)
