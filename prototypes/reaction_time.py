import h5py
import numpy as np

import nengo
import nengo_spa as spa

from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs

import itertools
import multiprocessing
import os


def iter_params(run_num: int, f1_nois, f1_reds, f2_nois, f2_reds):
    f1_prod = itertools.product(f1_nois, f1_reds)
    f2_prod = itertools.product(f2_nois, f2_reds)
    for (f1_nn, f1_re), (f2_nn, f2_re) in zip(f1_prod, f2_prod):
        for rr in range(run_num):
            fi_nm = f"more_react_explore_{rr}_{f1_nn}_{f1_re}_{f2_nn}_{f2_re}"
            yield pair_vecs.copy(), fi_nm, f1_nn, f1_re, f2_nn, f2_re


def run_react(p_vecs: np.ndarray, fi_name: str,
              fan1_noise_mag: float, fan1_reduce: float, fan2_noise_mag: float, fan2_reduce: float):
    print(fi_name)
    p_vecs[len(fan1):] += np.random.normal(size=p_vecs[len(fan1):].shape) * fan1_noise_mag
    p_vecs[:len(fan1)] += np.random.normal(size=p_vecs[:len(fan1)].shape) * fan2_noise_mag
    p_vecs[len(fan1):] *= fan1_reduce
    p_vecs[:len(fan1)] *= fan2_reduce

    with spa.Network() as model:
        in_nd = nengo.Node(lambda t: inp[int(t/dt)])
        accum_out = nengo.Node(lambda t: accum[int(t/dt)])

        clean_cmp = spa.Compare(D)

        nengo.Connection(accum_out, clean_cmp.input_a,
                         transform=p_vecs.T, synapse=None)

        nengo.Connection(in_nd, clean_cmp.input_b, synapse=None)

        p_clean_out = nengo.Probe(clean_cmp.output, synapse=0.01)

    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(t_range[-1])

    with h5py.File(f"data/{fi_name}.h5py", "w") as out_fi:
        cl = out_fi.create_dataset("clean_out", data=sim.data[p_clean_out])
        cl.attrs["fan1_noise"] = fan1_noise_mag
        cl.attrs["fan1_reduce"] = fan1_reduce
        cl.attrs["fan2_noise"] = fan2_noise_mag
        cl.attrs["fan2_reduce"] = fan2_reduce


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

vocab = spa.Vocabulary(D)
for val, vec in zip(v_strs, v_vecs):
    vocab.add(val, vec)

fan1_pair_vecs = norm_spa_vecs(vocab, fan1)
fan2_pair_vecs = norm_spa_vecs(vocab, fan2)
foil1_pair_vecs = norm_spa_vecs(vocab, foil1)
foil2_pair_vecs = norm_spa_vecs(vocab, foil2)

pair_vecs = np.array(fan1_pair_vecs + fan2_pair_vecs)

fan1_noises = (0.16,)
fan1_reduces = (0.9,)
fan2_noises = (0.03,)
fan2_reduces = (0.95,)

with multiprocessing.Pool(os.cpu_count()) as pool:
    pool.starmap(run_react, iter_params(10, fan1_noises, fan1_reduces, fan2_noises, fan2_reduces))
