import h5py
import numpy as np

import nengo
import nengo_spa as spa

from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs
import itertools


def run_react(p_vecs: np.ndarray, fi_name: str, noise_mag=0.1, reduce=0.8):
    p_vecs[:len(fan1)] += np.random.normal(size=p_vecs[:len(fan1)].shape) * noise_mag
    p_vecs[len(fan1):] += np.random.normal(size=p_vecs[len(fan1):].shape) * noise_mag
    p_vecs[len(fan1):] *= reduce

    with spa.Network() as model:
        in_nd = nengo.Node(lambda t: inp[int(t/dt)])
        accum_out = nengo.Node(lambda t: accum[int(t/dt)])

        clean_cmp = nengo.Node(lambda t, x: np.dot(x[:D], x[D:]))

        nengo.Connection(accum_out, clean_cmp[:D],
                         transform=p_vecs.T, synapse=None)

        nengo.Connection(in_nd, clean_cmp[:D], synapse=None)

        p_clean_out = nengo.Probe(clean_cmp.output, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(t_range[-1])

    with h5py.File(f"data/{fi_name}.h5py", "w") as out_fi:
        cl = out_fi.create_dataset("clean_out", data=sim.data[p_clean_out])
        cl.attrs["noise"] = noise_mag
        cl.attrs["reduce"] = reduce


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

all_vecs = vocab.parse("+".join(list(vocab.keys()))).v
all_vecs = all_vecs / np.linalg.norm(all_vecs)

pair_vecs = np.array(fan1_pair_vecs + fan2_pair_vecs)

for noise, red in itertools.product((0.05, 0.1, 0.2), (0.7, 0.8, 0.9)):
    for run in range(10):
        run_react(pair_vecs.copy(), f"fast_react_explore_{run}_{noise}_{red}")
