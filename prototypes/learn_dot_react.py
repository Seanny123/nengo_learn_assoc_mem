import h5py
import numpy as np

import nengo
import nengo_spa as spa

from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs


with h5py.File("data/meg_ia_full_shuffled.h5py", "r") as fi:
    print(list(fi.keys()))
    inp = list(np.array(fi['input']))
    cor = list(np.array(fi['correct']))

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

n_neurons = 1000
seed = 8

with nengo.Network(seed=seed) as model:
    in_nd = nengo.Node(lambda t: inp[int(t/dt)])
    accum_out = nengo.Node(lambda t: accum[int(t/dt)])
    correct = nengo.Node(lambda t: cor[int(t/dt)])

    err_nd = nengo.Node(lambda t, x: x[0] - x[1], size_in=2)
    output = nengo.Node(size_in=1)

    cmp = nengo.Ensemble(1000, 2*D)

    nengo.Connection(accum_out, cmp[D:],
                     transform=pair_vecs.T, synapse=None)
    nengo.Connection(in_nd, cmp[:D], synapse=None)
    conn_out = nengo.Connection(cmp, output,
                                transform=np.zeros((1, 2*D)),
                                learning_rule_type=nengo.PES(1e-5))
    nengo.Connection(err_nd, conn_out.learning_rule, synapse=None)
    nengo.Connection(output, err_nd[0])
    nengo.Connection(correct, err_nd[1])

    p_err = nengo.Probe(err_nd, synapse=0.01)
    p_clean_out = nengo.Probe(output, synapse=0.01)
    p_dec = nengo.Probe(conn_out, 'weights', sample_every=t_present+t_pause)

with nengo.Simulator(model) as sim:
    sim.run(t_range[-1])


fi_name = "learn_dot"
with h5py.File(f"data/{fi_name}.h5py", "w") as out_fi:
    out_fi.create_dataset("clean_out", data=sim.data[p_clean_out])
    out_fi.create_dataset("err", data=sim.data[p_err])
    out_fi.create_dataset("dec", data=sim.data[p_dec])
