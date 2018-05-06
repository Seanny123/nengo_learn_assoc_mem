from nengo_learn_assoc_mem.utils import ans_conf

import nengo_spa as spa
import nengo
import matplotlib.pyplot as plt
import numpy as np

import string


seed = 8
D = 16
n_items = 10

vocab = spa.Vocabulary(D, max_similarity=0.3)
vocab.populate(";".join([string.ascii_uppercase[i] for i in range(n_items)]))

td_item = 300


def in_func(t):
    if 0.1 < t < 0.3:
        return 'A'
    elif 0.4 < t < 0.6:
        return 'B'
    elif 0.7 < t < 0.9:
        return '0.5*C'
    else:
        return '0'


def corr_func(t):
    if 0.1 < t < 0.3:
        return 'A'
    elif 0.4 < t < 0.6:
        return 'C'
    elif 0.7 < t < 0.9:
        return 'C'
    else:
        return '0'


with spa.Network() as model:
    in_nd = spa.Transcode(in_func, output_vocab=D)
    cor = spa.Transcode(corr_func, output_vocab=D)
    output = spa.Scalar()
    comp = nengo.Node(lambda t, x: np.dot(x[:D], x[D:]), size_in=2*D, size_out=1)

    nengo.Connection(in_nd.output, comp[:D], synapse=None)
    nengo.Connection(cor.output, comp[D:], synapse=None)

    spa.dot(in_nd, cor) >> output

    p_in = nengo.Probe(in_nd.output)
    p_cor = nengo.Probe(cor.output)
    p_out = nengo.Probe(output.output, synapse=0.05)
    p_comp = nengo.Probe(comp)

with nengo.Simulator(model) as sim:
    sim.run(1.1)

plt.plot(sim.data[p_out])
plt.plot(sim.data[p_comp])
plt.show()

conf = ans_conf(sim.data[p_in][:900], sim.data[p_cor][:900], 3, td_item)
dot_conf = np.array([
    np.sum(sim.data[p_comp][:300]),
    np.sum(sim.data[p_comp][300:600]),
    np.sum(sim.data[p_comp][600:900])
])

assert conf[1] < conf[2] < conf[0]
assert np.allclose(conf, dot_conf)
