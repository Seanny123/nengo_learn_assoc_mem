import nengo
import numpy as np
import matplotlib.pyplot as plt

from nengo_learn_assoc_mem.utils import LearnDelayVecFeed

feed = LearnDelayVecFeed(np.eye(3), np.eye(3), 0.1, 3, 3, 0.1, 0.05)

with nengo.Network() as model:
    in_nd = nengo.Node(feed.feed)
    learn = nengo.Node(feed.get_learn)

    p_in = nengo.Probe(in_nd)
    p_l = nengo.Probe(learn)

with nengo.Simulator(model) as sim:
    sim.run(1.0)

plt.plot(sim.data[p_in])
plt.plot(sim.data[p_l], color='k', linestyle="--")
plt.show()
