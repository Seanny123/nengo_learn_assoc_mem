import numpy as np

from nengo_learn_assoc_mem.utils import conf_metric

res = np.ones((300, 10), dtype=np.float)
res[:, 0] = 1.
res[:, 1] = 0.5
conf = conf_metric(res, 0)
inc_conf = conf_metric(res, 1)

assert conf["correct"] is True
assert inc_conf["correct"] is False
assert conf["top_mag"] == inc_conf["top_mag"] == 1.0
assert conf["runnerup_dist"] == inc_conf["runnerup_dist"] == 0.5
