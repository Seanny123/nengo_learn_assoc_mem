import numpy as np

import re

loss_re = re.compile(r"(?<=]:) [0-9.]+")
param_re = re.compile(r"^[[\d.,\s]+]")

with open("data/param_explore_results.log") as fi:
    lines = fi.readlines()

params = []
losses = []

for li in lines:
    res = loss_re.search(li)
    if res is not None:
        losses.append(float(res.group(0)))
        params.append(param_re.search(li).group(0))

min_idx = np.argmin(losses)
print(f"Minimum {params[min_idx]}: {losses[min_idx]}")
