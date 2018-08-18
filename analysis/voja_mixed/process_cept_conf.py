import os
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd

from nengo_learn_assoc_mem.paths import data_path


def get_rt(integ_res, thresh: float) -> np.array:
    react_tms = []

    for i_res in integ_res:
        whr = np.where(i_res > thresh)
        if len(whr[0]) == 0:
            react_tms.append(-1)
        else:
            react_tms.append(whr[0][0])

    return np.array(react_tms)


file_names = ("low_cept_enc",
              "low2_cept_enc")

for fn in file_names:
    read_path = os.path.join(data_path, "static_react", f"{fn}_confidence.h5")

    stim_types = ("fan1", "fan2", "foil1", "foil2")

    total_err_rates = OrderedDict([(s_t, []) for s_t in stim_types])
    total_rt = OrderedDict([(s_t, []) for s_t in stim_types])
    n_vecs = 16
    train_len = n_vecs * 2

    for seed_val in range(10):
        with h5py.File(read_path, "r") as fi:
            comp_res = np.array(fi[f"comp_res_{seed_val}"]).squeeze()
            neg_integ_res = np.array(fi[f"integ_res_neg_{seed_val}"]).squeeze()
            pos_integ_res = np.array(fi[f"integ_res_pos_{seed_val}"]).squeeze()

        pos_thresh = np.mean(pos_integ_res[:train_len], axis=0)[-1] * 0.65
        neg_thresh = np.mean(neg_integ_res[:train_len], axis=0)[-1] * 1.6

        pos_rt = get_rt(pos_integ_res, pos_thresh)
        max_pos_rt = np.max(pos_rt)
        neg_rt = get_rt(neg_integ_res, neg_thresh)
        max_neg_rt = np.max(neg_rt)

        nans_idx = (pos_rt == -1) & (neg_rt == -1)
        if np.any(nans_idx):
            nans_loc = np.where(nans_idx == True)[0]
            print(f"No answer detected {len(nans_loc)} times for seed {seed_val}")

            pos_nans_loc = nans_loc[nans_loc < train_len]
            if len(pos_nans_loc) > 0:
                neg_rt[pos_nans_loc] = max_neg_rt

            neg_nans_loc = nans_loc[nans_loc > train_len]
            if len(neg_nans_loc) > 0:
                pos_rt[pos_nans_loc] = max_pos_rt

        all_max = np.max([max_pos_rt, max_neg_rt]) + 10
        pos_rt[pos_rt < 0] = all_max
        neg_rt[neg_rt < 0] = all_max
        all_rt = np.array([pos_rt, neg_rt]).max(axis=0)

        cor_targ_resp = np.zeros(train_len, dtype=np.bool)
        cor_targ_resp[pos_rt[:train_len] < neg_rt[:train_len]] = True

        cor_foil_resp = np.zeros(train_len, dtype=np.bool)
        cor_foil_resp[pos_rt[train_len:] > neg_rt[train_len:]] = True

        total_err_rates["fan1"].append(1 - (np.sum(cor_targ_resp[:16]) / 16))
        total_err_rates["fan2"].append(1 - (np.sum(cor_targ_resp[16:]) / 16))
        total_err_rates["foil1"].append(1 - (np.sum(cor_foil_resp[:16]) / 16))
        total_err_rates["foil2"].append(1 - (np.sum(cor_foil_resp[16:]) / 16))

        for s_i, stim_li in enumerate(total_rt.values()):
            rt_slc = slice(s_i * n_vecs, (s_i + 1) * n_vecs)
            stim_li.append(np.mean(all_rt[rt_slc]))

    df_rt = pd.DataFrame(total_rt)
    df_rt.to_hdf(os.path.join(data_path, "react_err_dfs", f"{fn}_rt.h5"), key="react", mode="w")

    df_err = pd.DataFrame(total_err_rates)
    df_err.to_hdf(os.path.join(data_path, "react_err_dfs", f"{fn}_err.h5"), key="error", mode="w")
