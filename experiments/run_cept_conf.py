from collections import OrderedDict
import os

import h5py
import numpy as np
import nengo_spa as spa

from experiments.conf_utils import train_decoders, run_comp, run_integ
from nengo_learn_assoc_mem.utils import numpy_bytes_to_str, norm_spa_vecs
from nengo_learn_assoc_mem.paths import data_path

stim_types = ("fan1", "fan2", "foil1", "foil2")

cept_types = ("low2_cept_enc",
              "low_cept_enc")

for c_t in cept_types:
    save_path = os.path.join(data_path, "static_react", f"{c_t}_confidence.h5")

    for seed_val in range(10):
        read_path = os.path.join(data_path, "mixed_static", f"static_{c_t}_{seed_val}.h5")
        stim_strs = []

        with h5py.File(read_path, "r") as fi:

            for s_t in stim_types:
                stim_strs.append((s_t, numpy_bytes_to_str(fi[s_t])))

            v_strs = numpy_bytes_to_str(fi['vocab_strings'])
            v_vecs = list(fi['vocab_vectors'])
            dimensions = fi['vocab_vectors'].attrs['dimensions']

            encoders = np.array(fi['encoders'])
            n_neurons = encoders.shape[0]
            intercepts = np.full(n_neurons, fi['encoders'].attrs['intercept'])
            init_seed = fi['encoders'].attrs['seed']

            tm = fi["t_range"]
            dt_sim = tm.attrs["dt"]
            t_pause = tm.attrs["t_pause"]
            t_present = tm.attrs["t_present"]

        vocab = spa.Vocabulary(dimensions)
        for val, vec in zip(v_strs, v_vecs):
            vocab.add(val, vec)

        stim_vecs = OrderedDict((
                        ("fan1", None),
                        ("fan2", None),
                        ("foil1", None),
                        ("foil2", None)))
        for (nm, strs) in stim_strs:
            stim_vecs[nm] = norm_spa_vecs(vocab, strs)

        train_vecs = stim_vecs["fan1"] + stim_vecs["fan2"]
        decs = train_decoders(train_vecs, n_neurons, dimensions,
                              encoders, intercepts, init_seed, t_present, t_pause)
        feed_vecs = []
        for vecs in stim_vecs.values():
            feed_vecs += vecs

        print("simulating comparisons")
        all_comp_res = []
        final_vals = []
        for f_i, f_vec in enumerate(feed_vecs):
            comp_res = run_comp(f_vec, decs, n_neurons, dimensions,
                                encoders, intercepts, init_seed, 0.2)
            final_vals.append(comp_res[-1])
            all_comp_res.append(comp_res)

        train_len = len(train_vecs)
        max_targ_comp = np.max(final_vals[:train_len])
        min_targ_comp = np.min(final_vals[:train_len])
        max_foil_comp = np.max(final_vals[train_len:])
        min_foil_comp = np.min(final_vals[train_len:])

        with h5py.File(save_path, "a") as w_fi:
            w_fi.create_dataset(f"comp_res_{seed_val}", data=all_comp_res)

        neg_integ_res = []
        pos_integ_res = []
        print("simulating integration")
        for c_i, c_res in enumerate(all_comp_res):
            integ_res = run_integ(list(c_res)[10:125], init_seed, pos_adjust=min_foil_comp - .05)
            neg_integ_res.append(integ_res["neg"])
            pos_integ_res.append(integ_res["pos"])

        print(f"Done seed {seed_val}")
        # save output confidence for each input vector
        with h5py.File(save_path, "a") as w_fi:
            w_fi.create_dataset(f"integ_res_pos_{seed_val}",
                                data=pos_integ_res)
            w_fi.create_dataset(f"integ_res_neg_{seed_val}",
                                data=neg_integ_res)
