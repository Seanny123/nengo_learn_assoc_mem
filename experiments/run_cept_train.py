"""Show variability caused by seed"""
import os

import numpy as np
import h5py

from experiments.seed_vocab_utils import get_ens_params, train_mixed_encoders, test_response, Constants
from nengo_learn_assoc_mem.utils import make_alt_vocab
from nengo_learn_assoc_mem.paths import data_path
from nengo_learn_assoc_mem.learning_rules.mixed_voja import StaticMixed

n_items = 16
dimensions = 32
n_neurons = 500

dt = 0.001

t_pause = 0.1
t_present = 0.3
past_encs = np.zeros((n_neurons, dimensions))

consts = Constants(n_neurons, dimensions, n_items, t_pause, t_present)

init_seed = 8
root_file_name = "solid_static"
load_path = os.path.join(data_path, "neg_voja_enc")
file_list = [
    "low_cept_enc",
    "low2_cept_enc"
]

for fi_nm in file_list:
    with h5py.File(os.path.join(load_path, f"{root_file_name}_{fi_nm}.h5"), "r") as fi:
        intercepts = np.array(fi["encoders"].attrs["intercept"])
        intercept = intercepts[0]
        thresh = fi["encoders"].attrs["thresh"]
        max_dist = fi["encoders"].attrs["max_dist"]
        learning_rate = fi["encoders"].attrs["learning_rate"]
        act_synapse = fi["encoders"].attrs["act_synapse"]

    rule_kwargs = {"learning_rate": learning_rate,
                   "thresh": thresh,
                   "max_dist": max_dist}
    learning_args = {"intercept": intercept,
                     "act_synapse": act_synapse,
                     **rule_kwargs}

    base_path = os.path.join(data_path, "mixed_static")

    start_encs, max_rates = get_ens_params(consts, intercepts, init_seed)

    for seed_val in range(10):
        vocab, fan1, fan1_pair_vecs, fan2, fan2_pair_vecs, \
            foil1, foil1_pair_vecs, foil2, foil2_pair_vecs = make_alt_vocab(n_items, n_items, dimensions,
                                                                            seed_val, norm=True)
        vocab_strs = {"fan1": fan1, "fan2": fan2, "foil1": foil1, "foil2": foil2}

        learned_encs = train_mixed_encoders(consts, StaticMixed, fan1_pair_vecs + fan2_pair_vecs,
                                            start_encs, max_rates, intercepts,
                                            rule_kwargs,
                                            act_synapse, init_seed)

        # check learning is happening and learned encoders are different with each seed
        assert not np.allclose(start_encs, learned_encs)
        assert not np.allclose(learned_encs, past_encs)
        past_encs = learned_encs.copy()

        save_path = os.path.join(base_path, f"static_{fi_nm}_{seed_val}.h5")
        test_response(consts, fan1_pair_vecs + fan2_pair_vecs + foil1_pair_vecs + foil2_pair_vecs,
                      vocab, vocab_strs,
                      learned_encs[-1].copy(), intercepts, max_rates,
                      save_path, learning_args, init_seed, False)
