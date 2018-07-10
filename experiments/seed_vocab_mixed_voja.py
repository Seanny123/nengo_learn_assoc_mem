"""Show variability caused by seed"""
import os

import numpy as np

from experiments.seed_vocab_utils import get_ens_params, train_mixed_encoders, test_response, Constants
from nengo_learn_assoc_mem.utils import make_alt_vocab
from nengo_learn_assoc_mem.paths import data_path
from nengo_learn_assoc_mem.learning_rules.mixed_voja import MeanMixed


n_items = 16
dimensions = 32
n_neurons = 2000

dt = 0.001

t_pause = 0.1
t_present = 0.3

consts = Constants(n_neurons, dimensions, n_items, t_pause, t_present)

init_seed = 8
intercept = 0.2
intercepts = np.ones(n_neurons) * intercept

rule_kwargs = {"learning_rate": -4, "bias": .5, "max_dist": 1.5}
act_synapse = .005
learning_args = {"intercept": intercept, "act_synapse": act_synapse, **rule_kwargs}

base_path = os.path.join(data_path, "mixed_mean")

past_encs = np.zeros((n_neurons, dimensions))
start_encs, max_rates = get_ens_params(consts, intercepts, init_seed)

for seed_val in range(10):
    vocab, fan1, fan1_pair_vecs, fan2, fan2_pair_vecs, \
        foil1, foil1_pair_vecs, foil2, foil2_pair_vecs = make_alt_vocab(n_items, n_items, dimensions,
                                                                        seed_val, norm=True)
    vocab_strs = {"fan1": fan1, "fan2": fan2, "foil1": foil1, "foil2": foil2}

    learned_encs = train_mixed_encoders(consts, MeanMixed, fan1_pair_vecs + fan2_pair_vecs,
                                        start_encs, max_rates, intercepts,
                                        rule_kwargs,
                                        act_synapse, init_seed)

    # check learning is happening and learned encoders are different with each seed
    assert not np.allclose(start_encs, learned_encs)
    assert not np.allclose(learned_encs, past_encs)
    past_encs = learned_encs.copy()

    save_path = os.path.join(base_path, f"mean_voja_with_dist_{seed_val}.h5")
    test_response(consts, fan1_pair_vecs + fan2_pair_vecs + foil1_pair_vecs + foil2_pair_vecs,
                  vocab, vocab_strs,
                  learned_encs[-1].copy(), intercepts, max_rates,
                  save_path, learning_args, init_seed, False)
