import numpy as np

from nengo_learn_assoc_mem.utils import make_alt_vocab

n_items = 5
seed = 8
dimensions = 32

vocab_1, fan1_1, fan1_pair_vecs_1, fan2_1, fan2_pair_vecs_1,\
    foil1_1, foil1_pair_vecs_1, foil2_1, foil2_pair_vecs_1 = make_alt_vocab(n_items, n_items,
                                                                            dimensions, seed, norm=True)


vocab_2, fan1_2, fan1_pair_vecs_2, fan2_2, fan2_pair_vecs_2,\
    foil1_2, foil1_pair_vecs_2, foil2_2, foil2_pair_vecs_2 = make_alt_vocab(n_items, n_items,
                                                                            dimensions, seed, norm=True)

for nm, pnt in vocab_1.items():
    assert np.allclose(pnt.v, vocab_2[nm].v) 
