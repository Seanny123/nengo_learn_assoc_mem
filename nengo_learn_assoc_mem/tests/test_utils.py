from nengo_learn_assoc_mem.utils import make_alt_vocab

dimensions = 32
seed = 8

vocab, fan1, fan1_vecs, fan2, fan2_vecs, foil1, foil1_vecs, foil2, foil2_vecs = make_alt_vocab(5, 5, dimensions, seed, norm=True)
