from nengo_learn_assoc_mem.utils import make_alt_vocab

dimensions = 32
seed = 8
num_fan1 = 5
num_fan2 = 5

vocab, fan1, fan1_vecs, fan2, fan2_vecs, foil1, foil1_vecs, foil2, foil2_vecs = make_alt_vocab(num_fan1, num_fan2, dimensions, seed, norm=True)

assert len(fan1) == num_fan1
assert len(fan2) == num_fan2

dimensions = 64

vocab, fan1, fan1_vecs, fan2, fan2_vecs, foil1, foil1_vecs, foil2, foil2_vecs = make_alt_vocab(num_fan1, num_fan2, dimensions, seed, norm=True)

assert len(fan1) == num_fan1
assert len(fan2) == num_fan2

num_fan1 = 11
num_fan2 = 11

vocab, fan1, fan1_vecs, fan2, fan2_vecs, foil1, foil1_vecs, foil2, foil2_vecs = make_alt_vocab(11, 11, dimensions, seed, norm=True)

assert len(fan1) == num_fan1
assert len(fan2) == num_fan2
