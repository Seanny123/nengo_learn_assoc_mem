import nengo
import nengo_spa as spa
import numpy as np

from random import shuffle
import itertools

from typing import Sequence, List, Tuple

dt = 0.001


def gen_fan1_pairs(n_items: int):
    src = ['F1_%d' % i for i in range(2 * n_items)]

    fan1 = [(f1, f2) for f1, f2 in zip(src[::2], src[1::2])]
    half_way = len(src) // 2
    foil1 = [(f1, f2) for f1, f2 in zip(src[:half_way], src[half_way:])]

    return fan1, foil1


def gen_foil2_pairs(src, picked_pairs, n_items: int):

    indices = set(range(len(src)))
    foil_pairs = {n: set() for n in range(len(src))}
    foil2 = []

    for key, val in picked_pairs.items():

        foil_pick = indices - {key} - val - foil_pairs[key]
        for foil_idx in foil_pick:
            foil2.append((src[key], src[foil_idx]))

            if len(foil2) == n_items:
                return foil2

            foil_pairs[key].add(foil_idx)
            foil_pairs[foil_idx].add(key)

    return foil2


def gen_fan2_pairs(n_items: int):
    assert n_items > 3
    src = ['F2_%d' % i for i in range(n_items)]

    picked_pairs = {n: set() for n in range(len(src))}

    fan2 = []

    d_list = list(itertools.chain(*zip(range(n_items), range(n_items))))
    d_list.append(d_list.pop(0))
    for i1, i2 in zip(d_list[::2], d_list[1::2]):
        fan2.append((src[i1], src[i2]))
        picked_pairs[i1].add(i2)
        picked_pairs[i2].add(i1)

    assert len(fan2) == n_items

    foil2 = gen_foil2_pairs(src, picked_pairs, n_items)

    assert len(fan2) == len(foil2)

    return fan2, foil2


def vecs_from_list(vocab: spa.Vocabulary, spa_strs: List[Tuple[str, str]], norm=False) -> List[np.ndarray]:
    res = []
    for i1, i2 in spa_strs:
        vec_str = f"{i1}+{i2}"

        if norm:
            res.append(spa_parse_norm(vocab, vec_str))
        else:
            res.append(vocab.parse(vec_str).v)

    return res


def check_fan_foil(fan, foil):
    foil_items = set()

    for i1, i2 in foil:
        foil_items.add(i1)
        foil_items.add(i2)

    fan_items = set()

    for i1, i2 in fan:
        fan_items.add(i1)
        fan_items.add(i2)

    assert len(foil_items - fan_items) == 0


def make_alt_vocab(n_fan1_items: int, n_fan2_items: int, dimensions: int, seed, norm=False):
    rng = np.random.RandomState(seed=seed)

    fan1, foil1 = gen_fan1_pairs(n_fan1_items)
    check_fan_foil(fan1, foil1)
    fan2, foil2 = gen_fan2_pairs(n_fan2_items)
    check_fan_foil(fan2, foil2)

    pairs = fan1 + fan2

    vocab = spa.Vocabulary(dimensions, max_similarity=0.9, rng=rng)
    items = set()
    for i1, i2 in pairs:
        items.add(i1)
        items.add(i2)
    for item in items:
        vocab.populate(item)

    fan1_vecs = vecs_from_list(vocab, fan1, norm)
    foil1_vecs = vecs_from_list(vocab, foil1, norm)
    fan2_vecs = vecs_from_list(vocab, fan2, norm)
    foil2_vecs = vecs_from_list(vocab, foil2, norm)

    return vocab, fan1, fan1_vecs, fan2, fan2_vecs, foil1, foil1_vecs, foil2, foil2_vecs


def spa_parse_norm(vocab: spa.Vocabulary, spa_str: str):
    big_vec = vocab.parse(spa_str).v
    return big_vec / np.linalg.norm(big_vec)


def norm_spa_vecs(vocab: spa.Vocabulary, spa_strs: List[str]):
    res = []

    for spa_str in spa_strs:
        norm_vec = spa_parse_norm(vocab, spa_str)
        res.append(norm_vec)

    return res


def make_fan_vocab(seed, dimensions: int):
    rng = np.random.RandomState(seed=seed)
    vocab = spa.Vocabulary(dimensions, rng=rng, strict=False)

    fan1 = ["CAT+DOG", "DUCK+FISH", "HORSE+COW"]
    fan1_vecs = norm_spa_vecs(vocab, fan1)
    fan1_labels = ["F1%s" % i for i in range(len(fan1))]

    fan2 = ["PIG+RAT", "PIG+GOAT", "SHEEP+EMU", "SHEEP+GOOSE", "FROG+TOAD", "FROG+NEWT"]
    fan2_vecs = norm_spa_vecs(vocab, fan2)
    fan2_labels = ["F2%s" % i for i in range(len(fan1))]

    return vocab, fan1, fan1_vecs, fan1_labels, fan2, fan2_vecs, fan2_labels


def meg_from_spikes(spikes: np.ndarray, meg_syn=0.1):
    return nengo.Lowpass(meg_syn).filt(np.sum(spikes, axis=1))


def gen_feed_func(vocab, vocab_items, t_present: float):

    def f(t):
        index = int(t / t_present)
        index = index % len(vocab_items)
        return vocab.parse(vocab_items[index]).v

    return f


class BasicVecFeed(object):

    def __init__(self, dataset: Sequence, correct: Sequence, t_len: float, dims: int, n_items: int, pause: float):
        self.data_index = 0
        self.paused = False

        self.time = 0.0
        self.sig_time = 0

        self.pause_time = pause
        self.q_duration = t_len
        self.ans_duration = self.q_duration + self.pause_time

        self.correct = correct
        self.qs = dataset
        self.num_items = n_items
        self.dims = dims
        self.indices = list(range(self.num_items))

    def get_answer(self, t):
        """Signal for correct answer"""
        if self.pause_time < self.time < self.ans_duration:
            return self.correct[self.indices[self.data_index]]
        else:
            return np.zeros(self.dims)

    def feed(self, t):
        """Feed the question into the network
        this is the main state machine of the network"""
        self.time += dt

        if self.time > self.pause_time and self.sig_time > self.q_duration:

            if self.data_index < self.num_items - 1:
                self.data_index += 1
            else:
                shuffle(self.indices)
                self.data_index = 0

            self.time = 0.0
            self.sig_time = 0.0

        elif self.time > self.pause_time:
            self.paused = False

            q_idx = self.indices[self.data_index]
            return_val = self.qs[q_idx]
            self.sig_time += dt
            return return_val

        else:
            self.paused = True

        return np.zeros(self.dims)
