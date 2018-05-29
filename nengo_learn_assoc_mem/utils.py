import nengo
import nengolib
from nengo.utils.ensemble import tuning_curves
import nengo_spa as spa
import numpy as np

from random import shuffle
import itertools

from typing import Sequence, List, Tuple, Dict

dt = 0.001


def get_activities(vecs: np.ndarray, n_neurons: int, dimensions: int,
                   enc: np.ndarray, intercepts: np.ndarray, seed: int):
    with nengolib.Network(seed=seed) as model:
        ens = nengo.Ensemble(n_neurons, dimensions, encoders=enc, intercepts=intercepts)

    with nengo.Simulator(model) as sim:
        pass

    _, activities = tuning_curves(ens, sim, inputs=vecs)

    return activities


def choose_encoders(n_neurons: int, dimensions: int, encoder_proportion: float, mean_fan1, mean_fan2):
    encoders = np.zeros((n_neurons, dimensions))

    fan1_end = int(n_neurons * encoder_proportion)

    for n_i in range(fan1_end):
        encoders[n_i] = mean_fan1 + np.random.normal(size=dimensions) * 0.1

    for n_i in range(fan1_end, n_neurons):
        encoders[n_i] = mean_fan2 + np.random.normal(size=dimensions) * 0.1

    return encoders


def numpy_bytes_to_str(lst):
    return [l.decode('utf-8') for l in lst]


def list_as_ascii(li: List[str]):
    return [l.encode("ascii", "ignore") for l in li]


def gen_added_strings(pairs):
    return [f"{p1}+{p2}" for p1, p2 in pairs]


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


def check_fan_foil(fan: List[Tuple[str, str]], foil: List[Tuple[str, str]]):
    """Verify no pairs found in the fans are found in the foils"""
    foil_items = set()

    for i1, i2 in foil:
        foil_items.add(i1)
        foil_items.add(i2)

    fan_items = set()

    for i1, i2 in fan:
        fan_items.add(i1)
        fan_items.add(i2)

    assert len(foil_items - fan_items) == 0


def gen_vocab(dimensions: int, pairs, seed: float) -> spa.Vocabulary:
    rng = np.random.RandomState(seed=seed)
    vocab = spa.Vocabulary(dimensions, max_similarity=0.9, rng=rng)

    items = set()
    for i1, i2 in pairs:
        items.add(i1)
        items.add(i2)
    for item in items:
        vocab.populate(item)

    return vocab


def make_alt_vocab(n_fan1_items: int, n_fan2_items: int, dimensions: int, seed: float, norm=False):

    fan1, foil1 = gen_fan1_pairs(n_fan1_items)
    check_fan_foil(fan1, foil1)
    fan2, foil2 = gen_fan2_pairs(n_fan2_items)
    check_fan_foil(fan2, foil2)

    pairs = fan1 + fan2

    vocab = gen_vocab(dimensions, pairs, seed)

    fan1_vecs = vecs_from_list(vocab, fan1, norm)
    foil1_vecs = vecs_from_list(vocab, foil1, norm)
    fan2_vecs = vecs_from_list(vocab, fan2, norm)
    foil2_vecs = vecs_from_list(vocab, foil2, norm)

    return vocab, fan1, fan1_vecs, fan2, fan2_vecs, foil1, foil1_vecs, foil2, foil2_vecs


def spa_parse_norm(vocab: spa.Vocabulary, spa_str: str) -> np.ndarray:
    big_vec = vocab.parse(spa_str).v
    return big_vec / np.linalg.norm(big_vec)


def norm_spa_vecs(vocab: spa.Vocabulary, spa_strs: List[str]) -> List[np.ndarray]:
    res = []

    for spa_str in spa_strs:
        norm_vec = spa_parse_norm(vocab, spa_str)
        res.append(norm_vec)

    return res


def meg_from_spikes(spikes: np.ndarray, meg_syn=0.1):
    return nengo.Lowpass(meg_syn).filt(np.sum(spikes, axis=1))


def conf_metric(ss_data: np.ndarray, expected: int) -> Dict:
    correct = False

    smoothed = np.mean(ss_data, axis=0)
    winner = np.argmax(smoothed)
    winner_mag = smoothed[winner]
    mask = np.ones(smoothed.shape[0], dtype=bool)
    mask[winner] = False
    runnerup = np.argmax(smoothed[mask])
    runnerup_dist = (smoothed[winner] - smoothed[mask][runnerup]) / winner_mag

    if winner == expected:
        correct = True

    return dict(correct=correct, top_mag=smoothed[winner], runnerup_dist=runnerup_dist)


def ans_conf(ans: np.ndarray, cor: np.ndarray, num_items: int, td_item: int) -> np.ndarray:
    individ_ans_conf = np.sum(ans*cor, axis=1).reshape((td_item, num_items, -1), order='F')
    conf = np.max(np.sum(individ_ans_conf, axis=0), axis=1)
    return conf


def cycle_array(x, period, dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period / dt))
    if i_every != period / dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))

    def f(t):
        i = int(round((t - dt) / dt))  # t starts at dt
        return x[int(i / i_every) % len(x)]

    return f


def gen_feed_func(vocab, vocab_items: List[str], t_present: float):

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


class VecToScalarFeed(BasicVecFeed):

    def __init__(self, dataset: Sequence, correct: Sequence, t_len: float, pause: float):
        dims = len(dataset[0])
        n_items = len(dataset)
        super(VecToScalarFeed, self).__init__(dataset, correct, t_len, dims, n_items, pause)

    def get_answer(self, t):
        if self.pause_time < self.time < self.ans_duration:
            return self.correct[self.indices[self.data_index]]
        else:
            return 0


class LearnDelayVecFeed(BasicVecFeed):

    def __init__(self, dataset: Sequence, correct: Sequence, t_len: float, dims: int, n_items: int,
                 pause: float, t_delay: float):
        self.t_delay = t_delay
        super(LearnDelayVecFeed, self).__init__(dataset, correct, t_len, dims, n_items, pause)

    def get_learn(self, t):
        if self.pause_time + self.t_delay < self.time < self.ans_duration:
            return 1
        else:
            return 0
