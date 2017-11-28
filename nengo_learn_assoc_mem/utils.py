import nengo
from nengo import spa
import numpy as np

from random import shuffle

from typing import Sequence

dt = 0.001


def make_fan_vocab(seed, dimensions: int):
    rng = np.random.RandomState(seed=seed)
    vocab = spa.Vocabulary(dimensions, rng=rng)

    fan1 = ["CAT+DOG", "DUCK+FISH", "HORSE+COW"]
    fan1_vecs = []
    fan1_labels = ["F11", "F12", "F13"]

    fan2 = ["PIG+RAT", "PIG+GOAT", "SHEEP+EMU", "SHEEP+GOOSE"]
    fan2_vecs = []
    fan2_labels = ["F21", "F22", "F23", "F24"]

    for vec in fan1:
        sum_vec = vocab.parse(vec).v
        norm_vec = sum_vec / np.linalg.norm(sum_vec)
        fan1_vecs.append(norm_vec)

    for vec in fan2:
        sum_vec = vocab.parse(vec).v
        norm_vec = sum_vec / np.linalg.norm(sum_vec)
        fan2_vecs.append(norm_vec)

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
