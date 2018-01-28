"""Modify the IA module to respond more to certain vectors"""
import numpy as np

import nengo_spa as spa
from nengo_spa.vocab import VocabularyOrDimParam

import nengo
from nengo.networks.ensemblearray import EnsembleArray

from nengo.exceptions import ValidationError
from nengo.utils.compat import is_iterable, is_string

from typing import List


class MegAssociativeMemory(spa.Network):
    input_vocab = VocabularyOrDimParam(
        'input_vocab', default=None, readonly=True)
    output_vocab = VocabularyOrDimParam(
        'output_vocab', default=None, readonly=True)

    def __init__(
            self, n_accum_neurons: List, n_thresholding_neurons: List, input_vocab, output_vocab=None, mapping=None,
            label=None, seed=None, add_to_container=None,
            vocabs=None, **selection_net_args):
        super(MegAssociativeMemory, self).__init__(
            label=label, seed=seed, add_to_container=add_to_container,
            vocabs=vocabs)

        if output_vocab is None:
            output_vocab = input_vocab
        elif mapping is None:
            raise ValidationError(
                "The mapping argument needs to be provided if an output "
                "vocabulary is given.", attr='mapping', obj=self)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        if mapping is None or mapping == 'by-key':
            mapping = {k: k for k in self.input_vocab.keys()}
        elif is_string(mapping):
            raise ValidationError(
                "The mapping argument must be a dictionary, the string "
                "'by-key' or None.", attr='mapping', obj=self)

        input_keys = mapping.keys()
        input_vectors = [input_vocab.parse(key).v for key in input_keys]
        output_keys = [mapping[k] for k in input_keys]
        output_vectors = [output_vocab.parse(key).v for key in output_keys]

        input_vectors = np.asarray(input_vectors)
        output_vectors = np.asarray(output_vectors)
        assert len(input_vectors) == len(n_accum_neurons)
        assert len(input_vectors) == len(n_thresholding_neurons)

        with self:
            self.selection = meg_IA(n_accum_neurons, n_thresholding_neurons,
                                    n_ensembles=len(input_vectors), label="selection", **selection_net_args)
            self.input = nengo.Node(size_in=self.input_vocab.dimensions,
                                    label="input")
            self.output = nengo.Node(size_in=self.output_vocab.dimensions,
                                     label="output")

            nengo.Connection(
                self.input, self.selection.input, transform=input_vectors)
            nengo.Connection(
                self.selection.output, self.output, transform=output_vectors.T)

        self.declare_input(self.input, self.input_vocab)
        self.declare_output(self.output, self.output_vocab)

        self.input_reset = self.selection.input_reset
        self.declare_input(self.input_reset, None)


def meg_IA(
        n_accum_neurons: List, n_thresholding_neurons: List, n_ensembles: int, accum_threshold=0.8,
        accum_timescale=0.2, feedback_timescale=0.005,
        accum_synapse=0.1, ff_synapse=0.005,
        intercept_width=0.15, radius=1., **kwargs) -> nengo.Network:

    bar_beta = 1. + radius * feedback_timescale / accum_timescale
    feedback_tr = (
        np.eye(n_ensembles) - bar_beta * (1. - np.eye(n_ensembles)) /
        feedback_timescale)

    with nengo.Network(**kwargs) as net:
        net.accumulators = Thresholding(
            n_accum_neurons, n_ensembles, threshold=0.,
            intercept_width=intercept_width, radius=radius)
        net.thresholding = Thresholding(
            n_thresholding_neurons, n_ensembles, threshold=accum_threshold,
            intercept_width=intercept_width, radius=radius,
            function=lambda x: x > accum_threshold)

        nengo.Connection(
            net.accumulators.output, net.accumulators.input,
            synapse=accum_synapse)
        nengo.Connection(
            net.accumulators.output, net.thresholding.input,
            synapse=ff_synapse)
        nengo.Connection(
            net.thresholding.output, net.accumulators.input,
            synapse=accum_synapse, transform=accum_synapse * feedback_tr)

        net.input_reset = nengo.Node(size_in=1)
        nengo.Connection(
            net.input_reset, net.accumulators.input, synapse=None,
            transform=-radius * np.ones((n_ensembles, 1)) / accum_synapse)

        net.input = nengo.Node(size_in=n_ensembles)
        nengo.Connection(net.input, net.accumulators.input, synapse=None,
                         transform=1. / accum_timescale)
        net.output = net.thresholding.output
    return net


def Thresholding(
        n_neurons: List, n_ensembles, threshold, intercept_width=0.15, function=None,
        radius=1., **kwargs):
    with nengo.Network(**kwargs) as net:
        with nengo.presets.ThresholdingEnsembles(
                0., intercept_width, radius=radius):
            net.thresholding = HeteroEnsembleArray(n_neurons, n_ensembles)

        net.bias = nengo.Node(1.)
        nengo.Connection(net.bias, net.thresholding.input,
                         transform=-threshold * np.ones((n_ensembles, 1)))

        net.input = net.thresholding.input
        net.thresholded = net.thresholding.output

        if function is None:
            function = lambda x: x
        function = lambda x, function=function: function(x + threshold)
        net.output = net.thresholding.add_output('function', function)
    return net


class HeteroEnsembleArray(EnsembleArray):

    def __init__(self, n_neurons: List, n_ensembles, ens_dimensions=1, label=None, seed=None,
                 add_to_container=None, **ens_kwargs):
        if "dimensions" in ens_kwargs:
            raise ValidationError(
                "'dimensions' is not a valid argument to EnsembleArray. "
                "To set the number of ensembles, use 'n_ensembles'. To set "
                "the number of dimensions per ensemble, use 'ens_dimensions'.",
                attr='dimensions', obj=self)

        super(EnsembleArray, self).__init__(label, seed, add_to_container)

        for param in ens_kwargs:
            if is_iterable(ens_kwargs[param]):
                ens_kwargs[param] = nengo.dists.Samples(ens_kwargs[param])

        self.config[nengo.Ensemble].update(ens_kwargs)

        label_prefix = "" if label is None else label + "_"

        assert len(n_neurons) == n_ensembles
        self.n_neurons_per_ensemble = n_neurons
        self.n_ensembles = n_ensembles
        self.dimensions_per_ensemble = ens_dimensions

        # These may be set in add_neuron_input and add_neuron_output
        self.neuron_input, self.neuron_output = None, None

        self.ea_ensembles = []

        with self:
            self.input = nengo.Node(size_in=self.dimensions, label="input")

            for e_i, n_n in enumerate(n_neurons):
                e = nengo.Ensemble(n_n, self.dimensions_per_ensemble,
                                   label="%s%d" % (label_prefix, e_i))
                nengo.Connection(self.input[e_i * ens_dimensions:
                                            (e_i + 1) * ens_dimensions],
                                 e, synapse=None)
                self.ea_ensembles.append(e)

        self.output = None
        self.add_output('output', function=None)
