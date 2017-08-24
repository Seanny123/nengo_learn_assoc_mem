import nengo


class NegBCM(nengo.BCM):
    learning_rate = nengo.params.NumberParam('learning_rate')
