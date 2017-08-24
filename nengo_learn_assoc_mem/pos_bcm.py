import nengo


class PosBCM(nengo.BCM):
    learning_rate = nengo.params.NumberParam('learning_rate')
