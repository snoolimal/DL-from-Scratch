from common.base import Model
from common.layers.time import *
from common.layers import TimeSoftmaxWithLoss


class SimpelRnnlm(Model):
    """
     raw_xs[N,T] -> TimeEmbedding -> xs[N,T,D] -> TimeRNN -> hs[N,T,H] -> TimeAffine -> ls[N,T,V]
    w_embed[V,D] ->                    wx[D,H] ->           w_aff[H,V] ->
                                       wh[H,H]              b_aff[V,]
                                       b[H,]

    ls[N,T,V] -> TimeSoftmaxWithLoss -> L
    ts[N,T,V] ->
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """
        Args:
            vocab_size V: 단어의 개수, num_classes
            wordvec_size D: input vector x의 dimension
            hidden_size H: RNN node(block)의 hidden state h의 dimension
        """
        super().__init__()

        V, D, H = vocab_size, wordvec_size, hidden_size

        # weight initialization (Xavier)
        w_embed = (np.random.randn(V, D) / 100).astype('f')     # linear transformation of X ~ N
        wx = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        wh = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        b = np.zeros(H, dtype=np.float32)
        w_aff = (np.random.randn(H, V) / np.sqrt(H)).astype('f')
        b_aff = np.zeros(V, dtype=np.float32)

        # model architecture
        self.layers = [
            TimeEmbedding(w_embed),
            TimeRNN(wx, wh, b, stateful=True, plot_grad=True),  # stateful=True for truncated BPTT
            TimeAffine(w_aff, b_aff)
        ]
        self.criterion = TimeSoftmaxWithLoss()

        # computation graph
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        # for reset hidden state(stateful), plot_grad
        self.plot_grad_layer = self.rnn_layer = self.layers[1]

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.criterion.forward(xs, ts)

        return loss

    def backward(self, dy=1):
        dy = self.criterion.backward(dy)
        for layer in reversed(self.layers):
            dy = layer.backward(dy)

        return dy

    def reset_state(self):
        self.rnn_layer.reset_state()
