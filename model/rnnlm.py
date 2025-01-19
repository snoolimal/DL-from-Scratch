import pickle
from common.base import Model
from common.layers.time import TimeEmbedding, TimeAffine
from common.layers.timegate import *
from common.layers.criterion import TimeSoftmaxWithLoss


class Rnnlm(Model):
    """
    Computation Graph
     raw_xs[N,T] -> Embedding Block -> xs[N,T,D] -> LSTM Block -> hs[N,T,H] -> Affine Block -> ys[N,T,V]
    w_embed[V,D] ->                     wx[D,4H] ->                 wa[H,V] ->
                                        wh[H,4H] ->                  ba[V,] ->
                                          b[4H,] ->

    ys[N,T,V] -> SoftmaxWithLoss Block -> L
    ts[N,T,V] ->
    ---
    cf1. T는 전체 sequence x의 길이이거나 truncation BPTT를 수행한다면
         chunk가 커버하는 tsteps의 수 -- i.e., LSTM block 속 LSTM cell의 수 -- 겠지.
    cf2. ts는 one-hot form일 때 [N,T,V]이고 index form일 때 [N,T]겠지.
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """
        Args:
            vocab_size V: 단어의 개수
                = num_classes C
            wordvec_size D: 입력 vector x의 embedding 차원
            hidden_size H: LSTM cell의 hidden state h의 차원
        """
        super().__init__()

        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # weight initialization (Xavier)
        w_embed = (rn(V, D) / 100).astype('f')
        wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        b = np.zeros(4 * H).astype('f')
        wa = (rn(H, V) / np.sqrt(H)).astype('f')
        ba = np.zeros(V).astype('f')

        # model architecture
        self.layers = [
            TimeEmbedding(w_embed),
            TimeLSTM(wx, wh, b, stateful=True),
            TimeAffine(wa, ba)
        ]
        self.criterion = TimeSoftmaxWithLoss()

        # computation graph
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.lstm_layer = self.layers[1]

    def predict(self, xs):
        """
        Args:
            xs: raw_xs  | [N,T]
        ---
        Returns:
            xs: ys      | [N,T,V]
        """
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        """
        Args:
            xs: raw_xs  | [N,T]
            ts: targets | [N,T,V] or [N,T]
        ---
        Returns:
            loss: L     | scalar
        """
        ys = self.predict(xs)
        loss = self.criterion.forward(ys, ts)
        return loss

    def backward(self, dy=1):
        dy = self.criterion.backward(dy)
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def reset_state(self):
        self.lstm_layer.reset_state()
