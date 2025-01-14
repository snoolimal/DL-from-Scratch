"""
loss.py
"""

from typing import Optional, List
from config import np
from common.functions import sigmoid, softmax, cross_entropy_error
from common.layers.embedding import EmbeddingDot
from utils.nlp_util import UnigramSampler


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t

        self.y = softmax(x)  # [N,C]
        loss = cross_entropy_error(self.y, self.t)

        return loss

    def backward(self, dy=1):
        # one-hot target vector라면 class idx vector로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        batch_size = self.t.shape[0]

        _y = self.y.copy()
        _y[np.arange(batch_size), self.t] -= 1
        dloc = _y / batch_size

        dx = dloc * dy

        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t                      # [N,]
        self.y = sigmoid(x)             # [N,]

        _y = np.c_[1 - self.y, self.y]  # binary의 sigmoid 출력은 1(정답)일 확률 (concat arrays along col) | [N,2]
        loss = cross_entropy_error(_y, self.t)

        return loss

    def backward(self, dy=1):
        batch_size = self.t.shape[0]

        dloc = (self.y - self.t) / batch_size
        dx = dloc * dy

        return dx


class NegativeSamplingLoss:
    def __init__(self, w, tokenized_corpus, power=0.75, sample_size=5):
        """
        cf. fig 4-17 (p.170(171))
        ---
        Args:
            w: w_out | [V,H]
                V: vocab size (=input size)
                H: hidden size
            tokenized_corpus: [NN,]
        """
        self.sample_size = sample_size
        self.sampler = UnigramSampler(tokenized_corpus, sample_size, power)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]      # neg spl size + target 1
        self.embed_dot_layers = [EmbeddingDot(w) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for embed_dot_layer in self.embed_dot_layers:   # loss layer에는 param이 없음
            self.params += embed_dot_layer.params
            self.grads += embed_dot_layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.negative_sample(target)

        # positive label fpass
        positive_label = np.ones(batch_size, dtype=np.int32)  # correct label (정답, 1)
        score = self.embed_dot_layers[0].forward(h, indices=target)
        loss = self.loss_layers[0].forward(x=score, t=positive_label)

        # negative label fpass
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):                                           # N개 data 각각에 대한
            negative_i = negative_sample[:, i]                                      # ith negative target의 | [N,]
            score = self.embed_dot_layers[1 + i].forward(h, indices=negative_i)     # score 평균
            loss += self.loss_layers[1 + i].forward(x=score, t=negative_label)      # loss 평균(을 전체 loss에 누적)

        return loss

    def backward(self, dy=1):
        """
        single data(target)에 대해 h와 각 target(pos or neg)의 fpass는
                             label (0 or 1) ->
                 h -> EmbeddingDot -> score -> SigmoidWithLoss -> loss
            target ->
        이므로 bpass는
                                       dlabel <-
                 dh <- EmbeddingDot <- dscore <- SigmoidWithLoss <- dy=1
            dtarget <-
        이다.
        ---
        Batch로 확장하면
            loss의 누적은 + node이므로 dy를 그대로 흘린다.
            EmbeddingDot node의 입력 w와 h는 공유되므로 gradient를 누적합:
                dw의 누적합은 이미 Embedding class의 backward()에 구현해 두었다.
                dh는 여기서!
            해야 한다.
        """
        dh = 0
        for loss_layer, embed_dot_layer in zip(self.loss_layers, self.embed_dot_layers):    # reversed() 먹일 필요 X
            dscore = loss_layer.backward(dy)
            dh += embed_dot_layer.backward(dscore)

        return dh


class TimeSoftmaxWithLoss:
    """
    TimeRNN에서 Time류 node를 만든 것과 마찬가지로
    SoftmaxWithLoss node를 T개 준비하여 (timestep 순서대로) 각 chunk를 처리하면 된다.
    """
    def __init__(self):
        self.params, self.grads = [], []

        self.ls = None
        self.layers: Optional[List] = None

    def forward(self, ls, ts):
        """
        Args:
            ls: [N,T,V]
                chunk l_t: ls[:, t, :] | [N,V]
            ts: [N,T,V]
                One-hot form이므로 axis 2는 V이다.
        ---
        Returns:
            loss: loss_t T개의 평균 | scalar
                loss_t: scalar
                    Chunk l_t를 구성하는 N개의 (timestep t의) single vector 각각에 대해
                    (one-hot form의 target vector로) 구한 softmax loss N개의 평균이다.
                    cf. eq 5.11 (p.225(6))
                        책은 batch size=1로 두고 설명하였으므로 loss_t가 L_t에 해당한다.
        """
        N, T, V = ls.shape

        self.ls = ls

        loss = 0
        for t in range(T):
            layer = SoftmaxWithLoss()
            loss_t = layer.forward(ls[:, t, :], ts[:, t])
            loss += loss_t
            self.layers.append(layer)
        loss /= T

        return loss

    def backward(self, dy=1):
        """
        Args:
            dy: 1 matrix | [N,T,V]
        ---
        Returns:
            dls: [N,T,V]
        """
        N, T, V = self.ls.shape

        dls = np.empty((N, T, V), dtype='f')
        dy = np.ones((N, T, V), dtype='f')
        dy *= (1 / T)
        for t in range(T):
            layer = self.layers[t]
            dls[:, t, :] = layer.backward(dy[:, t, :])
            """reversed(range(T))
            Affine node의 적용부터는, 고로 이어지는 SoftmaxWithLoss node의 적용에는
            time dependency가 사라진 일반적인(병럴적) copy gate를 통과한 w, h가 사용된다.
            따라서 reversed()는 불필요하다.
            """

        return dls


class SmartTimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, ls, ts):
        """
        Args:
            ls: [N,T,V]
                chunk l_t: ls[:, t, :] | [N,V]
            ts: [N,T,V] -> [N,T]
                One-hot form이므로 axis 2는 V이다. Class idx form으로 변환해 사용한다.
        ---
        Returns:
            loss: scalar
        """
        N, T, V = ls.shape

        # target ts가 one-hot form이라면 idx form으로 변환
        if ts.ndim == 3:
            ts = ts.argmax(axis=2)          # [N,T]

        mask = (ts != self.ignore_label)    # [N,T]

        # batch 속 sequence의 모든 single vector [V,] 단위로 한번에 다룸
        ls = ls.reshape(N*T, V)                 # [N*T,V]
        ts = ts.reshape(N*T)                    # [N*T,]
        mask = mask.reshape(N*T)                # [N*T,]

        _ys = softmax(ls)                       # [N*T,V]
        ys = np.log(_ys[np.arange(N*T), ts])    # [N*T,V], 각 single vector의 정답 idx만 log 먹여 ys로 (오답 indices는 +0)
        ys *= mask                              # ignore_label에 해당하는 input vector의 loss는 0으로 (-1 indexing dumping도)
        loss = np.negative(np.sum(ys))          # scalar
        loss /= mask.sum()                      # get avg

        self.cache = (ts, _ys, mask, (N, T, V))

        return loss

    def backward(self, dy=1):
        """
        Args:
            dy: dL = 1 (scalar)
        ---
        Returns:
            dls: [N,T,V]
        """
        ts, _ys, mask, (N, T, V) = self.cache

        _ys[np.arange(N*T), ts] -= 1    # [N*T,V], 각 single vector의 정답 idx만 -1
        dloc = _ys / mask.sum()         # [N*T,V]

        dls = dloc * dy                 # [N*T,V]
        dls *= mask[:, np.newaxis]      # [N*T,V], ignore_label에 해당하는 input vector의 gradient는 0으로

        dls = dls.reshape(N, T, V)

        return dls
