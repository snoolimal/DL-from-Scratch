"""
loss.py
"""

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
        for loss_layer, embed_dot_layer in zip(self.loss_layers, self.embed_dot_layers):
            dscore = loss_layer.backward(dy)
            dh += embed_dot_layer.backward(dscore)

        return dh