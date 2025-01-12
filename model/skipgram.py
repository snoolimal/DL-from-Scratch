from config import np
from common.base import Model
from common.layers.embedding import Embedding
from common.layers import NegativeSamplingLoss


class SkipGram(Model):
    def __init__(self, tokenized_corpus, vocab_size, hidden_size, window_size):
        """
        Args:
            tokenized_corpus: [NN,]
            vocab_size V: input_size
            hidden_size H:
        """
        super().__init__()

        context_window_size = 2 * window_size
        input_size = vocab_size

        # Weight Initialization (단어 벡터)
        w_in = 0.01 * np.random.randn(input_size, hidden_size)
        w_out = 0.01 * np.random.randn(input_size, hidden_size)

        # Model Architecture
        self.in_layer = Embedding(w_in)
        self.loss_layers = []
        for i in range(2 * window_size):
            layer = NegativeSamplingLoss(w_out, tokenized_corpus, power=0.75, sample_size=5)
            self.loss_layers.append(layer)

        # Computation Graph
        layers = [self.in_layer] + self.loss_layers
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 단어의 분산 표현 밀집 벡터
        self.word_vecs = w_in

    def forward(self, contexts, target):
        """
        Args:
            contexts: tokenized contexts (단어 ID) | [N,C]
                N: batch size
                    한 번에 처리하는 context의 개수
                C: context window size (2*window size)
            target: tokenized target (단어 ID) | [N,]
        Returns:
            loss: scalar
        """
        h = self.in_layer.forward(target)

        loss = 0
        for i, layer in enumerate(self.loss_layers):
            loss += layer.forward(h, contexts[:, i])

        return loss

    def backward(self, dy=1):
        dh = 0
        for i, layer in enumerate(self.loss_layers):
            dh += layer.backward(dy)

        self.in_layer.backward(dh)

        return None