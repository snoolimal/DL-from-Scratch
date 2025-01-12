from config import np
from common.base import Model
from common.layers.embedding import Embedding
from common.layers import NegativeSamplingLoss


class CBOW(Model):
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
        """vs. SimpleCBOW
        SimpleCBOW class에서 h와 행렬곱되는 w_out은 단어 벡터를 열 방향으로 담고 있었다.
        CBOW class는 h와 w_out의 행렬곱을 embedding layer로 실현하므로 w_in과 동일하게 단어 벡터를 행 방향으로 배치한다. 
        """

        # Model Architecture
        self.in_layers = []
        for i in range(context_window_size):
            layer = Embedding(w_in)     # 공유하는 가중치
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(w_out, tokenized_corpus, power=0.74, sample_size=5)

        # Computation Graph
        layers = self.in_layers + [self.ns_loss]
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
        h = 0
        # batch의 data 각각에 대해 ith context 처리
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(indices=contexts[:, i])
        h *= 1.0 / len(self.in_layers)          # avg layer
        loss = self.ns_loss.forward(h, target)

        return loss

    def backward(self, dy=1):
        dy = self.ns_loss.backward(dy)
        dy *= 1.0 / len(self.in_layers)         # avg layer
        for layer in self.in_layers:            # reversed() 먹일 필요 X
            layer.backward(dy)

        return None