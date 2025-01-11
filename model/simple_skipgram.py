from config import np
from common.base import Model
from common.layers import MatMul, SoftmaxWithLoss


class SimpleSkipGram(Model):
    def __init__(self, vocab_size, hidden_size, window_size=1):
        super().__init__()
        self.context_window_size = 2 * window_size

        input_size = vocab_size

        # Weight Initialization (단어의 분산 표현 밀집 벡터)
        w_in = 0.01 * np.random.randn(input_size, hidden_size)
        w_out = 0.01 * np.random.randn(hidden_size, input_size)

        # Model Architecture
        self.in_layer = MatMul(w_in)
        self.out_layer = MatMul(w_out)
        self.loss_layers = [SoftmaxWithLoss() for _ in range(self.context_window_size)]
        self.layers = [
            self.in_layer,
            self.out_layer,
            *self.loss_layers
        ]

        # Computation Graph
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        # 단어의 분산 표현 밀집 벡터
        self.word_vecs = w_in

    def forward(self, contexts, target):
        """
        Args:
            contexts: [N,C,V]
                N: batch size
                    한 번에 처리하는 context의 개수
                C: context window size (2*window size)
                V: vocab size (one-hot dim)
                cf. contexts[:, i, :]: [N, V]
            target: [N,V]
        """
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        ls = np.stack([loss_layer.forward(s, contexts[:, i, :])
                       for i, loss_layer in enumerate(self.loss_layers)])   # l1, l2
        loss = np.sum(ls, axis=0)

        return loss

    def backward(self, dy=1):
        dys = np.stack([loss_layer.backward(dy) for loss_layer in self.loss_layers])    # dl1, dl2
        dy = np.sum(dys, axis=0)                                                        # dl1 + dl2
        dy = self.out_layer.backward(dy)
        _ = self.in_layer.backward(dy)

        return None
