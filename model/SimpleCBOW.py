from config import np
from common.base import Model
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW(Model):
    def __init__(self, vocab_size, hidden_size):
        """
        window_size=1인 경우에 대해서만 작동한다.
        ---
        Args:
            vocab_size: input_size
            hidden_size: 단어 예측에 필요한 정보를 간결하게 담는 밀집 벡터 표현을 얻는 것이 핵심이므로 input_size보다 작게
        """
        super().__init__()

        input_size = vocab_size

        w_in = 0.01 * np.random.randn(input_size, hidden_size)      # encoding, 단어의 분산 표현 밀집 벡터 (행이 단어 ID에 대응)
        w_out = 0.01 * np.random.randn(hidden_size, input_size)     # encoding, 단어의 분산 표현 밀집 벡터 (열이 단어 ID에 대응)

        # Model Architecture
        self.in_layer0 = MatMul(w_in)
        self.in_layer1 = MatMul(w_in)           # context_window_size의 수만큼 w_in을 공유하는 상이한 layers 생성
        self.out_layer = MatMul(w_out)
        self.loss_layer = SoftmaxWithLoss()     # criterion
        layers = [
            self.in_layer0, self.in_layer1,
            # avg_layer,
            self.out_layer
        ]

        # Computation Graph
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 단어의 분산 표현
        self.word_vecs = w_in

    def forward(self, contexts, target):
        """
        Args:
            contexts: [N,C,V]
                N: batch size
                C: context window size (2*window size)
                V: vocab size (one-hot dim)
            target: [N,V]
        """
        h0 = self.in_layer0.forward(contexts[:, 0, :])
        h1 = self.in_layer1.forward(contexts[:, 1, :])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)

        return loss

    def backward(self, dy=1):
        dy = self.loss_layer.backward(dy)
        dy = self.out_layer.backward(dy)
        # avg layer
        dy *= 0.5
        _ = self.in_layer1.backward(dy)
        _ = self.in_layer0.backward(dy)

        return None
