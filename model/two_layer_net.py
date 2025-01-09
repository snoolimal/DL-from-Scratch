from common.base import Model
from common.layers import Affine, Sigmoid, SoftmaxWithLoss
from config import np


class TwoLayerNet(Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        # Weight Initialization (0 근처로)
        w1 = 0.01 * np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        w2 = 0.01 * np.random.randn(hidden_size, num_classes)
        b2 = np.zeros(num_classes)

        # Model Architecture
        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w2, b2)
        ]

        # Criterion
        self.loss_layer = SoftmaxWithLoss()

        # Computation Graph
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        logit = self.predict(x)
        loss = self.loss_layer.forward(logit, t)
        return loss

    def backward(self, dy=1):
        dy = self.loss_layer.backward(dy)
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy
