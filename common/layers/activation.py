from utils.np import np
from common.functions import relu, sigmoid, softmax


class ReLU:
    def __init__(self):
        # self.params, self.grads = [], []

        self.mask = None    # bpass를 위한 값 저장 그릇

    def forward(self, x):
        self.mask = (x > 0)  # bpass를 위한 값 저장

        y = relu(x)

        return y

    def backward(self, dy):
        dloc = self.mask.astype(np.int32)
        dx = dloc * dy  # hadmard product

        return dx


class Sigmoid:
    def __init__(self):
        # self.params, self.grads = [], []

        self.y = None

    def forward(self, x):
        y = sigmoid(x)

        self.y = y

        return y

    def backward(self, dy):
        dloc = (1.0 - self.y) * self.y
        dx = dloc * dy  # hadmard product

        return dx


class Softmax:
    def __init__(self):
        # self.params, self.grads = [], []

        self.y = None

    def forward(self, x):
        y = softmax(x)

        self.y = y

        return y

    def bacwkard(self, dy):
        dx = self.y * dy
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.y * sumdx

        return dx
