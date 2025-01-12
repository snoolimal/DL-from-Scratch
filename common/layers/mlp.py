from common.functions import sigmoid, softmax
from config import np


class MatMul:
    def __init__(self, w):
        self.params = [w]
        self.grads = [np.zeros_like(w)]
        self.x = None

    def forward(self, x):
        self.x = x
        w, = self.params
        return np.dot(x, w)

    def backward(self, dy):
        w, = self.params
        x = self.x

        dx = np.dot(dy, w.T)
        dw = np.dot(x.T, dy)

        self.grads[0][...] = dw
        return dx


class Affine:
    def __init__(self, w, b):
        self.params = [w, b]
        self.grads = [np.zeros_like(param) for param in self.params]
        self.x = None

    def forward(self, x):
        self.x = x
        w, b = self.params
        return np.dot(x, w) + b

    def backward(self, dy):
        w, b = self.params
        x = self.x

        dx = np.dot(dy, w.T)
        dw = np.dot(x.T, dy)
        db = np.sum(dy, axis=0)

        self.grads[0][...] = dw
        self.grads[1][...] = db
        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None

    def forward(self, x):
        self.y = softmax(x)
        return self.y

    def backward(self, dy):
        dx = self.y * dy    # hadmard product
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.y * sumdx

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None

    def forward(self, x):
        self.y = sigmoid(x)
        return self.y

    def backward(self, dy):
        dloc = (1.0 - self.y) * self.y
        dx = dloc * dy  # hadmard product

        return dx
