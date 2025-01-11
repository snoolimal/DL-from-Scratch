from common.functions import sigmoid, softmax, cross_entropy_error
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
