from config.np import np
from _common.functions import linear, sigmoid, softmax, cross_entropy_error, mean_squared_error


class SoftmaxWithLoss:
    def __init__(self):
        # self.params, self.grads = [], []

        self.y = None
        self.t = None

    def forward(self, x, t):
        y = softmax(x)  # [N,C]

        self.y = y
        self.t = t

        loss = cross_entropy_error(y, t)

        return loss

    def backward(self, dy=1):
        batch_size = self.y.shape[0]

        y = self.y.copy()
        y[np.arange(batch_size), self.t] -= 1
        dloc = y / batch_size

        dx = dloc * dy

        return dx


class SigmoidWithLoss:
    def __init__(self):
        # self.params, self.grads = [], []

        self.y = None
        self.t = None

    def forward(self, x, t):
        y = sigmoid(x)  # [N,]

        self.y = y
        self.t = t

        y = np.c_[1 - self.y, self.y]  # concat arrays along col -> [N,2]

        loss = cross_entropy_error(y, t)

        return loss

    def backward(self, dy=1):
        batch_size = self.y.shape[0]

        dloc = (self.y - self.t) / batch_size

        dx = dloc * dy

        return dx


class LinearWithLoss:
    def __init__(self):
        # self.params, self.grads = [], []

        self.y = None
        self.t = None

    def forward(self, x, t):
        y = linear(x)   # [N,]

        self.y = y
        self.t = t

        loss = mean_squared_error(y, t)

        return loss

    def backward(self, dy=1):
        batch_size = self.y.shape[0]

        dloc = (self.y - self.t) / batch_size

        dx = dloc * dy

        return dx
