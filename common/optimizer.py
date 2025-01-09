from config import np
from common.base import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent
    w <- w - (lr * dw)
    """
    def __init__(self, lr=1e-2):
        super().__init__(lr=lr)

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Adam(Optimizer):
    """Adaptive Moment
    m <- (beta1 * m) + ((1 - beta1) * dw)
    v <- (beta2 * v) + ((1 - beta2) * ||dw||)
    lr <- lr / sqrt(v + eps)
    w <- w - (lr * m)
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grads):
        if self.m is None: self.m = [np.zeros_like(param) for param in params]
        if self.v is None: self.v = [np.zeros_like(param) for param in params]
        self.t += 1

        for i in range(len(params)):
            self.m[i] = (self.beta1 * self.m[i]) + ((1 - self.beta1) * grads[i])
            self.v[i] = (self.beta2 * self.v[i]) + ((1 - self.beta2) * grads[i] ** 2)

            # bias correction
            mhat = self.m[i] / (1 - self.beta1 ** self.t)
            vhat = self.v[i] / (1 - self.beta2 ** self.t)

            lr = self.lr / (np.sqrt(vhat) + self.eps)

            params[i] -= lr * mhat