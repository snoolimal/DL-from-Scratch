from utils.np import np


class SGD:
    """Stochastic Gradient Descent
    w <- w - (lr * dLdw)
    """

    def __init__(self, lr=1e-2):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class AdaGrad:
    """Adaptive Gradient
    h <- h * ||dLdw||
    lr <- lr / sqrt(h + eps)
    w <- w - (lr * dLdw)
    """
    def __init__(self, lr=1e-2):
        self.lr = lr
        self.h = None

    def step(self, params, grads):
        if self.h is None: self.h = [np.zeros_like(param) for param in params]

        for i in range(len(params)):
            self.h[i] += grads[i] ** 2                  # 1st moment는 update마다 누적하지만
            lr = self.lr / (np.sqrt(self.h[i]) + 1e-7)  # lr은 매 update마다 새롭게 계산되므로 self.lr의 /=로 누적 처리 X

            params[i] -= lr * grads[i]


class RMSprop:
    """Root Mean Squared Propagation
    h <- (beta * h) + ((1 - beta) * ||dLdw||)
    lr <- lr / sqrt(h + eps)
    w <- w - (lr * dLdw)
    """
    def __init__(self, lr=1e-2, beta=0.99):
        self.lr = lr
        self.beta = beta
        self.h = None

    def step(self, params, grads):
        if self.h is None: self.h = [np.zeros_like(param) for param in params]

        for i in range(len(params)):
            self.h[i] = (self.beta * self.h[i]) + ((1 - self.beta) * grads[i] ** 2)
            lr = self.lr / (np.sqrt(self.h[i]) + 1e-7)

            params[i] -= lr * grads[i]


class Adam:
    """Adaptive Moment
    m <- (beta1 * m) + ((1 - beta1) * dLdw)
    v <- (beta2 * v) + ((1 - beta2) * ||dLdw||)
    lr <- lr / sqrt(v + eps)
    w <- w - (lr * m)
    """
    def __init__(self, lr=1e-2, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
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

            lr = self.lr / (np.sqrt(vhat) + 1e-7)

            params[i] -= lr * mhat

