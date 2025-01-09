from common.base import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=1e-2):
        super().__init__()

        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]