from infrastructure.np import np


class MatMul:
    def __init__(self, w):
        self.params = [w]
        self.grads = [np.zeros_like(w)]

        self.x = None

    def forward(self, x):
        self.x = x

        w, = self.params
        y = np.dot(x, w)

        return y

    def backward(self, dy):
        x = self.x

        w, = self.params
        dw = np.dot(x.T, dy)
        dx = np.dot(dy, w.T)

        self.grads[0][...] = dw
        """cf. [...]
        1. self.grads[0] = dw (shallow copy)
        dw를 위한 새로운 메모리 공간이 할당되고 self.grads[0]가 해당 공간을 참조
        2. self.grads[0][...] = dw (deep copy)
        self.grads[0]가 차지하는 메모리 공간의 값만 dw로 덮어 씀
        """
        return dx


class Bias:
    def __init__(self, b):
        self.params = [b]
        self.grads = [np.zeros_like(b)]

    def forward(self, x):
        b, = self.params
        y = x + b

        return y

    def backward(self, dy):
        db = np.sum(dy, axis=0)
        dx = dy

        self.grads[0][...] = db
        return dx


class Affine:
    def __init__(self, w, b):
        self.params = [w, b]
        self.grads = [np.zeros_like(param) for param in self.params]

        self.x = None

    def forward(self, x):
        self.x = x

        w, b = self.params
        y = np.dot(x, w) + b

        return y

    def backward(self, dy):
        x = self.x
        w, _ = self.params

        dw = np.dot(x.T, dy)
        db = np.sum(dy, axis=0)
        dx = np.dot(dy, w.T)

        self.grads[0][...] = dw
        self.grads[1][...] = db
        return dx
