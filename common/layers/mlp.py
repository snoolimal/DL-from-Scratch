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


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        """
        (Batch가 들어 와) Fpass 때마다 삭제할 neuron을 self.mask에 False로 표시한다.
        ---
        cf. MLP node 생각하고 행렬곱의 차원 mapping으로 생각하면
            입력 node는 x의 각 행이니까, 다음 layer의 첫 node와의 연결은 그 행의 새로운 공간의 첫 축으로의 mapping이니까
            그 연결선들은 w의 1열 벡터지.
        ---
        "neuron을 끈다"고 표현해서 좀 애매하지만, 실제로는 batch 내 sample마다 다른 dropout 패턴을 사용하므로
        아래와 같인 mask 생성(batch 전체에서 동일한 열을 끄는, 즉 열 단위로 랜덤하게 0이 찍히는 게 아니라)하는 것이 일반적.
        그러므로 구조적인 작동 자체는 element-wise인 ReLU인데, 다만 0 or identity가 random하게 결정되는 구조.
        그러므로 bpass에서 dy 받아서 1 or 0(f(x_ij)=0의 상수함수의 x_ij 미분은 0)인 loc grad(가 아님!)와 곱해서 흘리면 됨.
        그러면 끈 출력의 loss에 대한 영향력, 즉 흘러들어온 grad가 끈 출력에 흐른 grad는 죽임.
        """
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask / (1.0 - self.dropout_ratio)   # 두 scaling은
        else:
            return x     # 본질적으로는 동일

    def backward(self, dy):
        """
        Dropout은 fpass에서 element-wise로 mask에 따라 identity function이거나 0을 출력한다.
        그러므로 local gradient는 mask에 대해 1이거나 0이다. (Diagonal의 값이 1 또는 0)
        원리는 작동 자체가 ReLU랑 같다.
        """
        return self.mask * dy
















