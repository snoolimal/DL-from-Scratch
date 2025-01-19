from config import np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def matmul(x, w):
    return np.matmul(x, w)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    # single data는 batch form으로 일반화
    if x.ndim == 1:
        x = x.reshape(-1, x.size)

    # overflow 방지 (계산 결과는 동일)
    x -= x.max(axis=1, keepdims=True)

    # softmax
    x = np.exp(x)
    y = x / x.sum(axis=1, keepdims=True)  # keepdims for broadcasting [N,C]/[N,1]=[N,C]

    return y


def cross_entropy_error(y, t):
    # single data는 batch form으로 일반화
    if y.ndim == 1:
        t = t.reshape(-1, t.size)
        y = y.reshape(-1, y.size)

    batch_size = y.shape[0]

    if t.size == y.size:                        # one-hot target vector라면 class idx vector로 변환하여 손쉽게 계산
        t = t.argmax(axis=1, keepdims=False)    # (default) keepdims=False여야 ce 계산 시 indexing 가능

    _loss = np.negative(np.log(y[np.arange(batch_size), t] + 1e-7))
    loss = np.sum(_loss) / batch_size

    return loss
