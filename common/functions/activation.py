from infrastructure.np import np


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Args:
        x: logit vector의 row-wise matrix
    ---
    Returns: probability vector의 row-wise matrix
    """
    # single data는 batch form으로 일반화
    if x.ndim == 1:
        x = x.reshape(-1, x.size)

    # overflow 방지 (계산 결과는 동일)
    x -= x.max(axis=1, keepdims=True)

    x = np.exp(x)
    x /= x.sum(axis=1, keepdims=True)  # keepdims for broadcasting [N,C]/[N,1]=[N,C]

    return x


def linear(x):
    return x
