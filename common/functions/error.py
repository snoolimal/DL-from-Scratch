from utils.np import np


def mean_squared_error(y, t):
    assert (y.ndim == 1) and (t.ndim == 1)

    # single data는 batch form으로 일반화
    if np.isscalar(y):
        t = np.array([t])
        y = np.array([y])

    batch_size = y.shape[0]

    loss = (y - t) ** 2
    loss = np.sum(loss) / batch_size

    return loss


def cross_entropy_error(y, t):
    # single data는 batch form으로 일반화
    if y.ndim == 1:
        t = t.reshape(-1, t.size)
        y = y.reshape(-1, y.size)

    batch_size = y.shape[0]

    # one-hot target vector는 class idx vector로 변환 (ce loss의 정의대로, 편리하게 계산 가능)
    if t.size == y.size:
        t = t.argmax(axis=1, keepdims=False)    # default keepdims=False여야 ce loss 계산 시 indexing 가능

    loss = np.negative(np.log(y[np.arange(batch_size), t] + 1e-7))
    loss = np.sum(loss) / batch_size

    return loss
