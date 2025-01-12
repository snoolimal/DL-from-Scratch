import numpy as np  # np가 cupy면 normalize()에서 에러 발생


def normalize(x):
    """L2 Normalization"""
    assert x.ndim in [1, 2], 'Shape of x is either [N,] or [N,H].'

    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s

    return x
