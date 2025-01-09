from config import np


def normalize(x):
    """L2 Normalization"""
    assert x.ndim in [1, 2], 'Shape of x is either [B,] or [B,D].'
    flag = x.ndim - 1
    x /= np.sqrt((x ** 2).sum(axis=flag, keepdims=flag))    # ndim이 2면 keepdims [N,] as [N,1] for bdcast [N,D]/[N,1]
    return x
