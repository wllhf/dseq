import numpy as np


def prod(M, v):
    """
    Multiply matrix with batched vector.

    args:
      M: [m, n]
      v: [batch_size, n]
    returns:
      w: [batch_size, m]
    """
    return np.einsum('ij,ki->ki', M, v)


def softmax(x, axis=0):
    """
    Numerically stable softmax.
    """
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex/ex.sum(axis=axis, keepdims=True)