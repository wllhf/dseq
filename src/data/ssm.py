import numpy as np


def _prod(M, v):
    """ Multiply matrix with batched vector. """
    return np.einsum('ij,ki->ki', M, v)


def lin_gaussian_ssm(A, C, Q, R, seq_len, num_samples, seed=0, dtype=np.float32):
    """
    Linear Gaussian State Space Model.
      state:
        x_t+1 = A*x_t + N(0, Q)
      observation
        y_t = C*x_t + N(0, R)

    args:
      A: list of lists with d(x) x d(x)
      C: list of lists with d(y) x d(x)
      Q: list of lists with d(x) x d(x)
      R: list of lists with d(y) x d(y)
    return:
      sta, cov, obs
    """
    random_state = np.random.RandomState(seed)

    dim_state = len(A)
    dim_obs = len(C)

    A = np.array(A, dtype=dtype)
    C = np.array(C, dtype=dtype)
    Q = np.array(Q, dtype=dtype)
    R = np.array(R, dtype=dtype)

    cov = np.zeros([num_samples, seq_len, dim_state, dim_state], dtype=dtype)
    sta = np.zeros([num_samples, seq_len, dim_state], dtype=dtype)
    obs = np.zeros([num_samples, seq_len, dim_obs], dtype=dtype)

    I = np.eye(dim_state)
    loc_sta = np.zeros(dim_state)
    loc_obs = np.zeros(dim_obs)

    sta[:, 0, :] = loc_sta
    obs[:, 0, :] = _prod(C, sta[:, 0, :]) + random_state.normal(loc_obs, scale=R, size=(num_samples, dim_obs))

    for t in range(1, seq_len):
        sta[:, t, :] = _prod(A, sta[:, t-1, :]) + random_state.normal(loc=loc_sta, scale=Q, size=(num_samples, dim_state))
        obs[:, t, :] = _prod(C, sta[:, t, :]) + random_state.normal(loc=loc_obs, scale=R, size=(num_samples, dim_obs))

    # constant covariance
    cov = np.broadcast_to(Q, (num_samples, seq_len, dim_state, dim_state))
    return sta, cov, obs



if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt
    seq_len = 5
    sta, cov, obs = lin_gaussian_ssm(seq_len=seq_len, num_samples=1, A=[[0.5]], C=[[1.0]], Q=[[0.5]], R=[[0.1]])

    t = range(seq_len)
    plt.plot(t, sta[0, ...])
    plt.scatter(t, obs[0, ...])
    plt.show()
