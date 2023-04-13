import numpy as np


def lin_gaussian_ssm(
        dim_state=1, dim_obs=1, seq_len=10, batch_size=8,
        A=np.array([[0.5]]), C=np.array([[1]]), Q=np.array([[1]]), R=np.array([[1]])
        ):

    sta = np.empty([batch_size, seq_len, dim_state])
    obs = np.empty([batch_size, seq_len, dim_obs])

    I = np.eye(dim_state)
    loc_sta = np.zeros(dim_state)
    loc_obs = np.zeros(dim_state)

    def prod(M, v):
        return np.einsum('ij,ki->ki', M, v)

    sta[:, 0, :] = np.random.normal(loc=loc_sta, scale=I, size=(batch_size, dim_state))
    obs[:, 0, :] = prod(C, sta[:, 0, :]) + np.random.normal(loc=loc_sta, scale=R, size=(batch_size, dim_obs))

    for t in range(1, seq_len):
        sta[:, t, :] = prod(A, sta[:, t-1, :]) + np.random.normal(loc=loc_obs, scale=Q, size=(batch_size, dim_state))
        obs[:, t, :] = prod(C, sta[:, t, :]) + np.random.normal(loc=loc_sta, scale=R, size=(batch_size, dim_obs))

    return sta, obs



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    seq_len = 10
    sta, obs = lin_gaussian_ssm(seq_len=seq_len, batch_size=2)

    t = range(seq_len)
    plt.plot(t, sta[0, ...])
    plt.scatter(t, obs[0, ...])
    plt.show()
