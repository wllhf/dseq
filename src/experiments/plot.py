import numpy as np
import matplotlib.pyplot as plt

from data.ssm import lin_gaussian_ssm

def softmax(x, axis=0):
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex/ex.sum(axis=axis, keepdims=True)


def plot_lin_gaussian_ssm_1d(ax, params, state, cov, obs=None, c='k'):
    t = range(params['SEQ_LEN'])

    ax.plot(t, state, c=c)
    ax.plot(t, state + np.sqrt(cov), c+'--')
    ax.plot(t, state - np.sqrt(cov), c+'--')

    if obs is not None:
        ax.scatter(t, obs, c='g')

def plot_particle_ssm_1d(ax, params, particles, weights, c=None):
    t = range(params['SEQ_LEN'])

    weights = softmax(weights, axis=1)
    for i in t:
        ax.scatter([i]*particles.shape[1], particles[i, :], c=weights[i, :], marker='.', cmap='Blues')

    # mean particle
    ax.scatter(t, np.sum(particles*weights, axis=1), marker='x', c='r')
    # max particle
    best = np.take_along_axis(particles, np.expand_dims(np.argmax(weights, axis=1), axis=1), axis=1)
    ax.scatter(t, best, marker='*', c='r')
