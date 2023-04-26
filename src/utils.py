import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from data.ssm import lin_gaussian_ssm

tfpd = tfp.distributions

def softmax(x, axis=0):
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex/ex.sum(axis=axis, keepdims=True)


def ssm_loader(params, mode='trn'):
    return lin_gaussian_ssm(
        num_samples=params['NUM_SAMPLES'][mode], seq_len=params['SEQ_LEN'],
        A=params['SSM_A'], C=params['SSM_C'], Q=params['SSM_Q'], R=params['SSM_R'],
        seed=params['SEEDS'][mode])


def ssm_log_likelihood(params, target_state, target_obs):
    obs_loc = np.einsum('ij,ki->ki', np.array(params['SSM_C']), target_state[..., 0])[..., None]
    cov_chol = tf.linalg.cholesky(np.array(params['SSM_R']))
    return tfpd.MultivariateNormalTriL(obs_loc, cov_chol).log_prob(target_obs)


def ssm_plot(ax, params, target_obs, target_state, target_cov, est_state, est_cov):
    t = range(params['SEQ_LEN'])

    ax.plot(t, target_state, c='k')
    ax.plot(t, target_state + np.sqrt(target_cov), 'k--')
    ax.plot(t, target_state - np.sqrt(target_cov), 'k--')

    ax.plot(t, est_state, c='b')
    ax.plot(t, est_state + np.sqrt(est_cov), 'b--')
    ax.plot(t, est_state - np.sqrt(est_cov), 'b--')

    ax.scatter(t, target_obs, c='g')

def ssm_particle_plot(ax, params, target_obs, target_state, target_cov, particles, weights):
    t = range(params['SEQ_LEN'])

    ax.plot(t, target_state, c='k')
    ax.plot(t, target_state+np.sqrt(target_cov), 'k--')
    ax.plot(t, target_state-np.sqrt(target_cov), 'k--')

    weights = softmax(weights, axis=1)
    for i in t:
        ax.scatter([i]*particles.shape[1], particles[i, :], c=weights[i, :], marker='.', cmap='Blues')

    # mean particle
    ax.scatter(t, np.sum(particles*weights, axis=1), marker='x', c='r')
    # max particle
    best = np.take_along_axis(particles, np.expand_dims(np.argmax(weights, axis=1), axis=1), axis=1)
    ax.scatter(t, best, marker='*', c='r')
    # target states
    ax.scatter(t, target_obs, c='g')