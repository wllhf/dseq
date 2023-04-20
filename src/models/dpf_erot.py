"""
Differentiable Particle Filtering via Entropy-Regularized Optimal Transport
Adrien Corenflos, James Thornton, George Deligiannidis, Arnaud Doucet

https://arxiv.org/abs/2102.07850
"""
import tensorflow as tf


class DPF_EROT(tf.keras.Model):

    def __init__(self, epsilon=1.0, num_sinkhorn_iters=50):
        self._epsilon = epsilon
        self._num_sinkhorn_iters = num_sinkhorn_iters

    def _erot_resampling(self, particles, weights):
        n = particles.shape[0]

        # transport matrix
        pairwise_distances = tf.norm(particles[:, tf.newaxis] - particles[tf.newaxis, :], axis=-1)**2
        weight_matrix = tf.math.log(weights[:, tf.newaxis]) + tf.math.log(weights[tf.newaxis, :])

        K = tf.exp(-pairwise_distances / self._epsilon + weight_matrix)
        u = tf.ones(n) / n
        v = tf.ones(n) / n

        # Sinkhorn
        for _ in range(self._num_sinkhorn_iters):
            u = tf.reduce_sum(K * v[tf.newaxis, :], axis=-1)
            v = tf.reduce_sum(K * u[:, tf.newaxis], axis=0)

        T = K * u[:, tf.newaxis] * v[tf.newaxis, :]
        T = T / tf.reduce_sum(T, axis=1, keepdims=True)

        # sample particles
        indices = tf.random.categorical(tf.math.log(T), num_samples=n)
        resampled_particles = tf.gather(particles, indices, axis=1)
        return resampled_particles

    def _forward(self, particles):
        pass

    def _measurement(self, observation, particles):
        pass

    def loss():
        pass

    def call(self, observation, state):
        particles, weights = state
        # normalize weights

        # resample
        particles = self._erot_resampling(particles, weights)
        # sample
        particles = self._forward(particles)
        # compute weights
        weights = self._measurement(observation, particles)
        # compute likelihood

        state = (particles, weights)

        return state

    def train_step(self, data):
        pass