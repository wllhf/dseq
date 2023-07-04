import tensorflow as tf

from tensorflow.python.util import nest

from .pf_lib import *


class ParticleFilterCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, m_model, f_model, **kwargs):
        super().__init__(**kwargs)
        self._m_model = m_model
        self._f_model = f_model

    @property
    def n_particles(self):
        return self._f_model._n_particles

    @property
    def state_size(self):
        """ The state is a tuple of the state of the particle cell and weights. """
        return (self._f_model.state_size, tf.TensorShape(self._f_model._n_particles))

    @property
    def output_size(self):
        """ The output is a tuple of particles and weights. """
        return (self._f_model.output_size, tf.TensorShape(self._f_model._n_particles))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        batch_size = tf.shape(inputs)[0] if batch_size is None else batch_size
        dtype = inputs.dtype if dtype is None else dtype
        return (
            self._f_model.get_initial_state(inputs, batch_size, dtype),
            tf.ones([batch_size, self._f_model._n_particles, 1], dtype=dtype)/self._f_model._n_particles
        )

    def _resample(self, weights, particle_cell_state):
        """
        args:
          weights: [batch_size, n_particles, 1]
          particles: particle cell state
        """
        particle_cell_state = nest.flatten(particle_cell_state)
        indices = tf.random.categorical(tf.squeeze(weights), self.n_particles, dtype='int32')
        particle_cell_state = [tf.gather(s, indices, batch_dims=1) for s in particle_cell_state]
        return indices, nest.pack_sequence_as(self._f_model.state_size, particle_cell_state)

    def _measurement(self, observation, particles, training=None):
        """
        args:
          observation: [batch_size, dim_obs]
          particles: [batch_size, n_particles, dim_state]
        """
        return self._m_model(observation, particles, training=training)

    def _forward(self, observation, particle_cell_state, training=None):
        """
        args:
          observation: [batch_size, dim_obs]
          particles: particle cell state
        """
        return self._f_model(observation, particle_cell_state, training=training)

    def _normalize(self, weights):
        """
        args:
          weights: Tensor [batch_size, n_particles]
        """
        # TODO: maybe weights should be clipped
        # and gradient through very small weights be stopped
        weights = weights - tf.reduce_logsumexp(weights, axis=1, keepdims=True)
        weights = clip_weights(weights)
        weights = stop_gradient_for_dead_weights(weights)
        return weights

    def call(self, input, state, training=None):
        particle_cell_state, weights = state
        weights = self._normalize(weights)
        indices, particle_cell_state = self._resample(weights, particle_cell_state)
        particles, particle_cell_state = self._forward(input, particle_cell_state)
        new_weights = self._measurement(input, particles)
        new_weights = new_weights + tf.gather(weights, indices, batch_dims=1)
        return (particles, new_weights), (particle_cell_state, new_weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            'm_model': self._m_model,
            'f_model': self._f_model
        })
        return config


class SISParticleFilterCell(ParticleFilterCell):

    def call(self, input, state, training=None):
        particle_cell_state, weights = state
        weights = self._normalize(weights)
        particles, particle_cell_state = self._forward(input, particle_cell_state)
        new_weights = new_weights + self._measurement(input, particles)
        return (particles, weights), (particle_cell_state, weights)


class SoftParticleFilterCell(ParticleFilterCell):

    def __init__(self, m_model, f_model, alpha=0.5, **kwargs):
        super().__init__(m_model, f_model, **kwargs)
        self._alpha = alpha

    def _soft_weights(self, weights):
        return weights - tf.math.log(self._alpha*tf.exp(weights)+(1-self._alpha)/self.n_particles)

    def _resample(self, weights, particle_cell_state):
        """
        args:
          weights: [batch_size, n_particles, 1]
          particles: particle cell state
        """
        particle_cell_state = nest.flatten(particle_cell_state)
        soft_weights = self._soft_weights(weights)
        indices = tf.random.categorical(tf.squeeze(soft_weights), self.n_particles, dtype='int32')
        particle_cell_state = [tf.gather(s, indices, batch_dims=1) for s in particle_cell_state]
        return indices, nest.pack_sequence_as(self._f_model.state_size, particle_cell_state)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self._alpha
        })
        return config