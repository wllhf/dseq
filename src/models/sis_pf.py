import tensorflow as tf

from tensorflow.python.util import nest

from .pf_lib import *

class ParticleFilterCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, m_model, f_model, **kwargs):
        super().__init__(**kwargs)
        self._m_model = m_model
        self._f_model = f_model

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
        indices = tf.random.categorical(tf.squeeze(weights), self._f_model._n_particles, dtype='int32')
        particle_cell_state = [tf.gather(s, indices, batch_dims=1) for s in particle_cell_state]
        weights = tf.gather(weights, indices, batch_dims=1)
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
        return weights

    def call(self, input, state, training=None):
        particle_cell_state, weights = state
        weights = self._normalize(weights)
        _, particle_cell_state = self._resample(weights, particle_cell_state)
        particles, particle_cell_state = self._forward(input, particle_cell_state)
        weights = self._measurement(input, particles)
        return (particles, weights), (particle_cell_state, weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            'm_model': self._m_model,
            'f_model': self._f_model
        })
        return config


class ParticleFilter(tf.keras.Model):

    def __init__(self,
                 dim_state, dim_obs, n_particles,
                 m_model='lingau', f_model='lingau',
                 n_inner_cell_layers=2, inner_cell_cls=tf.keras.layers.LSTMCell,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        if m_model == 'lingau':
            m_model = LinGaussianMeasurementModel(n_particles, dim_state, dim_obs)

        if f_model == 'lingau':
            f_model = LinGaussianTransitionCell(n_particles, dim_state)
        elif f_model == 'rnn':
            f_model = RNNTransitionCell(tf.keras.layers.StackedRNNCells(
                [inner_cell_cls(dim_state) for _ in range(n_inner_cell_layers)]
            ), n_particles)

        self._pf_cell = tf.keras.layers.RNN(
            ParticleFilterCell(
                m_model=m_model,
                f_model=f_model,
            ),
            return_sequences=True
        )

        self._m_model = m_model
        self._f_model = f_model
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")

    def log_likelihood(self, observations):
        _, weights = self(observations)
        return tf.math.reduce_logsumexp(weights, axis=2)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(tf.reduce_sum(
                self.log_likelihood(data),
                  axis=1))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def test_step(self, data):
        loss = - tf.reduce_mean(tf.reduce_sum(
                self.log_likelihood(data),
                  axis=1))
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    @property
    def metrics(self):
        return [self._loss_tracker]

    def call(self, observations):
        return self._pf_cell(observations)

    def get_config(self):
        config = super().get_config()
        config.update({
            'm_model': self._m_model,
            'f_model': self._f_model
        })
        return config
