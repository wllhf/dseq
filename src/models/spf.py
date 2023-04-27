"""
Particle Filter Recurrent Neural Networks
Xiao Ma, Peter Karkus, David Hsu, Wee Sun Lee

https://arxiv.org/abs/1905.12885
"""

import tensorflow as tf

from tensorflow.python.util import nest

from models.pf import ParticleFilterCell, ParticleFilter


class SoftParticleFilterCell(ParticleFilterCell):

    def __init__(self, m_model, f_model, alpha=0.5, **kwargs):
        super().__init__(m_model, f_model, **kwargs)
        self._alpha = alpha

    def _soft_weights(self, weights):
        return weights - tf.log(self._alpha*tf.exp(weights)+(1-self._alpha)/self._n_particles)

    def call(self, input, state, training=None):
        particle_cell_state, weights = state
        norm_weights = self._normalize(weights)
        soft_weights = self._soft_weights(norm_weights)
        indices, particle_cell_state = self._resample(soft_weights, particle_cell_state)
        new_particles, new_particle_cell_state = self._forward(input, particle_cell_state)
        new_weights = self._measurement(input, new_particles)
        new_weights = new_weights + tf.gather(weights, indices, batch_dims=1)
        return (new_particles, new_weights), (new_particle_cell_state, new_weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self._alpha
        })
        return config


class SoftParticleFilter(ParticleFilter):

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
