import tensorflow as tf

from tensorflow.python.util import nest

from .pf_lib import *
from .pf_cells import *


class ParticleFilter(tf.keras.Model):

    def __init__(self,
                 dim_state, dim_obs, n_particles,
                 m_model='lingau', f_model='lingau',
                 n_inner_cell_layers=2, inner_cell_cls=tf.keras.layers.LSTMCell,
                 version = 'default',
                 **kwargs
                 ):
        super().__init__(**kwargs)

        if version == 'default':
            pf_cell_cls = ParticleFilterCell
        elif version == 'sis':
            pf_cell_cls = SISParticleFilterCell
        elif version == 'soft_resampling':
            pf_cell_cls = SoftParticleFilterCell
        else:
            raise NotImplementedError('Particle filter version unknown:', version)

        if m_model == 'lingau':
            m_model = LinGaussianMeasurementModel(n_particles, dim_state, dim_obs)

        if f_model == 'lingau':
            f_model = LinGaussianTransitionCell(n_particles, dim_state)
        elif f_model == 'rnn':
            f_model = RNNTransitionCell(
                n_particles, dim_state,
                tf.keras.layers.StackedRNNCells(
                    [inner_cell_cls(dim_state) for _ in range(n_inner_cell_layers)]
                )
            )

        self._pf_cell = tf.keras.layers.RNN(
            pf_cell_cls(
                m_model=m_model,
                f_model=f_model,
            ),
            return_sequences=True
        )

        self._m_model = m_model
        self._f_model = f_model
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._llh_tracker = tf.keras.metrics.Mean(name="log_llh")

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
        log_llh = - loss # self.log_likelihood(data)
        self._loss_tracker.update_state(loss)
        self._llh_tracker.update_state(log_llh)
        return {
            "loss": self._loss_tracker.result(),
            "log_llh": self._llh_tracker.result()
            }

    @property
    def metrics(self):
        return [self._loss_tracker, self._llh_tracker]

    def call(self, observations):
        return self._pf_cell(observations)

    def get_config(self):
        config = super().get_config()
        config.update({
            'm_model': self._m_model,
            'f_model': self._f_model
        })
        return config
