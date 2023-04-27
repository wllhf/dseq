import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.util import nest

from .utils import get_gaussian_diag_model

tfpd = tfp.distributions

MIN_LOG_WEIGHT = tf.math.log(1e-6)

MAX_LOG_WEIGHT = tf.constant(0.)
MIN_LOG_WEIGHT = tf.constant(-1000.)


def _stop_gradients_at_tensor_elements(tensor, mask):
    """ Stops gradient at mask == 1. """
    mask = tf.cast(mask, tensor.dtype)
    inv_mask = 1. - mask
    return tf.stop_gradient(mask*tensor) + inv_mask*tensor


def stop_gradient_for_dead_weights(weights):
    return _stop_gradients_at_tensor_elements(weights, weights < MIN_LOG_WEIGHT)


def clip_weights(weights):
    weights = tf.where(tf.math.is_nan(weights), MIN_LOG_WEIGHT, weights)
    return tf.clip_by_value(weights, MIN_LOG_WEIGHT, MAX_LOG_WEIGHT)


class TransitionCell(tf.keras.layers.AbstractRNNCell):
    """
    """

    def __init__(self, n_particles, dim_state, transition_noise_stddev=1.0, **kwargs):
        """
        args:
          n_particles: Integer
          dim_state: Integer
        """
        super(TransitionCell, self).__init__(**kwargs)
        self._n_particles = n_particles
        self._dim_state = dim_state
        self._transition_noise_stddev = transition_noise_stddev

    @property
    def state_size(self):
        return tf.TensorShape([self._n_particles, self._dim_state])

    @property
    def output_size(self):
        return self.state_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        batch_size = tf.shape(inputs)[0] if batch_size is None else batch_size
        dtype = inputs.dtype if dtype is None else dtype
        return tf.zeros([batch_size, self._n_particles, self._dim_state], dtype=dtype)

    def call(self, input, state, training=None):
        """
        Args:
          input:
          state:
        Returns:
          output, state:
        """
        raise NotImplementedError

    def log_likelihood(self, new_particles, particles):
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_particles': self._n_particles,
            'dim_state': self._dim_state,
            'transition_noise_stddev': self._transition_noise_stddev
        })
        return config


class LinGaussianTransitionCell(TransitionCell):
    """
    """

    def __init__(self, n_particles, dim_state, **kwargs):
        """
        args:
          n_particles: Integer
        """
        super(LinGaussianTransitionCell, self).__init__(n_particles, dim_state, **kwargs)
        self._convert_to_tensor_fn = tfp.distributions.Distribution.sample
        self._dist_cls = tfpd.MultivariateNormalDiag
        self._model = get_gaussian_diag_model(None, (None, self._dim_state), self._dim_state, name='t_inner')

    def call(self, input, particles):
        diff, cov = self._model(particles)
        dist = self._dist_cls(diff, cov)
        new_particles = particles + dist.sample()
        return new_particles, new_particles

    def log_likelihood(self, new_particles, particles):
        diff, cov = self._model(particles)
        dist = self._dist_cls(diff, cov)
        return dist.log_prob(new_particles - particles)


class RNNTransitionCell(TransitionCell):
    """
    Wrapper class to feed a batch of particles through a single RNN cell.
    The input is concatenated with a noise tensor that has the same
    dimensions as the particle tensor. Noise is only added to the input
    of the first cell in case of a stacked RNN cell.
    """

    def __init__(self, n_particles, dim_state, cell, **kwargs):
        """
        args:
          n_particles: Integer
          dim_state: Integer
          cell: RNNCell or StackedRNNCell
            cell.state_size needs to be a scalar or list of scalars,
            cells with with higher rank state spaces are not supported.
        """
        super(RNNTransitionCell, self).__init__(n_particles, dim_state, **kwargs)
        self._cell = cell

    @property
    def state_size(self):
        return nest.pack_sequence_as(
            self._cell.state_size,
            [tf.TensorShape([self._n_particles, s]) for s in nest.flatten(self._cell.state_size)]
        )

    @property
    def output_size(self):
        return tf.TensorShape([self._n_particles, self._cell.output_size])

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        init_state = self._cell.get_initial_state(inputs, batch_size, dtype)
        return nest.pack_sequence_as(
            init_state,
            [tf.tile(s[:, tf.newaxis, :], [1, self._n_particles, 1]) for s in nest.flatten(init_state)]
        )

    def call(self, input, state, training=None):
        """
        Args:
          states:
        Returns:
          output, states:
        """
        state = nest.flatten(state)

        dtype = state[0].dtype
        particle_tensor_shape = tf.shape(state[0])
        state_dim = particle_tensor_shape[2]

        # Generate noise
        input_tensor = tf.random.normal(particle_tensor_shape, stddev=self._forward_noise_stddev, dtype=dtype)

        # Merge batch and particle dimension
        state = [tf.reshape(s, [-1, state_dim]) for s in state]
        input_tensor = tf.reshape(input_tensor, [-1, state_dim])

        # Concatenate input with noise vector
        if input is not None:
            obs_dim = tf.shape(input)[-1]
            input = tf.tile(input[:, tf.newaxis, :], [1, self._n_particles, 1])
            input = tf.reshape(input, [-1, obs_dim])
            input_tensor = tf.concat([input_tensor, input], axis=-1)

        # Apply inner cell
        state = nest.pack_sequence_as(self._cell.state_size, state)
        outputs, state = self._cell(input_tensor, state, training=training)

        # Unmerge batch and particle dimension
        outputs = tf.reshape(outputs, particle_tensor_shape)
        state = [tf.reshape(state, particle_tensor_shape) for state in nest.flatten(state)]

        return outputs, nest.pack_sequence_as(self._cell.state_size, state)

    def get_config(self):
        config = super().get_config()
        config.update({
            'cell': self._cell,
        })
        return config


class MeasurementModelBase(tf.keras.layers.Layer):

    def __init__(self, n_particles, dim_state, dim_obs, **kwargs):
        super(MeasurementModelBase, self).__init__(**kwargs)
        self._dim_obs = dim_obs
        self._dim_state = dim_state
        self._n_particles = n_particles

    def call(self, observation, particles, training=None):
        """
        args:
          observation: Tensor
          particles: Tensor
        """
        raise NotImplementedError

    def log_likelihood(self, observation, particles):
        return self(observation, particles)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_particles': self._n_particles,
            'dim_state': self._dim_state,
            'dim_obs': self._dim_obs
        })
        return config




class MeasurementModel(MeasurementModelBase):

    def __init__(self, n_particles, dim_state, dim_obs, m_net, noise_stddev=1.0, **kwargs):
        super(MeasurementModel, self).__init__(n_particles, dim_state, dim_obs, **kwargs)
        self._m_net = m_net
        self._noise_stddev = noise_stddev

    def call(self, observation, particles, training=None):
        """
        args:
          observation: Tensor
          particles: Tensor
        """
        dtype = particles[0].dtype
        particle_tensor_shape = tf.shape(particles)
        noise_tensor = tf.random.normal(particle_tensor_shape, stddev=self._measurement_noise_stddev, dtype=dtype)
        observation = tf.tile(observation[:, tf.newaxis], [1, tf.shape(particles)[1], 1])
        input_tensor = tf.concat([observation, particles, noise_tensor], axis=-1)
        weights = self._m_net(input_tensor, training=training)
        # weights = _clip_weights(weights)
        return weights

    def get_config(self):
        config = super().get_config()
        config.update({
            'm_net': self._m_net,
            'noise_stddev': self._noise_stddev
        })
        return config


class LinGaussianMeasurementModel(MeasurementModelBase):

    def __init__(self, n_particles, dim_state, dim_obs, **kwargs):
        """
        args:
          n_particles: Integer
          dim_state: Integer
        """
        super(LinGaussianMeasurementModel, self).__init__(n_particles, dim_state, dim_obs, **kwargs)
        self._VAR_EPS = 10e-10
        self._dist_cls = tfpd.MultivariateNormalDiag
        self._model = get_gaussian_diag_model(None, (None, self._dim_state), self._dim_obs, name='m_inner')

    def call(self, observation, particles):
        obs_est, obs_cov = self._model(particles)
        obs_cov = obs_cov + self._VAR_EPS
        dist = self._dist_cls(tf.zeros(tf.shape(obs_est)), obs_cov)
        #tf.print('state', tf.math.reduce_min(obs_est), tf.math.reduce_max(obs_est))
        #tf.print('cov', tf.math.reduce_min(obs_cov), tf.math.reduce_max(obs_cov))
        diff = observation[:, tf.newaxis, :] - obs_est
        #tf.print('diff', tf.math.reduce_min(diff), tf.math.reduce_max(diff))
        weights = dist.log_prob(diff)
        #tf.print('weights', tf.math.reduce_min(weights), tf.math.reduce_max(weights))
        return clip_weights(weights[:, :, tf.newaxis])
