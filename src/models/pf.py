import tensorflow as tf

from tensorflow.python.util import nest


class ParticleCell(tf.keras.layers.AbstractRNNCell):
    """
    Wrapper class to feed a batch of particles through a single RNN cell.
    The input is concatenated with a noise tensor that has the same
    dimensions as the particle tensor. Noise is only added to the input
    of the first cell in case of a stacked RNN cell.
    """

    def __init__(
            self,
            cell,
            n_particles,
            **kwargs
            ):
        """
        args:
          cell: RNNCell or StackedRNNCell
            cell.state_size needs to be a scalar or list of scalars,
            cells with with higher rank state spaces are not supported.
          n_particles: Integer
        """
        super(ParticleCell, self).__init__(**kwargs)
        self._cell = cell
        self._n_particles = n_particles
        self._forward_noise_stddev = 1.0

    @property
    def state_size(self):
        return nest.pack_sequence_as(
            self._cell.state_size,
            [tf.TensorShape([self._n_particles, s]) for s in nest.flatten(self._cell.state_size)]
        )

    @property
    def output_size(self):
        return tf.TensorShape([self._n_particles, self._cell.output_size])

    def get_initial_state(self, **kwargs):
        init_state = self._cell.get_initial_state(**kwargs)
        return tf.tile(init_state[:, tf.newaxis, :], [1, self._n_particles, 1])

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
        states = [tf.reshape(state, [-1, state_dim]) for state in states]
        input_tensor = tf.reshape(input_tensor, [-1, state_dim])

        # Concatenate input with noise vector
        if input is not None:
            obs_dim = tf.shape(input)[-1]
            input = tf.tile(input[:, tf.newaxis, :], [1, self._n_particles, 1])
            input = tf.reshape(input, [-1, obs_dim])
            input_tensor = tf.concat([input_tensor, input], axis=-1)

        # Apply inner cell
        states = nest.pack_sequence_as(self._cell.state_size, states)
        outputs, states = self._cell(input_tensor, states, training=training)

        # Unmerge batch and particle dimension
        outputs = tf.reshape(outputs, particle_tensor_shape),
        states = [tf.reshape(state, particle_tensor_shape) for state in nest.flatten(states)]

        return outputs, nest.pack_sequence_as(self._cell.state_size, states)

    def get_config(self):
        config = super().get_config()
        config.update({
            'cell': self._cell,
            'n_particles': self._n_particles
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ParticleFilterCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, m_model, f_model):
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
            tf.ones([batch_size, self._f_model._n_particles], dtype=dtype)/self._f_model._n_particles
        )

    def _resample(self, weights, particle_cell_state):
        """
        args:
          weights: [batch_size, n_particles]
          particles: particle cell state
        """
        particle_cell_state = nest.flatten(particle_cell_state)
        # TODO: weights need to be logits, are they?
        indices = tf.random.categorical(weights, self._f_model._n_particles, dtype='int32')
        particle_cell_state = tf.gather(particle_cell_state, indices, batch_dims=1)
        weights = tf.gather(weights, indices, batch_dims=1)
        return weights, nest.pack_sequence_as(self._f_model.state_size, particle_cell_state)

    def _measurement(self, observation, particles, training=None):
        # TODO: Add noise!
        """
        args:
          observation: [batch_size, dim_obs]
          particles: [batch_size, n_particles, dim_state]
        """
        observation = tf.expand_dims(observation, axis=1)
        observation = tf.tile(observation, [1, tf.shape(particles)[1], 1])
        input_tensor = tf.concat([observation, particles], axis=-1)
        weights = self._m_model(input_tensor, training=training)
        return weights

    def _forward(self, observation, particle_cell_state, training=None):
        """
        args:
          observation: [batch_size, dim_obs]
          particles: particle cell state
        """
        return self._f_model(observation, particle_cell_state, training=training)

    def call(self, input, state, training=None):
        particle_cell_state, weights = state
        # TODO: Normalize weights!
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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ParticleFilter(tf.keras.Model):

    def __init__(self,
                 dim_state, n_particles,
                 m_model=None,
                 n_inner_cell_layers=2, inner_cell_cls=tf.keras.layers.LSTMCell
                 ):

        inner_cell = tf.keras.layers.StackedRNNCells(
            [inner_cell_cls(dim_state) for _ in range(n_inner_cell_layers)]
        )

        self._pf_cell = tf.keras.layers.RNN(
            ParticleFilterCell(
                m_model=m_model,
                f_model=ParticleCell(inner_cell, n_particles)
            )
        )

    def train_step(self, data):
        pass

    def call(self, observations):
        pass

    def get_config(self):
        config = super().get_config()
        config.update({
            'm_model': self._m_model,
            'f_model': self._f_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)