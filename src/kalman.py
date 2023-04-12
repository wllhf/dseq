import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions

class Kalman(tf.keras.layers.Layer):

    def __init__(
            self,
            dim_state=None, dim_obs=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            **kwargs
            ):

        super(Kalman, self).__init__(**kwargs)
        self._dim_state = dim_state
        self._dim_obs = dim_obs
        self._kernel_initializer = kernel_initializer
        # self._bias_initializer = bias_initializer

        kinit = tf.keras.initializers.get(self._kernel_initializer)
        # binit = tf.keras.initializers.get(self._bias_initializer)
        self.A = tf.Variable(kinit((self._dim_state, self._dim_state)), trainable=True)
        self.C = tf.Variable(kinit((self._dim_obs, self._dim_state)), trainable=True)
        self.R = tf.Variable(kinit((self._dim_state, self._dim_state)), trainable=True)
        self.Q = tf.Variable(kinit((self._dim_state, self._dim_state)), trainable=True)
        self.I = tf.eye(self._dim_state)

    @property
    def state_size(self):
        """ This refers to the shape of the RNN cell state which combines mean and variance. """
        return [(self._dim_state, 1), (self._dim_state, self._dim_state)]

    @property
    def output_size(self):
        # return self._dim_state
        return self.state_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """ This refers to the RNN cell state which combines mean and variance. """
        batch_size = inputs.shape[0] if batch_size is None else batch_size
        dtype = inputs.dtype if dtype is None else dtype
        return (
            tf.zeros(shape=[batch_size, self._dim_state, 1], dtype=dtype),
            tf.eye(self._dim_state, batch_shape=(batch_size,), dtype=dtype)
            )

    def _step(self, obs, mean, cov):
        """
        This method contains the actual Kalman filter update.

        args:
          obs: Tensor [batch_size, dim_obs, 1]
          mean: Tensor [batch_size, dim_state, 1]
          cov: Tensor [batch_size, dim_state, dim_state]
        """
        AT = tf.transpose(self.A)
        CT = tf.transpose(self.C)

        new_mean = tf.matmul(self.A, mean)
        new_cov = tf.matmul(tf.matmul(self.A, cov), AT) + self.R

        K = tf.matmul(tf.matmul(new_cov, CT), tf.linalg.inv(tf.matmul(tf.matmul(self.C, new_cov), CT)+self.Q))
        new_mean = new_mean + tf.matmul(K, (obs-tf.matmul(self.C, new_mean)))
        new_cov = tf.matmul((self.I - tf.matmul(K, self.C)), new_cov)

        return new_mean, new_cov

    def call(self, obs, state, training=None):
        """
        This method is implemented according to the RNN cell interface
        to be used with the Keras RNN layer.

        args:
          obs: observation at t tensor [dim_obs]
          state: RNN state at t [dim_state (dim_state, dim_state)]

        returns:
          output: output at t
          state: state at t+1
        """
        obs = tf.cast(obs, dtype=tf.float32)
        new_state = self._step(tf.expand_dims(obs, axis=-1), state[0], state[1])
        #return tfpd.Normal(new_state[0], new_state[1]).sample()[..., 0], new_state
        return new_state, new_state

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_state': self._dim_state,
            'dim_obs': self._dim_obs,
            'kernel_initializer': self._kernel_initializer,
            'bias_initializer': self._kernel_initializer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
