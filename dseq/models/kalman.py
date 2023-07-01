import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions


class KalmanCell(tf.keras.layers.Layer):

    def __init__(
            self,
            dim_state=None, dim_obs=None,
            kernel_initializer='glorot_uniform',
            name='kalman_cell', **kwargs
            ):

        super(KalmanCell, self).__init__(name=name, **kwargs)
        self._dim_state = dim_state
        self._dim_obs = dim_obs
        self._kernel_initializer = kernel_initializer

        def kinit():
            return tf.keras.initializers.get(self._kernel_initializer)

        self.A = tf.Variable(kinit()((self._dim_state, self._dim_state)), trainable=True)
        self.C = tf.Variable(kinit()((self._dim_obs, self._dim_state)), trainable=True)
        self.Q = tf.Variable(kinit()((self._dim_state, self._dim_state)), trainable=True)
        self.R = tf.Variable(kinit()((self._dim_obs, self._dim_obs)), trainable=True)
        self.I = tf.eye(self._dim_state)

    @property
    def state_size(self):
        """ This refers to the shape of the RNN cell state which combines mean and variance. """
        return [(self._dim_state, 1), (self._dim_state, self._dim_state)]

    @property
    def output_size(self):
        """ This refers to the RNN cell state which combines mean and variance. """
        return [(self._dim_state,), (self._dim_state, self._dim_state)]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """ This refers to the RNN cell state which combines mean and variance. """
        batch_size = inputs[0] if batch_size is None else batch_size
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

        new_mean = self.A @ mean
        new_cov = (self.A @ cov) @ AT + tf.math.softplus(self.Q)

        K = new_cov @ CT @ tf.linalg.inv(self.C @ new_cov @ CT + tf.math.softplus(self.R))
        new_mean = new_mean + K @ (obs - self.C @ new_mean)
        new_cov = (self.I - K @ self.C) @ new_cov

        return new_mean, new_cov

    def call(self, obs, state):
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
        return (new_state[0][..., 0], new_state[1]), new_state

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_state': self._dim_state,
            'dim_obs': self._dim_obs,
            'kernel_initializer': self._kernel_initializer
        })
        return config


class KalmanFilter(tf.keras.Model):

    def __init__(self,
                dim_state=None, dim_obs=None,
                name='kalman_filter', **kwargs
            ):
        super(KalmanFilter, self).__init__(name=name, **kwargs)
        self._dim_state = dim_state
        self._dim_obs = dim_obs
        self._cell = KalmanCell(dim_state=dim_state, dim_obs=dim_obs, **kwargs)
        self._rnn = tf.keras.layers.RNN(
            self._cell, return_sequences=True, return_state=False,
        )
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._llh_tracker = tf.keras.metrics.Mean(name="log_llh")

    def call(self, observations):
        return self._rnn(observations)

    def log_likelihood(self, observations):
        states, _ = self(observations)
        cov_chol = tf.linalg.cholesky(tf.math.softplus(self._cell.R))
        return tfpd.MultivariateNormalTriL(
            tf.matmul(states, self._cell.C, transpose_b=True), cov_chol
            ).log_prob(observations)

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
        loss = -tf.reduce_mean(tf.reduce_sum(
                    self.log_likelihood(data),
                        axis=1))
        log_llh = self.log_likelihood(data)
        self._loss_tracker.update_state(loss)
        self._llh_tracker.update_state(log_llh)
        return {
            "loss": self._loss_tracker.result(),
            "log_llh": self._llh_tracker.result()
            }

    @property
    def metrics(self):
        return [self._loss_tracker, self._llh_tracker]

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_state': self._dim_state,
            'dim_obs': self._dim_obs
        })
        return config
