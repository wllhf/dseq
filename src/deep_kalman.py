import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions


class DeepKalman(tf.keras.layers.Layer):

    def __init__(
            self,
            dim_state, dim_obs,
            g_stub=None, f_stub=None, r_stub=None,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
            **kwargs
            ):

        super(DeepKalman, self).__init__(**kwargs)
        self._dim_state = dim_state
        self._dim_obs = dim_obs
        self._dist_state_cls = tfpd.MultivariateNormalDiag
        self._dist_obs_cls = tfpd.MultivariateNormalDiag
        self._convert_to_tensor_fn = convert_to_tensor_fn

        def _get_dynamics_model(stub):
            """ p(z_t|z_t-1) """
            input = tf.keras.layers.Input((self._dim_state,))
            if stub is None:
                h = input
            else:
                h = stub(input)
            mean = tf.keras.layers.Dense(self._dim_state)(h)
            cov = tf.keras.layers.Dense(self._dim_state, activation='softplus')(h)
            return tf.keras.Model(inputs=[input], outputs=[mean, cov], name='dynamics')

        self._f = _get_dynamics_model(f_stub)

        def _get_generator_model(stub):
            """ p(x_t|z_t) """
            input = tf.keras.layers.Input((None, self._dim_state,))
            if stub is None:
                h = input
            else:
                h = stub(input)
            mean = tf.keras.layers.Dense(self._dim_obs)(h)
            cov = tf.keras.layers.Dense(self._dim_obs, activation='softplus')(h)
            return tf.keras.Model(inputs=[input], outputs=[mean, cov], name='generator')

        self._g = _get_generator_model(g_stub)

        def _get_recognition_model(stub):
            """ p(z|x) """
            seq_inp = tf.keras.layers.Input((None, dim_obs))
            # ini_inp = tf.keras.layers.Input((self._dim_state,))

            if stub is None:
                h = tf.keras.layers.RNN(
                    tf.keras.layers.LSTMCell(2*self._dim_state),
                    return_sequences=True,
                    return_state=False,
                    go_backwards=False,
                )(seq_inp) # , initial_state=ini_inp)
            else:
                h = stub(seq_inp) # , initial_state=ini_inp)

            mean = tf.keras.layers.Dense(self._dim_state)(h)
            cov = tf.keras.layers.Dense(self._dim_state, activation='softplus')(h)

            return tf.keras.Model(inputs=[seq_inp], outputs=[mean, cov], name='recognition')

        self._r = _get_recognition_model(g_stub)

    def train_step(self, obs, initial_state):
        """
        args:
          obs: Tensor [batch_size, seq_len, dim_obs]
          state: Tuple ([batch_size, dim_state], [batch_size, cov_shape])
        """
        # compute prior
        prior = [initial_state]
        for t in range(obs.shape[1]):
            d = self._dist_state_cls(prior[-1][0], prior[-1][1])
            input = self._convert_to_tensor_fn(d)
            prior.append(self._f(input))
        prior = prior[1:]
        prior = tf.stack([p[0] for p in prior], axis=1), tf.stack([p[1] for p in prior], axis=1)
        dist_prior = self._dist_state_cls(prior[0], prior[1])

        # compute posterior
        posterior = self._r([obs])
        dist_posterior = self._dist_state_cls(posterior[0], posterior[1])

        # compute output distribution
        input = self._convert_to_tensor_fn(dist_posterior)
        obs_est = self._g(input)
        dist_obs_est = self._dist_obs_cls(obs_est[0], obs_est[1])

        # loss
        kl = tfp.distributions.kl_divergence(dist_posterior, dist_prior)
        reconstruction_loss = -tf.reduce_sum(dist_obs_est.log_prob(tf.cast(obs, tf.float32)), axis=1)
        loss = tf.reduce_mean(kl) + tf.reduce_mean(reconstruction_loss)

        return loss


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

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_state': self._dim_state,
            'dim_obs': self._dim_obs
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
