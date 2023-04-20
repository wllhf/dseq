"""
Deep Kalman Filters
Rahul G. Krishnan, Uri Shalit, David Sontag

https://arxiv.org/abs/1511.05121
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions


class DeepKalmanFilter(tf.keras.Model):

    def __init__(
            self,
            dim_state, dim_obs,
            g_stub=None, f_stub=None, r_rnn=None,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
            **kwargs
            ):

        super(DeepKalmanFilter, self).__init__(**kwargs)
        self._dim_state = dim_state
        self._dim_obs = dim_obs
        self._g_stub = g_stub
        self._f_stub = f_stub
        self._r_rnn = r_rnn
        self._dist_state_cls = tfpd.MultivariateNormalDiag
        self._dist_obs_cls = tfpd.MultivariateNormalDiag
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")

        def _get_gaussian_model(base, dim_in, dim_out, name):
            input = tf.keras.layers.Input(dim_in)
            h = input if base is None else base(input)
            mean = tf.keras.layers.Dense(dim_out, name='dense_mean')(h)
            cov = tf.keras.layers.Dense(dim_out, activation='softplus', name='dense_cov')(h)
            return tf.keras.Model(inputs=input, outputs=[mean, cov], name=name)

        # p(z_t|z_t-1)
        self._f = _get_gaussian_model(f_stub, (None, self._dim_state,), dim_state, 'dynamics')

        # p(x_t|z_t)
        self._g = _get_gaussian_model(g_stub, (None, self._dim_state,), dim_obs, 'generator')

        # p(z|x)
        if r_rnn is None:
            r_rnn = tf.keras.layers.RNN(
                    [
                        tf.keras.layers.LSTMCell(2*self._dim_state),
                        tf.keras.layers.LSTMCell(2*self._dim_state)
                    ],
                    return_sequences=True,
                )

        self._r = _get_gaussian_model(r_rnn, (None, self._dim_state), dim_state, 'recognition')

    def _neg_elbo(self, obs):
        """
        args:
          obs: Tensor [batch_size, seq_len, dim_obs]
        """
        # compute posterior p(z|x)
        q_mean, q_cov = self._r(obs)
        q = self._dist_state_cls(q_mean, q_cov)

        z = self._convert_to_tensor_fn(q)

        # compute prior p(z_t|z_t-1)
        p0_mean, p0_cov = self._f(z[:, :-1, :])
        p0 = self._dist_state_cls(p0_mean, p0_cov)

        # compute prior p(z_1)
        p_init = self._dist_state_cls(tf.zeros(self._dim_state), tf.eye(self._dim_state))

        # compute likelihood p(x_t|z_t)
        p_mean, p_cov = self._g(z)
        p = self._dist_obs_cls(p_mean, p_cov)

        # loss
        kl1 = tfp.distributions.kl_divergence(q[:, 0], p_init)
        klt = tf.reduce_sum(tfp.distributions.kl_divergence(q[:, 1:], p0), axis=1)
        reconstruction_loss = -tf.reduce_sum(p.log_prob(tf.cast(obs, tf.float32)), axis=1)
        loss = tf.reduce_mean(kl1) + tf.reduce_mean(klt) + tf.reduce_mean(reconstruction_loss)

        return loss

    def log_likelihood(self, observations):
        q_mean, q_cov = self._r(observations)
        q = self._dist_state_cls(q_mean, q_cov)

        z = self._convert_to_tensor_fn(q)

        p_mean, p_cov = self._g(z)
        p = self._dist_obs_cls(p_mean, p_cov)

        return p.log_prob(tf.cast(observations, tf.float32))

    def call(self, observations):
        """
        args:
          obs: observations [batch_size, seq_len, dim_obs]

        returns:
          mean, cov: [batch_size, seq_len, dim_state], [batch_size, seq_len, dim_state]
        """
        return self._r(observations)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._neg_elbo(data)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    @property
    def metrics(self):
        return [self._loss_tracker]

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_state': self._dim_state,
            'dim_obs': self._dim_obs,
            'g_stub': self._g_stub,
            'f_stub': self._f_stub,
            'r_rnn': self._r_rnn,
            'convert_to_tensor_fn': self._convert_to_tensor_fn
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
