import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.util import nest

from .utils import get_gaussian_diag_model, get_mlp

tfpd = tfp.distributions


class VRNNCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self,
            dim_state, dim_obs, dim_feat=None, dim_rnn=None,
            rnn_cell=None, x_feat_net=None, z_feat_net=None,
            prior_stub=None, enc_stub=None, dec_stub=None,
            convert_to_tensor_fn=tfp.distributions.Distribution.sample,
            **kwargs
            ):
        """
        args:
          dim_state: Integer state dimensionality
          dim_obs: Integer, observation dimensionality
          dim_feat: Integer, output dimensionality of both feature extractors.
          dim_rnn: Integer, hidden state dimensionality of the recurrent network.
        """
        super(VRNNCell, self).__init__(**kwargs)

        self._dim_state = dim_state
        self._dim_obs = dim_obs
        self._dim_feat = dim_feat or dim_state
        self._dim_rnn = dim_rnn or dim_state

        self._rnn_cell = rnn_cell
        self._x_feat_net = x_feat_net
        self._z_feat_net = z_feat_net
        self._prior_stub = prior_stub or get_mlp(dims_hidden=[1], dim_out=self._dim_state, name='pstub')
        self._enc_stub = enc_stub or get_mlp(dims_hidden=[1], dim_out=self._dim_state, name='estub')
        self._dec_stub = dec_stub or get_mlp(dims_hidden=[1], dim_out=self._dim_obs, name='dstub')

        self._dist_state_cls = tfpd.MultivariateNormalDiag
        self._dist_obs_cls = tfpd.MultivariateNormalDiag
        self._convert_to_tensor_fn = convert_to_tensor_fn

        # feature extractors
        self._phi_x = x_feat_net or get_mlp(dims_hidden=[1], dim_out=self._dim_feat, name='x_feat_net')
        self._phi_z = z_feat_net or get_mlp(dims_hidden=[1], dim_out=self._dim_feat, name='z_feat_net')

        # distribution paramters
        self._phi_prior = get_gaussian_diag_model(self._prior_stub, (self._dim_rnn,), dim_state, name='prior')
        self._phi_dec = get_gaussian_diag_model(self._enc_stub, (self._dim_feat+self._dim_rnn,), dim_obs, name='decoder')
        self._phi_enc = get_gaussian_diag_model(self._dec_stub, (self._dim_feat+self._dim_rnn,), dim_state, name='encoder')

        # recurrent network
        self._f = rnn_cell or tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(self._dim_rnn), tf.keras.layers.LSTMCell(self._dim_rnn)]
        )
        # self._f = rnn_cell or tf.keras.layers.StackedRNNCells(
        #     [tf.keras.layers.LSTMCell(self._dim_rnn)]
        # )

    @property
    def state_size(self):
        """ This corresponds to the cell state of the inner RNN cell. """
        return self._f.state_size

    @property
    def output_size(self):
        """ This is mean and cov of the posterior. """
        return [
            ((self._dim_state,), (self._dim_state,)),
            ((self._dim_state,), (self._dim_state,)),
            ((self._dim_obs,), (self._dim_obs,)),
            self._f.state_size
        ]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """ This corresponds to the cell state of the inner RNN cell. """
        return self._f.get_initial_state(inputs, batch_size, dtype)

    def _state_input(self, state):
        #state = nest.flatten(state)
        return state[-1][0] if isinstance(state, (tuple, list)) else state

    def _decode(self, z, state):
        z_feat = self._phi_z(z)
        input_tensor = tf.concat([z_feat, self._state_input(state)], axis=-1)
        return self._phi_dec(input_tensor)

    def call(self, input, state, training=None):

        x_feat = self._phi_x(input)

        # posterior
        input_tensor = tf.concat([x_feat, self._state_input(state)], axis=-1)
        mean_zt, cov_zt = self._phi_enc(input_tensor)
        posterior = self._dist_state_cls(mean_zt, cov_zt)
        z_enc = self._convert_to_tensor_fn(posterior)

        # rnn update
        z_feat = self._phi_z(z_enc)
        input_tensor = tf.concat([x_feat, z_feat], axis=-1)
        output, new_state = self._f(input_tensor, state)

        # prior
        mean_0t, cov_0t = self._phi_prior(self._state_input(state))
        #prior = self._dist_state_cls(mean_0t, cov_0t)
        #z_dec = self._convert_to_tensor_fn(prior)

        # decoding
        mean_xt, cov_xt = self._decode(z_enc, state)

        return [(mean_zt, cov_zt), (mean_0t, cov_0t), (mean_xt, cov_xt), new_state], new_state


class VRNN(tf.keras.Model):

    def __init__(self, cell=None, dim_state=None, dim_obs=None, dim_feat=None, dim_rnn=None, **kwargs):
        """
        args:
          cell: VRNNCell object
        """
        super().__init__(**kwargs)

        self._cell = cell or VRNNCell(
            dim_state, dim_obs, dim_feat, dim_rnn,
            )

        self._rnn = tf.keras.layers.RNN(self._cell, return_sequences=True)
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")

    def _neg_elbo(self, rnn_output, observations):
        posterior = self._cell._dist_state_cls(*rnn_output[0])
        prior = self._cell._dist_state_cls(*rnn_output[1])
        llh = self._cell._dist_obs_cls(*rnn_output[2])
        kl = tf.reduce_sum(tfpd.kl_divergence(posterior, prior), axis=1)
        reconstruction_loss = -tf.reduce_sum(llh.log_prob(tf.cast(observations, tf.float32)), axis=1)
        return tf.reduce_mean(kl) + tf.reduce_mean(reconstruction_loss)

    def log_likelihood(self, observations):
        outputs = self._rnn(observations)
        dist = self._cell._dist_obs_cls(*outputs[2])
        return dist.log_prob(observations)

    def call(self, inputs):
        return self._rnn(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self._rnn(data)
            loss = self._neg_elbo(output, data)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def test_step(self, data):
        output = self._rnn(data)
        loss = self._neg_elbo(output, data)
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    @property
    def metrics(self):
        return [self._loss_tracker]

    def sample(self, input, seq_len):
        pass
