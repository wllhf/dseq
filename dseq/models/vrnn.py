"""
A Recurrent Latent Variable Model for Sequential Data
Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, Yoshua Bengio

https://arxiv.org/abs/1506.02216
"""
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.util import nest

from .utils import get_gaussian_diag_model, get_mlp

tfpd = tfp.distributions


class VRNNCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self,
            dim_state, dim_obs, dim_feat=None,
            rnn_cell=None, x_feat_net=None, z_feat_net=None,
            prior_stub=None, enc_stub=None, dec_stub=None,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
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
        self._dim_rnn = rnn_cell.output_size

        self._f = rnn_cell
        self._x_feat_net = x_feat_net
        self._z_feat_net = z_feat_net
        self._prior_stub = prior_stub
        self._enc_stub = enc_stub
        self._dec_stub = dec_stub

        self._dist_state_cls = tfpd.MultivariateNormalDiag
        self._dist_obs_cls = tfpd.MultivariateNormalDiag
        self._convert_to_tensor_fn = convert_to_tensor_fn

        # feature extractors
        self._phi_x = tf.identity if x_feat_net is None else x_feat_net
        self._phi_z = tf.identity if z_feat_net is None else z_feat_net

        # distribution paramters
        self._phi_prior = get_gaussian_diag_model(self._prior_stub, (self._dim_rnn,), dim_state, name='prior')
        self._phi_dec = get_gaussian_diag_model(self._enc_stub, (self._dim_feat+self._dim_rnn,), dim_obs, name='decoder')
        self._phi_enc = get_gaussian_diag_model(self._dec_stub, (self._dim_feat+self._dim_rnn,), dim_state, name='encoder')

    @property
    def state_size(self):
        """ This corresponds to the cell state of the inner RNN cell. """
        return [self._f.output_size, self._f.state_size]

    @property
    def output_size(self):
        """ This is mean and cov of the posterior. """
        return [
            ((self._dim_state,), (self._dim_state,)),
            ((self._dim_state,), (self._dim_state,)),
            ((self._dim_obs,), (self._dim_obs,))
        ]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """ This corresponds to the cell state of the inner RNN cell. """
        batch_size = tf.shape(inputs)[0] if batch_size is None else batch_size
        dtype = inputs.dtype if dtype is None else dtype
        return [
            tf.zeros([batch_size, self.state_size[0]], dtype=dtype),
            self._f.get_initial_state(inputs, batch_size, dtype)
            ]

    def call(self, input, state, training=None):

        input_state, rnn_state = state
        x_feat = self._phi_x(input)

        # posterior
        input_tensor = tf.concat([x_feat, input_state], axis=-1)
        mean_zt, cov_zt = self._phi_enc(input_tensor)
        posterior = self._dist_state_cls(mean_zt, cov_zt)
        z_enc = self._convert_to_tensor_fn(posterior)

        z_feat = self._phi_z(z_enc)

        # rnn update
        input_tensor = tf.concat([x_feat, z_feat], axis=-1)
        output, new_state = self._f(input_tensor, rnn_state)

        # prior
        mean_0t, cov_0t = self._phi_prior(input_state)

        # decoding
        input_tensor = tf.concat([z_feat, input_state], axis=-1)
        mean_xt, cov_xt = self._phi_dec(input_tensor)

        return [(mean_zt, cov_zt), (mean_0t, cov_0t), (mean_xt, cov_xt)], [output, new_state]

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_state': self._dim_state,
            'dim_obs': self._dim_obs,
            'dim_feat': self._dim_feat,
            'rnn_cell': self._f,
            'x_feat_net': self._x_feat_net,
            'z_feat_net': self._z_feat_net,
            'prior_stub': self._prior_stub,
            'enc_stub': self._enc_stub,
            'dec_stub': self._dec_stub,
            'convert_to_tensor_fn': self._convert_to_tensor_fn
        })
        return config

class VRNN(tf.keras.Model):
    # TODO: sample

    def __init__(self, cell=None, dim_state=None, dim_obs=None, dim_feat=None, dim_rnn=None,
                 feat_ext_layers=1, feat_ext_width=1, stub_layers=0, stub_width=1,
                 n_rnn_cell_layers=1, rnn_cell_cls=tf.keras.layers.LSTMCell,
                 **kwargs
                 ):
        """
        args:
          cell: VRNNCell object
        """
        super().__init__(**kwargs)

        self._cell = cell

        if self._cell is None:
            rnn_cell = tf.keras.layers.StackedRNNCells(
                [rnn_cell_cls(dim_rnn) for _ in range(n_rnn_cell_layers)]
                )

            hidden_feat_layers = [feat_ext_width for _ in range(feat_ext_layers-1)]
            x_feat_net = get_mlp(dims_hidden=hidden_feat_layers, dim_out=dim_feat, name='x_feat_net')
            z_feat_net = get_mlp(dims_hidden=hidden_feat_layers, dim_out=dim_feat, name='z_feat_net')

            prior_stub, enc_stub, dec_stub = None, None, None

            if stub_layers > 0:
                hidden_stub_layers = [stub_width for _ in range(stub_layers-1)]
                prior_stub = get_mlp(dims_hidden=hidden_stub_layers, activation='relu', dim_out=dim_state, name='pstub')
                enc_stub = get_mlp(dims_hidden=hidden_stub_layers, activation='relu', dim_out=dim_state, name='estub')
                dec_stub = get_mlp(dims_hidden=hidden_stub_layers, activation='relu', dim_out=dim_obs, name='dstub')

            self._cell = VRNNCell(
                dim_state, dim_obs, dim_feat,
                rnn_cell, x_feat_net, z_feat_net,
                prior_stub, enc_stub, dec_stub
            )

        self._rnn = tf.keras.layers.RNN(self._cell, return_sequences=True)
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._llh_tracker = tf.keras.metrics.Mean(name="log_llh")

    def _neg_elbo(self, rnn_output, observations):
        posterior = self._cell._dist_state_cls(*rnn_output[0])
        prior = self._cell._dist_state_cls(*rnn_output[1])
        llh = self._cell._dist_obs_cls(*rnn_output[2])
        kl = tf.reduce_sum(tfpd.kl_divergence(posterior, prior), axis=1)
        reconstruction_loss = -tf.reduce_sum(llh.log_prob(tf.cast(observations, tf.float32)), axis=1)
        return tf.reduce_mean(kl) + tf.reduce_mean(reconstruction_loss)

    def log_likelihood(self, observations):
        mean_xt, cov_xt = self._rnn(observations)[2]
        dist = self._cell._dist_obs_cls(mean_xt, cov_xt)
        return dist.log_prob(tf.cast(observations, tf.float32))

    def call(self, inputs):
        return self._rnn(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self(data)
            loss = self._neg_elbo(output, data)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def test_step(self, data):
        output = self(data)
        loss = self._neg_elbo(output, data)
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

    def sample(self, input, seq_len):
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            'cell': self._cell
        })
        return config