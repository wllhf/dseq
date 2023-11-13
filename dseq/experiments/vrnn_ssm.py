import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from dseq.models.vrnn import VRNN

from dseq.configs.training import params
from dseq.configs.models import lin_gaussian_ssm_params

from dseq.data.ssm import lin_gaussian_ssm_loader as data_loader
from dseq.data.ssm import plot_lin_gaussian_ssm_1d as plotter
from dseq.data.ssm import lin_gaussian_ssm_log_likelihood as log_likelihood

from dseq.experiments.utils import get_default_callbacks, set_seed

tfpd = tfp.distributions

params.update(lin_gaussian_ssm_params)
params['NUM_SAMPLES'] = {'trn': 50*params['BATCH_SIZE'], 'val': 128, 'tst': 100}
# params['SEQ_LEN'] = 5
params['LEARNING_RATE'] = 0.0001
params['NUM_EPOCHS'] = 100
params['MODEL_PATH'] = os.path.join(params['MODEL_PATH'], 'lin_gau_ssm_vrnn')

set_seed(params['SEED'])

# data
data_trn_target_ssm_state, data_trn_target_ssm_cov, data_trn_obs = data_loader(params, mode='trn')

data_val_target_ssm_state, data_val_target_ssm_cov, data_val_obs = data_loader(params, mode='val')

data_tst_target_ssm_state, data_tst_target_ssm_cov, data_tst_obs = data_loader(params, mode='tst')

# model
model = VRNN(
    dim_state=params['DIM_STATE'], dim_obs=params['DIM_OBS'], dim_feat=params['DIM_STATE'], dim_rnn=2*params['DIM_STATE'],
    feat_ext_layers=0, feat_ext_width=8, stub_layers=0, stub_width=8,
    n_rnn_cell_layers=2, rnn_cell_cls=tf.keras.layers.LSTMCell,
    )
#optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'])
optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])
model.compile(optimizer=optimizer)
# model.summary()

# train
callbacks = get_default_callbacks(params)

model.fit(
    data_trn_obs,
    batch_size=params['BATCH_SIZE'],
    epochs=params['NUM_EPOCHS'],
    validation_data=(data_val_obs,),
    callbacks=callbacks
)

# tf.keras.saving.save_model(model, params['MODEL_PATH'])

# evaluate
est_ssm_state, est_ssm_cov = model(data_tst_obs)[0]

# likelihood
log_llh = log_likelihood(params, data_tst_target_ssm_state, data_tst_obs)
print(np.mean(log_llh))

log_llh = model.log_likelihood(data_tst_obs)
print(np.mean(log_llh))

# plot
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
for r in range(4):
    for c in range(4):
        plotter(
            axs[r, c],
            params,
            data_tst_target_ssm_state[r*c],
            np.squeeze(data_tst_target_ssm_cov[r*c]),
            data_tst_obs[r*c]
            )
        plotter(
            axs[r, c],
            params,
            est_ssm_state[r*c],
            np.squeeze(est_ssm_cov[r*c]),
            c='b'
            )

plt.show()