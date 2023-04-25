import math
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns

from models.vrnn import VRNN

from utils import ssm_params as params
from utils import ssm_loader, ssm_plot, ssm_log_likelihood

tfpd = tfp.distributions

params['LEARNING_RATE'] = 0.00005
params['NUM_EPOCHS'] = 500

# data
data_trn_target_ssm_state, data_trn_target_ssm_cov, data_trn_obs  = ssm_loader(params, mode='trn')

data_val_target_ssm_state, data_val_target_ssm_cov, data_val_obs = ssm_loader(params, mode='val')

data_tst_target_ssm_state, data_tst_target_ssm_cov, data_tst_obs = ssm_loader(params, mode='tst')

# model
model = VRNN(
    dim_state=params['DIM_STATE'], dim_obs=params['DIM_OBS'], dim_feat=params['DIM_STATE'], dim_rnn=params['DIM_STATE'],
    feat_ext_layers=1, feat_ext_width=1, stub_layers=0, stub_width=1,
    n_rnn_cell_layers=1, rnn_cell_cls=tf.keras.layers.LSTMCell,
    )
optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'])
model.compile(optimizer=optimizer)
# model.summary()

model.fit(
    data_trn_obs,
    batch_size=params['BATCH_SIZE'],
    epochs=params['NUM_EPOCHS'],
    validation_data=(data_val_obs,)
)

# tf.keras.saving.save_model(model, params['MODEL_PATH'])

# evaluate
est_ssm_state, est_ssm_cov = model(data_tst_obs)
est_ssm_state, est_ssm_cov = est_ssm_state.numpy(), est_ssm_cov.numpy()
print(est_ssm_state.min(), est_ssm_state.max())
print(est_ssm_cov.min(), est_ssm_cov.max())

# likelihood
log_llh = ssm_log_likelihood(params, data_tst_target_ssm_state, data_tst_obs).numpy()
print(np.mean(log_llh))

log_llh = model.log_likelihood(data_tst_obs).numpy()
print(np.mean(log_llh))

# plot
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
for r in range(4):
    for c in range(4):
        ssm_plot(
            axs[r, c],
            params,
            data_tst_obs[r*c],
            data_tst_target_ssm_state[r*c],
            np.squeeze(data_tst_target_ssm_cov[r*c]),
            est_ssm_state[r*c],
            est_ssm_cov[r*c]
            )

plt.show()
