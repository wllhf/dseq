import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from models.vrnn import VRNN

from utils import ssm_params as params
from utils import ssm_loader, ssm_plot, ssm_log_likelihood

tfpd = tfp.distributions

params['LEARNING_RATE'] = 0.0001
params['NUM_EPOCHS'] = 50

# data
data_trn_target_ssm_state, data_trn_target_ssm_cov, data_trn_obs  = ssm_loader(params, mode='trn')

data_val_target_ssm_state, data_val_target_ssm_cov, data_val_obs = ssm_loader(params, mode='val')

data_tst_target_ssm_state, data_tst_target_ssm_cov, data_tst_obs = ssm_loader(params, mode='tst')

# model
model = VRNN(dim_state=params['DIM_STATE'], dim_obs=params['DIM_OBS'])
optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])
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
(est_ssm_state, est_ssm_cov), _, _, _ = model(data_tst_obs)
print(est_ssm_state.shape, est_ssm_cov.shape)
print(data_tst_target_ssm_state.shape, data_tst_target_ssm_cov.shape)

# likelihood
log_llh = ssm_log_likelihood(params, data_tst_target_ssm_state, data_tst_obs)
print(np.mean(log_llh))

log_llh = model.log_likelihood(data_tst_obs)
print(np.mean(log_llh))

# plot
ssm_plot(
    params,
    data_tst_obs[0],
    data_tst_target_ssm_state[0],
    np.squeeze(data_tst_target_ssm_cov[0]),
    est_ssm_state[0],
    est_ssm_cov[0]
    )

plt.show()
