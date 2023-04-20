import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from models.kalman import KalmanFilter

from utils import ssm_params as params
from utils import ssm_loader, ssm_plot, ssm_log_likelihood

tfpd = tfp.distributions

params['LEARNING_RATE'] = 0.0005
params['NUM_EPOCHS'] = 20

# data
data_trn_target_ssm_state, data_trn_target_ssm_cov, data_trn_obs  = ssm_loader(params, mode='trn')

data_val_target_ssm_state, data_val_target_ssm_cov, data_val_obs = ssm_loader(params, mode='val')

data_tst_target_ssm_state, data_tst_target_ssm_cov, data_tst_obs = ssm_loader(params, mode='tst')

# model
model = KalmanFilter(dim_state=params['DIM_STATE'], dim_obs=params['DIM_OBS'])
optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'])
model.compile(optimizer=optimizer)
# model.summary()

model.fit(
    data_trn_obs,
    batch_size=params['BATCH_SIZE'],
    epochs=params['NUM_EPOCHS']
)

# tf.keras.saving.save_model(model, params['MODEL_PATH'])

# evaluate
est_ssm_state, est_ssm_cov = model(data_tst_obs)

# likelihood
log_llh = ssm_log_likelihood(params, data_tst_target_ssm_state, data_tst_obs)
print(np.mean(log_llh))

log_llh = model.log_likelihood(data_tst_obs)
print(np.mean(log_llh))

# plot
ssm_plot(
    params,
    data_tst_obs,
    data_tst_target_ssm_state,
    data_tst_target_ssm_cov,
    est_ssm_state,
    est_ssm_cov
    )