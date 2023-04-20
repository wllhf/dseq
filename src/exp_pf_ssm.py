import os
import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from models.pf import ParticleFilter

from utils import ssm_params as params
from utils import ssm_loader, ssm_plot, ssm_log_likelihood

tfpd = tfp.distributions

# data
data_trn_target_ssm_state, data_trn_target_ssm_cov, data_trn_obs  = ssm_loader(params, mode='trn')

data_val_target_ssm_state, data_val_target_ssm_cov, data_val_obs = ssm_loader(params, mode='val')

data_tst_target_ssm_state, data_tst_target_ssm_cov, data_tst_obs = ssm_loader(params, mode='tst')

# model
N_PARTICLES = 10
model = ParticleFilter(dim_state=params['DIM_STATE'], n_particles=N_PARTICLES)
optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'])
model.compile(optimizer=optimizer)
# model.summary()

# train
model.fit(
    data_trn_obs,
    batch_size=params['BATCH_SIZE'],
    epochs=params['NUM_EPOCHS']
    )

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