import os
import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from models.deep_kalman import DeepKalmanFilter

from utils import ssm_params as params
from utils import ssm_loader, ssm_plot

tfpd = tfp.distributions

# data
data_trn_target_ssm_state, data_trn_target_ssm_cov, data_trn_obs  = ssm_loader(params, mode='trn')

data_val_target_ssm_state, data_val_target_ssm_cov, data_val_obs = ssm_loader(params, mode='val')

data_tst_target_ssm_state, data_tst_target_ssm_cov, data_tst_obs = ssm_loader(params, mode='tst')

# model
# observations = tf.keras.Input(shape=(params['SEQ_LEN'], params['DIM_OBS']), name='observations')
model = DeepKalmanFilter(dim_state=params['DIM_STATE'], dim_obs=params['DIM_OBS']) # , inputs=observations) # (observations)
# model = tf.keras.Model(inputs=observations, outputs=estimates)
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

# plot
ssm_plot(
    params,
    data_tst_obs,
    data_tst_target_ssm_state,
    data_tst_target_ssm_cov,
    est_ssm_state,
    est_ssm_cov
    )