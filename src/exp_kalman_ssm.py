import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from data.ssm import ssm_loader
from models.kalman import Kalman

tfpd = tfp.distributions

params = {
    'SSM_A': [[0.5]],
    'SSM_C': [[1.0]],
    'SSM_Q': [[0.5]],
    'SSM_R': [[0.1]],
    'DIM_OBS': 1,
    'DIM_STATE': 1,
    'SEQ_LEN': 20,
    'NUM_SAMPLES': 4096,
    'BATCH_SIZE': 512,
    'LEARNING_RATE': 0.0005,
    'NUM_EPOCHS': 100,
    'MODEL_PATH': '~/proj/dpf_final/'
}

# data
data_trn_target_ssm_state, data_trn_target_ssm_cov, data_trn_obs  = ssm_loader(
    num_samples=params['NUM_SAMPLES'], seq_len=params['SEQ_LEN'],
    A=params['SSM_A'], C=params['SSM_C'], Q=params['SSM_Q'], R=params['SSM_R'],
    seed=0)

data_val_target_ssm_state, data_val_target_ssm_cov, data_val_obs = ssm_loader(
   num_samples=32, seq_len=params['SEQ_LEN'],
   A=params['SSM_A'], C=params['SSM_C'], Q=params['SSM_Q'], R=params['SSM_R'],
   seed=1234)

data_tst_target_ssm_state, data_tst_target_ssm_cov, data_tst_obs = ssm_loader(
   num_samples=100, seq_len=params['SEQ_LEN'],
   A=params['SSM_A'], C=params['SSM_C'], Q=params['SSM_Q'], R=params['SSM_R'],
   seed=9000)

# model
observations = tf.keras.Input(shape=(params['SEQ_LEN'], params['DIM_OBS']), name='observations')
kalman_cell = Kalman(dim_state=params['DIM_STATE'], dim_obs=params['DIM_OBS'])
estimates = tf.keras.layers.RNN(
    kalman_cell, return_sequences=True, return_state=False,
    )(observations)
model = tf.keras.Model(inputs=observations, outputs=estimates)
model.compile()
model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'])

for epoch in range(params['NUM_EPOCHS']):

    for batch_start in range(0, params['NUM_SAMPLES'], params['BATCH_SIZE']):
        batch_end = batch_start + params['BATCH_SIZE']
        batch = data_trn_obs[batch_start:batch_end]
        target = (data_trn_target_ssm_state[batch_start:batch_end], data_trn_target_ssm_cov[batch_start:batch_end])

        with tf.GradientTape() as tape:
            estimates = model(batch)
            loss = tf.reduce_mean(tf.reduce_sum(
                kalman_cell.likelihood(batch, estimates[0]),
                  axis=1))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print('epoch:', epoch, 'loss:', loss.numpy())


# tf.keras.saving.save_model(model, params['MODEL_PATH'])

# evaluate
est_ssm_state, est_ssm_cov = model(data_tst_obs)

# likelihood
obs_loc = np.einsum('ij,ki->ki', np.array(params['SSM_C']), data_tst_target_ssm_state[..., 0])[..., None]
likelihood = tfpd.MultivariateNormalTriL(obs_loc, np.array(params['SSM_R'])).prob(data_tst_obs)
print(np.mean(likelihood))

likelihood = kalman_cell.likelihood(data_tst_obs, est_ssm_state)
print(np.mean(likelihood))

# plot
t = range(params['SEQ_LEN'])
plt.plot(t, data_tst_target_ssm_state[0, ...], c='k')
plt.plot(t, data_tst_target_ssm_state[0, ...]+np.sqrt(data_tst_target_ssm_cov[0, ..., 0]), 'k--', )
plt.plot(t, data_tst_target_ssm_state[0, ...]-np.sqrt(data_tst_target_ssm_cov[0, ..., 0]), 'k--', )
plt.plot(t, est_ssm_state[0, ...], c='b')
plt.plot(t, est_ssm_state[0, ...]+np.sqrt(est_ssm_cov[0, ..., 0]), 'b--', )
plt.plot(t, est_ssm_state[0, ...]-np.sqrt(est_ssm_cov[0, ..., 0]), 'b--', )
plt.scatter(t, data_tst_obs[0, ...], c='g')
plt.show()

# y_est, mean, cov = filter(data)

# sample = 0
# plt.plot(data[sample], c='k')
# plt.plot(y_est[sample], c='b')
# plt.hlines(mean[sample], 0, 10, colors=['g'])
# plt.hlines(mean[sample]+math.sqrt(cov[sample]), 0, 10, colors=['g'])
# plt.hlines(mean[sample]-math.sqrt(cov[sample]), 0, 10, colors=['g'])

# plt.show()