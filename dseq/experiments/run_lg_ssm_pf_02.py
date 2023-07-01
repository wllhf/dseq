import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf

from dseq.experiments import lin_gaussian_ssm

from dseq.models.pf import ParticleFilter

from dseq.configs.training import params
from dseq.configs.models import lin_gaussian_ssm_params

params.update(lin_gaussian_ssm_params)

params['LEARNING_RATE'] = 0.001
params['NUM_EPOCHS'] = 20
params['N_PARTICLES'] = 25

tf.keras.utils.set_random_seed(params['SEED']+10)

root = params['MODEL_PATH']

# model
params['MODEL_PATH'] = os.path.join(root, 'lin_gau_ssm_pf_rnn')
params['NUM_SAMPLES'] = {'trn': 50*params['BATCH_SIZE'], 'val': 128, 'tst': 100}
model = ParticleFilter(
    dim_state=params['DIM_STATE'], dim_obs=params['DIM_OBS'], n_particles=params['N_PARTICLES'],
    f_model='rnn'
    )
optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'])
model.compile(optimizer=optimizer)

lin_gaussian_ssm.main(params, model, repr='particle')
