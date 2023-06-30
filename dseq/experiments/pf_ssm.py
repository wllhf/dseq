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

params['LEARNING_RATE'] = 0.00001
params['NUM_EPOCHS'] = 10
params['N_PARTICLES'] = 25
params['MODEL_PATH'] = os.path.join(params['MODEL_PATH'], 'lin_gau_ssm_pf')

tf.keras.utils.set_random_seed(params['SEED']+10)

# model
model = ParticleFilter(dim_state=params['DIM_STATE'], dim_obs=params['DIM_OBS'], n_particles=params['N_PARTICLES'])
optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'])
model.compile(optimizer=optimizer)

lin_gaussian_ssm.main(params, model, repr='particle')