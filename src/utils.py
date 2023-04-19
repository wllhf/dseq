import numpy as np
import matplotlib.pyplot as plt

from data.ssm import lin_gaussian_ssm



ssm_params = {
    'SSM_A': [[0.5]],
    'SSM_C': [[1.0]],
    'SSM_Q': [[0.5]],
    'SSM_R': [[0.1]],
    'DIM_OBS': 1,
    'DIM_STATE': 1,
    'SEQ_LEN': 20,
    'NUM_SAMPLES': {'trn': 20480, 'val': 128, 'tst': 100},
    'BATCH_SIZE': 1024,
    'LEARNING_RATE': 0.001,
    'NUM_EPOCHS': 200,
    'SEEDS': {'trn': 0, 'val': 1234, 'tst': 9000},
    'MODEL_PATH': '~/proj/dpf_final/'
}


def ssm_loader(params, mode='trn'):
    return lin_gaussian_ssm(
        num_samples=params['NUM_SAMPLES'][mode], seq_len=params['SEQ_LEN'],
        A=params['SSM_A'], C=params['SSM_C'], Q=params['SSM_Q'], R=params['SSM_R'],
        seed=params['SEEDS'][mode])


def ssm_plot(params, target_obs, target_state, target_cov, est_state, est_cov):
    t = range(params['SEQ_LEN'])
    plt.plot(t, target_state[0, ...], c='k')
    plt.plot(t, target_state[0, ...]+np.sqrt(target_cov[0, ..., 0]), 'k--', )
    plt.plot(t, target_state[0, ...]-np.sqrt(target_cov[0, ..., 0]), 'k--', )
    plt.plot(t, est_state[0, ...], c='b')
    plt.plot(t, est_state[0, ...]+np.sqrt(est_cov[0, ..., 0]), 'b--', )
    plt.plot(t, est_state[0, ...]-np.sqrt(est_cov[0, ..., 0]), 'b--', )
    plt.scatter(t, target_obs[0, ...], c='g')
    plt.show()