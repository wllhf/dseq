import os
import tensorflow as tf

from data.ssm import lin_gaussian_ssm


def lin_gaussian_ssm_loader(params, mode='trn'):
    return lin_gaussian_ssm(
        num_samples=params['NUM_SAMPLES'][mode], seq_len=params['SEQ_LEN'],
        A=params['SSM_A'], C=params['SSM_C'], Q=params['SSM_Q'], R=params['SSM_R'],
        seed=params['DATA_SEEDS'][mode])


def get_default_callbacks(params):
    nn_cb = tf.keras.callbacks.TerminateOnNaN()

    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(params['MODEL_PATH'], 'logs'),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        write_steps_per_second=True,
        update_freq='batch',
        profile_batch=1
    )

    cp_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(params['MODEL_PATH'], 'ckpt'),
        save_freq='epoch',
    )

    return [nn_cb, tb_cb, cp_cb]