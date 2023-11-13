import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_determinism(seed=0):
    set_seed(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


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

    # return [nn_cb, tb_cb, cp_cb]
    return [nn_cb, tb_cb]