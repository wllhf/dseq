import os
import tensorflow as tf


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
    #return [nn_cb, tb_cb]