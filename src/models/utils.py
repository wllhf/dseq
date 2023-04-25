import tensorflow as tf


def get_mlp(dims_hidden, dim_out, activation='linear', name='dense_net'):
    layers = [tf.keras.layers.Dense(d, activation='relu') for d in dims_hidden]
    layers.append(tf.keras.layers.Dense(dim_out, activation=activation))
    return tf.keras.Sequential(layers, name=name)


def get_gaussian_diag_model(base, input_shape, dim_gaussian, name='gaussian_diag'):
    """
    args:
      base: Keras layer or model.
      input_shape: Shape object describing the input tensor shape.
      dim_gaussian: Integer, dimensionality of the Gaussian
      name: Sting, Model name
    """
    input = tf.keras.layers.Input(input_shape)
    h = input if base is None else base(input)
    mean = tf.keras.layers.Dense(dim_gaussian, name='dense_mean')(h)
    cov = tf.keras.layers.Dense(dim_gaussian, activation='softplus', name='dense_cov')(h)
    return tf.keras.Model(inputs=input, outputs=[mean, cov], name=name)