import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, PReLU, ReLU

TRAIN_DIM = 96


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 255.0
    mse = tf.keras.losses.MeanSquaredError()
    return 10.0 * tf_log10((max_pixel ** 2) / (mse(y_true, y_pred)))


def fsrcnn(d=56, s=12, m=4, input_shape=(None, None, 1), scale_factor=2):
    model = tf.keras.Sequential()

    # features
    model.add(Conv2D(d, 5, input_shape=input_shape, padding="same"))
    model.add(ReLU())

    # shrinking
    model.add(Conv2D(s, 1, padding="same"))
    model.add(ReLU())

    # mapping
    for i in range(m):
        model.add(Conv2D(s, 3, padding="same"))
        model.add(ReLU())

    # expanding
    model.add(Conv2D(d, 1, padding="same"))
    model.add(ReLU())

    model.add(Conv2DTranspose(1, 9, strides=scale_factor, padding="same"))

    model.compile(optimizer="adam", loss="mse", metrics=[PSNR, "accuracy"])
    return model
