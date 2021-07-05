import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, PReLU


def fsrcnn(d=56, s=12, m=4):
    model = tf.keras.Sequential()

    # features
    model.add(Conv2D(d, 5, input_shape=(28, 28, 3)))
    model.add(PReLU())

    # shrinking
    model.add(Conv2D(s, 1))
    model.add(PReLU())

    # mapping
    for i in range(m):
        model.add(Conv2D(s, 3))
        model.add(PReLU())

    # expanding
    model.add(Conv2D(d, 1))
    model.add(PReLU())

    model.add(Conv2DTranspose(1, 9))

    model.summary()
    return model
