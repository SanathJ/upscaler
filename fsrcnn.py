import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, PReLU

TRAIN_DIM = 64


def fsrcnn(d=56, s=12, m=4):
    model = tf.keras.Sequential()

    # features
    model.add(Conv2D(d, 5, input_shape=(TRAIN_DIM, TRAIN_DIM, 1), padding="same"))
    model.add(PReLU())

    # shrinking
    model.add(Conv2D(s, 1, padding="same"))
    model.add(PReLU())

    # mapping
    for i in range(m):
        model.add(Conv2D(s, 3, padding="same"))
        model.add(PReLU())

    # expanding
    model.add(Conv2D(d, 1, padding="same"))
    model.add(PReLU())

    model.add(Conv2DTranspose(1, 9, padding="same"))

    model.summary()
    return model
