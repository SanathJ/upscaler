import tensorflow as tf
import pathlib
import os
import cv2
from fsrcnn import TRAIN_DIM

data_dir = ""


def process_path(file_path):
    """
    Given a file_path, returns the decoded image
    at that path and its corresponding label
    """

    label_path = tf.strings.join(
        [f"./{data_dir}/split-label/", tf.strings.split(file_path, os.sep)[-1]]
    )
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=3)

    return image, label


def split_channels0(image, label):
    image_channels = tf.numpy_function(
        cv2.split,
        [image],
        [
            tf.uint8,
            tf.uint8,
            tf.uint8,
        ],
    )
    label_channels = tf.numpy_function(
        cv2.split,
        [label],
        [
            tf.uint8,
            tf.uint8,
            tf.uint8,
        ],
    )
    i = tf.ensure_shape(
        tf.expand_dims(image_channels[0], -1), [TRAIN_DIM, TRAIN_DIM, 1]
    )
    l = tf.ensure_shape(
        tf.expand_dims(label_channels[0], -1), [TRAIN_DIM * 2, TRAIN_DIM * 2, 1]
    )

    return i, tf.cast(l, tf.float32)


def split_channels1(image, label):
    image_channels = tf.numpy_function(
        cv2.split,
        [image],
        [
            tf.uint8,
            tf.uint8,
            tf.uint8,
        ],
    )
    label_channels = tf.numpy_function(
        cv2.split,
        [label],
        [
            tf.uint8,
            tf.uint8,
            tf.uint8,
        ],
    )
    i = tf.ensure_shape(
        tf.expand_dims(image_channels[1], -1), [TRAIN_DIM, TRAIN_DIM, 1]
    )
    l = tf.ensure_shape(
        tf.expand_dims(label_channels[1], -1), [TRAIN_DIM * 2, TRAIN_DIM * 2, 1]
    )

    return i, tf.cast(l, tf.float32)


def split_channels2(image, label):
    image_channels = tf.numpy_function(
        cv2.split,
        [image],
        [
            tf.uint8,
            tf.uint8,
            tf.uint8,
        ],
    )
    label_channels = tf.numpy_function(
        cv2.split,
        [label],
        [
            tf.uint8,
            tf.uint8,
            tf.uint8,
        ],
    )

    i = tf.ensure_shape(
        tf.expand_dims(image_channels[2], -1), [TRAIN_DIM, TRAIN_DIM, 1]
    )
    l = tf.ensure_shape(
        tf.expand_dims(label_channels[2], -1), [TRAIN_DIM * 2, TRAIN_DIM * 2, 1]
    )

    return i, tf.cast(l, tf.float32)


def load(dir):
    global data_dir
    data_dir = dir
    root = pathlib.Path(f"./{data_dir}/split-image/")

    list_ds = tf.data.Dataset.list_files(str(root / "*"))
    images_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    split_ds = images_ds.map(
        split_channels0, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    split_ds = split_ds.concatenate(
        images_ds.map(
            split_channels1, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )
    )
    split_ds = split_ds.concatenate(
        images_ds.map(
            split_channels2, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )
    )

    return split_ds
