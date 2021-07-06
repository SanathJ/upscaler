from fsrcnn import TRAIN_DIM, fsrcnn, PSNR

import tensorflow as tf
import numpy as np

import glob
import cv2
import os
import sys

from sklearn.model_selection import train_test_split
from utils import images_to_chunked_pairs, preprocess
from pathlib import Path

EPOCHS = 70

# 60 - 20 - 20
TEST_SIZE = 0.4
VALIDATION_SPLIT = 0.25


def load_data(data_dir, chunk=True):
    """
    Load image data from directory `data_dir`.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    if chunk:
        images_to_chunked_pairs(data_dir)

    # splits image/labels into separate channels and appends them to their lists
    image_names = glob.glob(str(Path(data_dir, "split-image")) + "/*")
    label_names = glob.glob(str(Path(data_dir, "split-label")) + "/*")
    for (image_name, label_name) in zip(image_names, label_names):
        assert Path(image_name).stem == Path(label_name).stem
        image_name = str(Path(image_name))
        label_name = str(Path(label_name))

        image = cv2.imread(image_name)
        images.extend(cv2.split(image))
        label = cv2.imread(label_name)
        labels.extend(cv2.split(label))

    return (images, labels)


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python train.py data_directory")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    y_train = preprocess(y_train)
    y_test = preprocess(y_test)

    # Get a compiled neural network
    # model = tf.keras.models.load_model("train/model", custom_objects={"PSNR": PSNR})
    # model.compile(optimizer="adam", loss="mse", metrics=[PSNR, "accuracy"])
    model = fsrcnn()

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(cooldown=15, verbose=1)
    checkpoint_filepath = f"./{sys.argv[1]}/checkpoint"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    # print(type(x_train), len(y_train))
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        verbose=1,
        callbacks=[lr_callback, model_checkpoint_callback],
    )

    model.save(os.path.join(sys.argv[1], "model"))


if __name__ == "__main__":
    main()
