import cv2
import os

from pathlib import Path
from fsrcnn import TRAIN_DIM


def split_image(data_dir, img_name_stem, img, square_size):
    for r in range(0, img.shape[0], square_size):
        for c in range(0, img.shape[1], square_size):
            if img[r : r + square_size, c : c + square_size, :].shape == (
                square_size,
                square_size,
                3,
            ):
                cv2.imwrite(
                    f"{data_dir}/{img_name_stem}-{r}_{c}.png",
                    img[r : r + square_size, c : c + square_size, :],
                )


def images_to_chunked_pairs(data_dir):
    """
    downsample image to label

    split image and labels into TRAIN_DIMxTRAIN_DIM chunks
    """
    Path(data_dir, "split-image").mkdir(exist_ok=True)
    Path(data_dir, "split-label").mkdir(exist_ok=True)
    Path(data_dir, "original").mkdir(exist_ok=True)
    for filename in os.listdir(Path(data_dir, "original")):
        image = cv2.imread(os.path.join(data_dir, "original", filename))
        train_image = cv2.resize(
            image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )
        train_image = cv2.resize(
            train_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )

        split_image(
            data_dir + "/split-image",
            Path(filename).stem,
            train_image,
            TRAIN_DIM,
        )
        split_image(
            data_dir + "/split-label",
            Path(filename).stem,
            image,
            TRAIN_DIM,
        )


def preprocess(x):
    x = x.reshape(x.shape[0], TRAIN_DIM, TRAIN_DIM, 1)
    x = x.astype("float32")
    # x = x / 255.0
    return x
