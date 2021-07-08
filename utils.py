import cv2
import os
import glob

from pathlib import Path
from fsrcnn import TRAIN_DIM


def split_image(data_dir, img_name_stem, img, square_size, stride_factor):
    i = 0
    for r in range(0, img.shape[0], square_size // stride_factor):
        j = 0
        for c in range(0, img.shape[1], square_size // stride_factor):
            if img[r : r + square_size, c : c + square_size, :].shape == (
                square_size,
                square_size,
                3,
            ):
                cv2.imwrite(
                    f"{data_dir}/{img_name_stem}-{i}_{j}.png",
                    img[r : r + square_size, c : c + square_size, :],
                )
            j += 1
        i += 1


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

        split_image(
            data_dir + "/split-image", Path(filename).stem, train_image, TRAIN_DIM, 4
        )
        split_image(
            data_dir + "/split-label", Path(filename).stem, image, TRAIN_DIM * 2, 4
        )
    # delete files that aren't common (edge case when file size is odd and rounding half the dimension
    # is an exact multiple of TRAIN_DIM)
    image_names = set(
        map(lambda x: x.split(os.sep)[-1], glob.glob(f"{data_dir}/split-image/*"))
    )
    label_names = set(
        map(lambda x: x.split(os.sep)[-1], glob.glob(f"{data_dir}/split-label/*"))
    )
    for p in image_names.symmetric_difference(label_names):
        try:
            os.remove(f"{data_dir}/split-image/{p}")
        except FileNotFoundError:
            os.remove(f"{data_dir}/split-label/{p}")
