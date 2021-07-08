import cv2
import sys

import tensorflow as tf

from fsrcnn import PSNR, fsrcnn


def main():
    old_model = tf.keras.models.load_model(
        f"{sys.argv[1]}/checkpoint-relu", custom_objects={"PSNR": PSNR}
    )
    model = fsrcnn(input_shape=(None, None, 1))
    model.set_weights(old_model.get_weights())
    filename = sys.argv[2]

    pred_image, img = upscale(filename, model)
    cv2.imwrite("model_out.png", pred_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite("cubic_out.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def upscale(filename, model):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    height = img.shape[0]
    width = img.shape[1]

    channels = []
    for channel in cv2.split(img):
        channel = channel.reshape(1, height, width, 1)
        with tf.device("/CPU:0"):
            channels.append(model(channel).numpy().reshape(height * 2, width * 2))

    height *= 2
    width *= 2

    pred_image = cv2.merge(channels)

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC).astype(
        "float32"
    )
    return (pred_image, img)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python supersample.py training_dir /path/to/image.ext")
    else:
        main()
