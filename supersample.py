import cv2
import sys

import tensorflow as tf

from fsrcnn import PSNR, fsrcnn


def main():
    old_model = tf.keras.models.load_model(
        f"{sys.argv[1]}/checkpoint-relu", custom_objects={"PSNR": PSNR}
    )
    img = cv2.imread(sys.argv[2])

    height = img.shape[0]
    width = img.shape[1]

    (b, g, r) = cv2.split(img)

    model = fsrcnn(input_shape=(None, None, 1))
    model.set_weights(old_model.get_weights())

    b = b.reshape(1, height, width, 1)
    g = g.reshape(1, height, width, 1)
    r = r.reshape(1, height, width, 1)

    with tf.device("/CPU:0"):
        b_pred = model(b)
        g_pred = model(g)
        r_pred = model(r)

    height *= 2
    width *= 2

    pred_image = cv2.merge(
        [
            b_pred.numpy().reshape(height, width),
            g_pred.numpy().reshape(height, width),
            r_pred.numpy().reshape(height, width),
        ]
    )

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC).astype(
        "float32"
    )

    cv2.imwrite("model_out.png", pred_image)
    cv2.imwrite("cubic_out.png", img)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python supersample.py training_dir /path/to/image.ext")
    else:
        main()
