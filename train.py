from fsrcnn import fsrcnn, PSNR

import tensorflow as tf

import sys
import loader

from utils import images_to_chunked_pairs

EPOCHS = 50

# 60 - 20 - 20
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2
TRAIN_FRACTION = 1 - TEST_FRACTION - VALIDATION_FRACTION


def load_data(data_dir, chunk=True):
    """
    Load image data from directory `data_dir`.

    Return tensorflow dataset of image and labels (individual channel).
    """
    if chunk:
        print("Splitting images...")
        images_to_chunked_pairs(data_dir)
        print("Done.")
    return loader.load(data_dir)


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python train.py data_directory")

    # Get image and label dataset
    ds = load_data(sys.argv[1], chunk=False).batch(20)

    train_size = int(TRAIN_FRACTION * len(ds))
    val_size = int(VALIDATION_FRACTION * len(ds))
    test_size = int(TEST_FRACTION * len(ds))

    train_dataset = ds.take(train_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = ds.skip(train_size)
    val_dataset = test_dataset.skip(val_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.take(test_size).prefetch(tf.data.AUTOTUNE)

    # Get a compiled neural network
    # model = tf.keras.models.load_model(
    #    f"{sys.argv[1]}/checkpoint-relu", custom_objects={"PSNR": PSNR}
    # )
    # model.compile(optimizer="adam", loss="mse", metrics=[PSNR, "accuracy"])
    model = fsrcnn()
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(cooldown=15, verbose=1)
    checkpoint_filepath = f"./{sys.argv[1]}/checkpoint-ssim"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[lr_callback, model_checkpoint_callback],
    )

    model.evaluate(test_dataset, verbose=2)


if __name__ == "__main__":
    main()
