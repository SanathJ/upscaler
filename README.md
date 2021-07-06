# Upscaler

Implements a convolutional neural network to upscale images.

## Usage

### Training

To train the model, run `python train.py <training_dir>`. where `training_dir` is a directory that contains a sub-directory `original` with the images to train on. By default, the program will split the images into 96x96 training images, and 192x192 target images. This can be disabled (if the images have already been split), by setting `chunk=True` as an option to `load_data`. The sub-images will be stored in two directories that can safely be deleted after training. A model is also stored at `training_dir/checkpoint-relu`.

### Upscaling

After training, the model is ready to be used. To do so, run `python supersample.py training_dir /path/to/image.ext`. The model outputs two upscaled images, one using the model, and the other using cubic interpolation.
