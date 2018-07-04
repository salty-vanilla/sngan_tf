from PIL import Image
import numpy as np


def save_images(file_path, images, rows=None, cols=None):
    nb_image = len(images)

    if rows is None or cols is None:
        rows = int(np.sqrt(nb_image))
        cols = int(np.sqrt(nb_image))

    _images = np.array(images)
    _images = _images.reshape(rows, cols, *_images.shape[1:])

    __images = np.array([np.concatenate(im, axis=1) for im in _images])
    __images = np.concatenate(__images, axis=0)

    Image.fromarray(__images).save(file_path)
