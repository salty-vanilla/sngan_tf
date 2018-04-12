from tensorflow.python.keras.preprocessing.image import Iterator
import os
import numpy as np
from PIL import Image


class ImageSampler:
    def __init__(self, image_dir,
                 is_flip=True,
                 target_size=None,
                 color_mode='rgb',
                 normalize_mode='tanh',
                 nb_sample=None):
        self.image_dir = image_dir
        self.is_flip = is_flip
        self.target_size = target_size
        self.color_mode = color_mode
        self.image_paths = np.array(sorted([path for path in get_image_paths(image_dir)]))
        self.normalize_mode = normalize_mode
        if nb_sample is not None:
            self.image_paths = self.image_paths[:nb_sample]
        self.nb_sample = len(self.image_paths)

    def flow_from_directory(self, batch_size, shuffle=True, seed=None):
        return DataIterator(paths=self.image_paths,
                            is_flip=self.is_flip, target_size=self.target_size,
                            color_mode=self.color_mode, batch_size=batch_size,
                            normalize_mode=self.normalize_mode, shuffle=shuffle, seed=seed)


class DataIterator(Iterator):
    def __init__(self, paths,
                 is_flip,
                 target_size,
                 color_mode,
                 batch_size,
                 normalize_mode,
                 shuffle,
                 seed):
        self.paths = paths
        self.is_flip = is_flip
        self.target_size = target_size
        self.color_mode = color_mode
        self.nb_sample = len(self.paths)
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.current_paths = None

    def __call__(self, *args, **kwargs):
        with self.lock:
            index_array = next(self.index_generator)
        image_path_batch = self.paths[index_array]
        image_batch = np.array([load_image(path, self.is_flip, self.target_size, self.color_mode)
                                for path in image_path_batch])
        self.current_paths = image_path_batch
        return image_batch

    def data_to_image(self, x):
        return denormalize(x, self.normalize_mode)


def load_image(path, is_flip=True, target_size=None, color_mode='rgb'):
    assert color_mode in ['grayscale', 'gray', 'rgb']
    image = Image.open(path)

    if color_mode in ['grayscale', 'gray']:
        image = image.convert('L')

    if target_size is not None and target_size != image.size:
        image = image.resize(target_size, Image.BILINEAR)

    image_array = np.asarray(image)
    image_array = normalize(image_array)

    if len(image_array.shape) == 2:
        image_array = np.expand_dims(image_array, axis=-1)

    #
    if is_flip:
        if np.random.uniform() < 0.5:
            image_array = image_array[:, ::-1, :]
        else:
            pass
    return image_array


def normalize(x, mode='tanh'):
    if mode == 'tanh':
        return (x.astype('float32') / 255 - 0.5) / 0.5
    elif mode == 'sigmoid':
        return x.astype('float32') / 255
    else:
        raise NotImplementedError


def denormalize(x, mode='tanh'):
    if mode == 'tanh':
        return ((x + 1.) / 2 * 255).astype('uint8')
    elif mode == 'sigmoid':
        return (x * 255).astype('uint8')
    else:
        raise NotImplementedError


def get_image_paths(src_dir):
    def get_all_paths():
        for root, dirs, files in os.walk(src_dir):
            yield root
            for file in files:
                yield os.path.join(root, file)

    def is_image(path):
        if 'png' in path or 'jpg' in path or 'bmp' in path:
            return True
        else:
            return False

    return [path for path in get_all_paths() if is_image(path)]