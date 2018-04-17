import tensorflow as tf


class Discriminator:
    def __init__(self, input_shape,
                 normalization=None):
        self.input_shape = input_shape
        self.name = 'model/discriminator'
        self.normalization = normalization
        with tf.variable_scope(self.name):
            self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.conv_kwargs = {'activation_': 'lrelu'}

    def __call__(self, x, reuse=True, is_feature=False):
        raise NotImplementedError

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


class Generator:
    def __init__(self, noise_dim,
                 last_activation='tanh',
                 color_mode='rgb',
                 normalization='batch',
                 upsampling='deconv'):
        self.noise_dim = noise_dim
        self.last_activation = last_activation
        self.name = 'model/generator'
        assert color_mode in ['grayscale', 'gray', 'rgb']
        self.channel = 1 if color_mode in ['grayscale', 'gray'] else 3
        self.normalization = normalization
        self.upsampling = upsampling
        with tf.variable_scope(self.name):
            self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.conv_kwargs = {'activation_': 'relu',
                            'normalization': self.normalization}

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]
