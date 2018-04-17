import tensorflow as tf
from layers import conv2d, dense, flatten, activation, global_average_pool2d
from sn_layers import sn_dense
from blocks import conv_block
from models import Discriminator as D


def discriminator_block(x,
                        is_training,
                        filters,
                        activation_='lrelu',
                        kernel_size=(3, 3),
                        normalization='spectral',
                        residual=True):
    with tf.variable_scope(None, discriminator_block.__name__):
        _x = conv_block(x,
                        is_training,
                        filters,
                        activation_,
                        kernel_size,
                        'same',
                        normalization,
                        0.,)
        _x = conv_block(_x,
                        is_training,
                        filters,
                        None,
                        kernel_size,
                        'same',
                        normalization,
                        0.,)
        if residual:
            _x += x
        _x = activation(_x, 'lrelu')

        return _x


class ResidualDiscriminator(D):
    def __call__(self, x, reuse=True, is_feature=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _x = conv_block(x,
                            self.is_training,
                            filters=32,
                            activation_='lrelu',
                            kernel_size=(4, 4),
                            sampling='down',
                            normalization=self.normalization)

            for i in range(2):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=32,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         normalization=self.normalization,
                                         residual=True)
            _x = conv_block(_x,
                            self.is_training,
                            filters=64,
                            activation_='lrelu',
                            kernel_size=(4, 4),
                            sampling='down',
                            normalization=self.normalization)

            for i in range(4):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=64,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         normalization=self.normalization,
                                         residual=True)

            _x = conv_block(_x,
                            self.is_training,
                            filters=128,
                            activation_='lrelu',
                            kernel_size=(4, 4),
                            sampling='down',
                            normalization=self.normalization)

            for i in range(4):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=128,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         normalization=self.normalization,
                                         residual=True)

            _x = conv_block(_x,
                            self.is_training,
                            filters=256,
                            activation_='lrelu',
                            kernel_size=(4, 4),
                            sampling='down',
                            normalization=self.normalization)

            for i in range(4):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=256,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         normalization=self.normalization,
                                         residual=True)

            _x = conv_block(_x,
                            self.is_training,
                            filters=512,
                            activation_='lrelu',
                            kernel_size=(4, 4),
                            sampling='down',
                            normalization=self.normalization)

            for i in range(4):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=512,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         normalization=self.normalization,
                                         residual=True)

            _x = conv_block(_x,
                            self.is_training,
                            filters=1024,
                            activation_='lrelu',
                            kernel_size=(4, 4),
                            sampling='down',
                            normalization=self.normalization)

            if is_feature:
                return _x

            _x = global_average_pool2d(_x)

            if self.normalization == 'spectral':
                _x = sn_dense(_x,
                              self.is_training,
                              units=1,
                              activation_=None)
            else:
                _x = dense(_x,
                           units=1,
                           activation_=None)
            return _x


class Discriminator(D):
    def __call__(self, x, reuse=True, is_feature=False):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            # if reuse:
            #     vs.reuse_variables()

            _x = x
            first_filters = 32
            for i in range(4):
                filters = first_filters * (2**i)
                _x = conv_block(_x,
                                self.is_training,
                                filters=filters,
                                activation_='lrelu',
                                sampling='same',
                                normalization=self.normalization)
                _x = conv_block(_x,
                                self.is_training,
                                filters=filters,
                                activation_='lrelu',
                                sampling='down',
                                normalization=self.normalization)
            _x = flatten(_x)

            if self.normalization == 'spectral':
                _x = sn_dense(_x,
                              self.is_training,
                              units=1,
                              activation_=None)
            else:
                _x = dense(_x,
                           units=1,
                           activation_=None)
            return _x
