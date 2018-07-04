import os
import sys
import tensorflow as tf
sys.path.append(os.getcwd())
from models import Generator as G
from layers import dense, reshape, batch_norm, layer_norm, \
    activation, conv2d, conv2d_transpose, subpixel_conv2d
from blocks import residual_block, conv_block


def first_block(x,
                noise_dim,
                normalization='batch',
                is_training=True):
    _x = reshape(x, (1, 1, noise_dim))
    _x = conv2d_transpose(_x, 1024, (16, 16), strides=(1, 1), padding='valid')

    if normalization == 'batch':
        _x = batch_norm(_x, is_training=is_training)
    elif normalization == 'layer':
        _x = layer_norm(_x, is_training=is_training)
    else:
        raise ValueError
    _x = activation(_x, 'relu')
    return _x


class ResidualGenerator(G):
    def __call__(self, x, reuse=False, is_training=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            _x = first_block(x, self.noise_dim,
                             self.normalization,
                             is_training)

            # _x = reshape(x, (1, 1, self.noise_dim))
            # _x = conv2d_transpose(_x, 1024, (16, 16), strides=(1, 1), padding='valid')
            # if self.normalization == 'batch':
            #     _x = batch_norm(_x, is_training=is_training)
            # _x = activation(_x, 'relu')

            # residual_inputs = conv2d(_x, 64, (1, 1))
            residual_inputs = conv_block(_x,
                                         is_training=is_training,
                                         filters=64,
                                         activation_='relu',
                                         sampling='same',
                                         normalization=self.normalization,
                                         dropout_rate=0.,
                                         mode='conv_first')

            with tf.variable_scope('residual_blocks'):
                for i in range(16):
                    _x = residual_block(_x,
                                        is_training=is_training,
                                        filters=64,
                                        activation_='relu',
                                        sampling='same',
                                        normalization=self.normalization,
                                        dropout_rate=0.,
                                        mode='conv_first')

            _x = conv_block(_x,
                            is_training=is_training,
                            filters=64,
                            activation_='relu',
                            sampling='same',
                            normalization=self.normalization,
                            dropout_rate=0.,
                            mode='conv_first')
            _x += residual_inputs

            with tf.variable_scope('upsampling_blocks'):
                for i in range(3):
                    _x = conv_block(_x,
                                    is_training=is_training,
                                    filters=64,
                                    activation_='relu',
                                    sampling=self.upsampling,
                                    normalization=self.normalization,
                                    dropout_rate=0.,
                                    mode='conv_first')

            __x = _x
            for _ in range(3):
                __x = conv_block(__x,
                                 is_training=is_training,
                                 filters=64,
                                 activation_='relu',
                                 sampling='same',
                                 normalization=self.normalization,
                                 dropout_rate=0.,
                                 mode='conv_first')
            _x += __x
            _x = conv_block(_x,
                            is_training=is_training,
                            kernel_size=(9, 9),
                            filters=self.channel,
                            activation_=self.last_activation,
                            sampling='same',
                            normalization=None,
                            dropout_rate=0.,
                            mode='conv_first')
            return _x


class Generator(G):
    def __call__(self, x, reuse=False, is_training=True):
        with tf.variable_scope(self.name, reuse=reuse) as vs:

            _x = dense(x, 4*4*64*16, activation_=None)
            _x = reshape(_x, (4, 4, 64*16))
            _x = batch_norm(_x, is_training=is_training)
            _x = activation(_x, 'relu')

            for i in range(5):
                with tf.variable_scope(None, 'conv_blocks'):
                    filters = 64*(2**(4 - i))
                    _x = conv_block(_x,
                                    is_training=is_training,
                                    filters=filters,
                                    activation_='relu',
                                    sampling='same',
                                    normalization=self.normalization,
                                    dropout_rate=0.,
                                    mode='conv_first')
                    _x = conv_block(_x,
                                    is_training=is_training,
                                    filters=filters,
                                    activation_='relu',
                                    sampling=self.upsampling,
                                    normalization=self.normalization,
                                    dropout_rate=0.,
                                    mode='conv_first')

            _x = conv_block(_x,
                            is_training=is_training,
                            kernel_size=(9, 9),
                            filters=self.channel,
                            activation_=self.last_activation,
                            sampling='same',
                            normalization=None,
                            dropout_rate=0.,
                            mode='conv_first')
            return _x
