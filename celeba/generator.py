import os
import sys
import tensorflow as tf
import numpy as np
sys.path.append(os.getcwd())
from models import Generator as G
from layers import dense, reshape, batch_norm, layer_norm, \
    activation, conv2d, conv2d_transpose, subpixel_conv2d
from blocks import residual_block, conv_block


def first_block(x,
                target_size,
                noise_dim,
                upsampling='deconv',
                normalization='batch',
                is_training=True):
    if upsampling == 'deconv':
        _x = reshape(x, (1, 1, noise_dim))
        _x = conv2d_transpose(_x, 1024, target_size, strides=(1, 1), padding='valid')
    elif upsampling == 'dense':
        _x = dense(x, target_size[0]*target_size[1]*1024)
        _x = reshape(_x, (target_size[1], target_size[0], 1024))
    else:
        raise ValueError

    if normalization == 'batch':
        _x = batch_norm(_x, is_training=is_training)
    elif normalization == 'layer':
        _x = layer_norm(_x, is_training=is_training)
    elif normalization is None:
        pass
    else:
        raise ValueError
    _x = activation(_x, 'relu')
    return _x


class ResidualGenerator(G):
    def __call__(self, x, reuse=False, is_training=True):
        nb_upsampling = int(np.log2(self.target_size[0] // 16))
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _x = first_block(x,
                             (16, 16),
                             self.noise_dim,
                             'deconv',
                             self.normalization,
                             is_training)

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
                for i in range(nb_upsampling):
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
        nb_upsampling = int(np.log2(self.target_size[0] // 4))
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            _x = first_block(x,
                             (4, 4),
                             self.noise_dim,
                             'dense',
                             self.normalization,
                             is_training)

            for i in range(nb_upsampling):
                with tf.variable_scope(None, 'conv_blocks'):
                    filters = 1024 // (2**(i+1))
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
