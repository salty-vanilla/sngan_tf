import os
import sys
import tensorflow as tf
sys.path.append(os.getcwd())
from models import Generator as G
from layers import dense, reshape, batch_norm, activation, conv2d
from blocks import residual_block, conv_block


class ResidualGenerator(G):
    def __call__(self, x, reuse=False, is_training=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _x = dense(x, 16*16*64, activation_=None)
            _x = reshape(_x, (16, 16, 64))
            _x = batch_norm(_x, is_training=is_training)
            _x = activation(_x, 'relu')

            residual_inputs = _x

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

            for i in range(4):
                _x = conv_block(_x,
                                is_training=is_training,
                                kernel_size=(3, 3),
                                filters=self.channel,
                                activation_='relu' if i < 4 else self.last_activation,
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
