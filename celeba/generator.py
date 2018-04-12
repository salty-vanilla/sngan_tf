import os
import sys
import tensorflow as tf
sys.path.append(os.getcwd())
from models import Generator as G
from layers import dense, reshape, batch_norm, activation, conv2d
from blocks import residual_block, conv_block


class Generator(G):
    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _x = dense(x, 16*16*1024, activation_=None)
            _x = reshape(_x, (16, 16, 1024))
            _x = batch_norm(_x, is_training=self.is_training)
            _x = activation(_x, 'relu')

            _x = conv2d(_x, filters=64, kernel_size=(1, 1), activation_='relu')
            residual_inputs = _x

            with tf.name_scope('residual_blocks'):
                for i in range(16):
                    _x = residual_block(_x,
                                        is_training=self.is_training,
                                        filters=64,
                                        activation_='relu',
                                        sampling='same',
                                        normalization=self.normalization,
                                        dropout_rate=0.,
                                        mode='conv_first')

            _x = conv_block(_x,
                            is_training=self.is_training,
                            filters=64,
                            activation_='relu',
                            sampling='same',
                            normalization=self.normalization,
                            dropout_rate=0.,
                            mode='conv_first')
            _x += residual_inputs

            with tf.name_scope('upsampling_blocks'):
                for i in range(3):
                    _x = conv_block(_x,
                                    is_training=self.is_training,
                                    filters=64,
                                    activation_='relu',
                                    sampling=self.upsampling,
                                    normalization=self.normalization,
                                    dropout_rate=0.,
                                    mode='conv_first')

            _x = conv_block(_x,
                            is_training=self.is_training,
                            filters=self.channel,
                            activation_=self.last_activation,
                            sampling='same',
                            normalization=None,
                            dropout_rate=0.,
                            mode='conv_first')
            return _x
