import tensorflow as tf
from layers import conv2d, dense, flatten, activation
from sn_layers import sn_dense
from blocks import sn_conv_block
from models import Discriminator as D


def discriminator_block(x,
                        is_training,
                        filters,
                        activation_='lrelu',
                        kernel_size=(3, 3),
                        residual=True):
    with tf.variable_scope(None, discriminator_block.__name__):
        _x = sn_conv_block(x,
                           is_training,
                           filters,
                           activation_,
                           kernel_size,
                           'same',
                           0.,)
        _x = sn_conv_block(_x,
                           is_training,
                           filters,
                           None,
                           kernel_size,
                           'same',
                           0., )
        if residual:
            _x += x
        _x = activation(_x, 'lrelu')

        return _x


class Discriminator(D):
    def __call__(self, x, reuse=True, is_feature=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _x = sn_conv_block(x,
                               self.is_training,
                               filters=32,
                               activation_='lrelu',
                               kernel_size=(4, 4),
                               sampling='down')

            for i in range(2):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=32,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         residual=True)
            _x = sn_conv_block(_x,
                               self.is_training,
                               filters=64,
                               activation_='lrelu',
                               kernel_size=(4, 4),
                               sampling='down')

            for i in range(4):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=64,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         residual=True)

            _x = sn_conv_block(_x,
                               self.is_training,
                               filters=128,
                               activation_='lrelu',
                               kernel_size=(4, 4),
                               sampling='down')

            for i in range(4):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=128,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         residual=True)

            _x = sn_conv_block(_x,
                               self.is_training,
                               filters=256,
                               activation_='lrelu',
                               kernel_size=(4, 4),
                               sampling='down')

            for i in range(4):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=256,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         residual=True)

            _x = sn_conv_block(_x,
                               self.is_training,
                               filters=512,
                               activation_='lrelu',
                               kernel_size=(4, 4),
                               sampling='down')

            for i in range(4):
                _x = discriminator_block(_x,
                                         self.is_training,
                                         filters=512,
                                         activation_='lrelu',
                                         kernel_size=(3, 3),
                                         residual=True)

            _x = sn_conv_block(_x,
                               self.is_training,
                               filters=1024,
                               activation_='lrelu',
                               kernel_size=(4, 4),
                               sampling='down')

            if is_feature:
                return _x

            _x = flatten(_x)
            _x = sn_dense(_x,
                          self.is_training,
                          units=1,
                          activation_=None)
            return _x

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]