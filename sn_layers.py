import tensorflow as tf
from layers import activation
from tensorflow.python.layers.base import Layer


def get_max_singular_value(w, u, ip=1, eps=1e-12):
    with tf.variable_scope('PowerIteration'):
        _u = u
        for _ in range(ip):
            _v = tf.matmul(_u, w)
            _v = _v / (tf.reduce_sum(_v**2)**0.5 + eps)
            _u = tf.matmul(_v, tf.transpose(w))
            _u = _u / (tf.reduce_sum(_u**2)**0.5 + eps)

        sigma = tf.reduce_sum((tf.matmul(_u, w) * _v))
    return sigma, _u


class SNConv2d(Layer):
    def __init__(self,
                 is_training,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation_='relu'):
        self.is_training = is_training
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = [1, strides[1], strides[0], 1]
        self.activation = activation_

        if padding in ['same', 'SAME']:
            self.padding = 'SAME'
        elif padding in ['valid', 'VALID']:
            self.padding = 'VALID'
        else:
            raise NotImplementedError

        super().__init__()

    def build(self, input_shape):
        self._input_shape = input_shape
        self.w = self.add_variable('weight',
                                   (self.kernel_size[1], self.kernel_size[0], input_shape[-1], self.filters),
                                   tf.float32,
                                   trainable=True)
        self.bias = self.add_variable('bias',
                                      (self.filters,),
                                      tf.float32,
                                      trainable=True)
        self.u = self.add_variable('u',
                                   (1, self.filters),
                                   tf.float32,
                                   initializer=tf.initializers.random_normal,
                                   trainable=False)
        self.w_sn = self.add_variable('weight_sn',
                                      (self.kernel_size[1], self.kernel_size[0], input_shape[-1], self.filters),
                                      tf.float32,
                                      trainable=False)
        self.built = True

    def call(self, x):
        return tf.cond(
            self.is_training,
            lambda: self._layer(x, is_training=True, reuse=None),
            lambda: self._layer(x, is_training=False, reuse=True),
        )

    def _layer(self, x, is_training, reuse):
        with tf.variable_scope(self.scope_name, reuse=reuse) as vs:
            control_inputs = []
            if is_training:
                w_mat = tf.transpose(self.w, (3, 2, 0, 1))
                w_mat = tf.reshape(w_mat,
                                   (w_mat.shape[0], w_mat.shape[1] * w_mat.shape[2] * w_mat.shape[3]))
                sigma, _u = get_max_singular_value(w_mat, self.u)

                control_inputs.append(tf.assign(self.u, _u))
                control_inputs.append(tf.assign(self.w_sn, self.w / sigma))

            else:
                pass

            with tf.control_dependencies(control_inputs):
                _h = tf.nn.bias_add(tf.nn.conv2d(x,
                                                 self.w_sn,
                                                 strides=self.strides,
                                                 padding=self.padding),
                                    self.bias)
                return activation(_h, self.activation)


class SNDense(Layer):
    def __init__(self,
                 is_training,
                 units,
                 activation_='relu'):
        self.is_training = is_training
        self.units = units
        self.activation = activation_
        super().__init__()

    def build(self, input_shape):
        self._input_shape = input_shape
        self.w = self.add_variable('weight',
                                   (input_shape[-1], self.units),
                                   tf.float32,
                                   trainable=True)
        self.bias = self.add_variable('bias',
                                      (self.units,),
                                      tf.float32,
                                      trainable=True)
        self.u = self.add_variable('u',
                                   (1, self.units),
                                   tf.float32,
                                   initializer=tf.initializers.random_normal,
                                   trainable=False)
        self.w_sn = self.add_variable('weight_sn',
                                      (input_shape[-1], self.units),
                                      tf.float32,
                                      trainable=False)
        self.built = True

    def call(self, x):
        return tf.cond(
            self.is_training,
            lambda: self._layer(x, is_training=True, reuse=None),
            lambda: self._layer(x, is_training=False, reuse=True)
        )

    def _layer(self, x, is_training, reuse):
        with tf.variable_scope('SpectralNormalization', reuse=reuse):
            control_inputs = []
            if is_training:
                w_mat = tf.transpose(self.w)
                sigma, _u = get_max_singular_value(w_mat, self.u)

                control_inputs.append(tf.assign(self.u, _u))
                control_inputs.append(tf.assign(self.w_sn, self.w / sigma))
            else:
                pass
        with tf.control_dependencies(control_inputs):
            _h = tf.nn.bias_add(tf.matmul(x, self.w_sn),
                                self.bias)
            return activation(_h, self.activation)


def sn_conv2d(x,
              is_training,
              filters,
              kernel_size=(3, 3),
              strides=(1, 1),
              activation_='relu',
              padding='same'):
    return SNConv2d(is_training,
                    filters,
                    kernel_size,
                    strides,
                    padding,
                    activation_)(x)


def sn_dense(x,
             is_training,
             units,
             activation_='relu'):
    return SNDense(is_training,
                   units,
                   activation_)(x)
