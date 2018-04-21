import tensorflow as tf
from layers import activation
from tensorflow.python.layers.base import Layer


def l2_normalize(x, eps=1e-12):
    return x / (tf.reduce_sum(x**2)**0.5 + eps)


def get_max_singular_value(w, u, ip=1, eps=1e-12):
    with tf.variable_scope('PowerIteration'):
        _u = u
        _v = None
        for _ in range(ip):
            _v = tf.matmul(_u, w)
            _v = l2_normalize(_v, eps)
            _u = tf.matmul(_v, tf.transpose(w))
            _u = l2_normalize(_u, eps)
        sigma = tf.matmul(_u,
                          tf.matmul(w, tf.transpose(_v)))
    return sigma, _u


class SNConv2d(Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation_='relu',
                 is_training=True):
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
        self.w = self.add_variable('kernel',
                                   (self.kernel_size[1], self.kernel_size[0], input_shape[-1], self.filters),
                                   tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.02),
                                   trainable=True)
        self.bias = self.add_variable('bias',
                                      (self.filters,),
                                      tf.float32,
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True)
        self.u = self.add_variable('u',
                                   (1, self.filters),
                                   tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.02),
                                   trainable=False)
        self.w_sn = self.add_variable('kernel_sn',
                                      (self.kernel_size[1], self.kernel_size[0], input_shape[-1], self.filters),
                                      tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.02),
                                      trainable=False)
        self.built = True

    def call(self, x, *args, **kwargs):
        with tf.variable_scope(self.scope_name) as vs:
            if not self.is_training:
                vs.reuse_variables()

            control_inputs = []
            if self.is_training:
                w_mat = tf.transpose(self.w, (3, 2, 0, 1))
                w_mat = tf.reshape(w_mat,
                                   (w_mat.shape[0], -1))
                sigma, _u = get_max_singular_value(w_mat, self.u)
                w = self.w / sigma

                control_inputs.append(tf.assign(self.u, _u))
                control_inputs.append(tf.assign(self.w_sn, w))
            else:
                w = self.w_sn
            with tf.control_dependencies(control_inputs):
                _h = tf.nn.bias_add(tf.nn.conv2d(x,
                                                 w,
                                                 strides=self.strides,
                                                 padding=self.padding),
                                    self.bias)
                return activation(_h, self.activation)


class SNDense(Layer):
    def __init__(self,
                 units,
                 activation_='relu',
                 is_training=True):
        self.is_training = is_training
        self.units = units
        self.activation = activation_
        super().__init__()

    def build(self, input_shape):
        self._input_shape = input_shape
        self.w = self.add_variable('kernel',
                                   (input_shape[-1], self.units),
                                   tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.02),
                                   trainable=True)
        self.bias = self.add_variable('bias',
                                      (self.units,),
                                      tf.float32,
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True)
        self.u = self.add_variable('u',
                                   (1, self.units),
                                   tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.02),
                                   trainable=False)
        self.w_sn = self.add_variable('kernel_sn',
                                      (input_shape[-1], self.units),
                                      tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.02),
                                      trainable=False)
        self.built = True

    def call(self, x, *args, **kwargs):
        with tf.variable_scope(self.scope_name) as vs:
            if not self.is_training:
                vs.reuse_variables()
            control_inputs = []
            if self.is_training:
                w_mat = tf.transpose(self.w)
                sigma, _u = get_max_singular_value(w_mat, self.u)
                w = self.w / sigma

                control_inputs.append(tf.assign(self.u, _u))
                control_inputs.append(tf.assign(self.w_sn, w))
            else:
                w = self.w_sn
        with tf.control_dependencies(control_inputs):
            _h = tf.nn.bias_add(tf.matmul(x, w),
                                self.bias)
            return activation(_h, self.activation)


def sn_conv2d(x,
              filters,
              kernel_size=(3, 3),
              strides=(1, 1),
              padding='same',
              activation_='relu',
              is_training=True):
    return SNConv2d(filters,
                    kernel_size,
                    strides,
                    padding,
                    activation_,
                    is_training)(x)


def sn_dense(x,
             units,
             activation_='relu',
             is_training=True):
    return SNDense(units,
                   activation_,
                   is_training)(x)
