'''
This is a nerual nets base function upon which the GAN model was built.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_impl

class NN_Base(object):
  def __init__(self, is_training, batch_norm_momentum = 0.999,
                 batch_norm_epsilon = 0.001):
    self._batch_norm_momentum = batch_norm_momentum
    self._batch_norm_epsilon = batch_norm_epsilon
    self._is_training = is_training
    
  def forward_pass(self, x):
    raise NotImplementedError(
          'forward_pass() is implemented in Model sub classes')
    
  def _batch_norm(self, x):
    x = tf.layers.batch_normalization(
      x,
      axis = -1,
      momentum=self._batch_norm_momentum,
      epsilon=self._batch_norm_epsilon,
      center=True,
      scale=True,
      training=self._is_training)
    return x
    
  def _relu(self, x):
    return tf.nn.relu(x)
  
  def _leakyrelu(self, x, leak = 0.2, name = "lrelu"):
    with tf.name_scope(name):
      f1 = 0.5 * (1 + leak)
      f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)
  
  def _softplus(self, x):
    return tf.nn.softplus(x, name = 'softplus')
  
  def _fully_connected(self, x, out_dim, use_bias=True):
    x = tf.layers.dense(x, out_dim, use_bias = use_bias)
    return x
  
  def _drop_out(self, x, rate = 0.5):
    return tf.layers.dropout(x, rate = rate, training = self._is_training)
  
  def _add_noise(self, inputs, mean = 0.0, stddev = 0.001):
    with tf.name_scope('Add_Noise'):
      noise = tf.random_normal(shape = tf.shape(inputs),
                              mean = mean,
                              stddev = stddev,
                              dtype = inputs.dtype,
                              name = 'noise'
                            )
      inputs = inputs + noise
    return inputs
  
  def _dense_WN(self,
        inputs, units,
        activation=None,
        weight_norm=True,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    '''
    Dense layer using weight normalizaton
    '''
    layer = Dense(units,
                  activation=activation,
                  weight_norm=weight_norm,
                  use_bias=use_bias,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer,
                  activity_regularizer=activity_regularizer,
                  kernel_constraint=kernel_constraint,
                  bias_constraint=bias_constraint,
                  trainable=trainable,
                  name=name,
                  dtype=inputs.dtype.base_dtype,
                  _scope=name,
                  _reuse=reuse)
    return layer.apply(inputs)

  
  
class Dense(core_layers.Dense):
  '''
  Dense layer implementation using weight normalization.
  Code borrowed from:
  https://github.com/llan-ml/weightnorm/blob/master/dense.py
  '''
  def __init__(self, *args, **kwargs):
      self.weight_norm = kwargs.pop("weight_norm")
      super(Dense, self).__init__(*args, **kwargs)

  def build(self, input_shape):
      input_shape = tensor_shape.TensorShape(input_shape)
      if input_shape[-1].value is None:
          raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
      self.input_spec = base.InputSpec(
            min_ndim=2, axes={-1: input_shape[-1].value})
      kernel = self.add_variable(
            'kernel',
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
      if self.weight_norm:
          self.g = self.add_variable(
                "wn/g",
                shape=(self.units,),
                initializer=init_ops.ones_initializer(),
                dtype=kernel.dtype,
                trainable=True)
          self.kernel = nn_impl.l2_normalize(kernel, dim=0) * self.g
      else:
          self.kernel = kernel
      if self.use_bias:
          self.bias = self.add_variable(
                'bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
      else:
          self.bias = None
      self.built = True



if __name__ == "__main__":
  tf.reset_default_graph()
  is_training = tf.placeholder(tf.bool)
  model = NN_Base(is_training)
  inputs = tf.random_normal((128, 797))
  outputs = model._dense_WN(inputs, 500)
  with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), \
              tf.local_variables_initializer()])
    inputs_o, outputs_o = sess.run([inputs, outputs], \
                                   feed_dict = {is_training: False})