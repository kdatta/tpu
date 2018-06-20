# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def preconv_kernel(prefix, inputs, kH, kW, dH, dW, data_format, preconv_counter):
  name = prefix + '_preConv_' + str(preconv_counter)
  with tf.name_scope(name) as scope:
      if data_format == 'NCHW': #'channels_first':
          ksize = [1, 1, kH, kW]
          strides = [1, 1, dH, dW]
      else:
          ksize = [1, kH, kW, 1]
          strides = [1, dH, dW, 1]

      return tf.nn.max_pool(inputs,
                            ksize=ksize,
                            strides=strides,
                            padding='VALID',
                            data_format=data_format,
                            name=name)


def conv_kernel(name_prefix, inputs, nIn, nOut, kH, kW, dH, dW, padType, data_format):

  with tf.variable_scope(name_prefix, reuse=tf.AUTO_REUSE):
    if data_format == 'NCHW': #'channels_first':
      strides = [1, 1, dH, dW]
    else:
      strides = [1, dH, dW, 1]
    kernel_shape = [kH, kW, nIn, nOut]
    kernel = tf.get_variable(name_prefix,
                             shape=kernel_shape,
                             initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.Variable(tf.zeros(shape=[nOut], dtype=tf.float32),
                         name='biases',
                         trainable=True)

    conv = tf.nn.conv2d(inputs,
                        kernel,
                        strides,
                        padding=padType,
                        use_cudnn_on_gpu=False,
                        data_format=data_format)
    bias = tf.nn.bias_add(conv,
                          biases,
                          data_format=data_format)
    conv1 = tf.nn.relu(bias)
    return conv1


def max_pool_kernel(name, inputs, kH, kW, dH, dW, data_format):
  with tf.name_scope(name) as scope:
    if data_format == 'NCHW': #'channels_first':

      ksize = [1, 1, kH, kW]
      strides = [1, 1, dH, dW]
    else:
      ksize = [1, kH, kW, 1]
      strides = [1, dH, dW, 1]
    return tf.nn.max_pool(inputs,
                          ksize=ksize,
                          strides=strides,
                          padding='SAME',
                          data_format=data_format,
                          name=name)


def inner_product(prefix, inputs, nIn, nOut):
  name = prefix + '_fc'
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    kernel_shape = [nIn, nOut]
    kernel = tf.get_variable(name+'_kernel',
                             shape=kernel_shape,
                             initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.Variable(tf.zeros(shape=[nOut], dtype=tf.float32),
                         name=name+'_biases',
                         trainable=True)
    return tf.matmul(inputs, kernel) + biases


def pc_par_conv_pool(prefix, inp, kHpc, kWpc, dHpc, dWpc, nIn,
                       nOut, kH, kW, dH, dW, kHp, kWp, dHp, dWp, padType, padding, pV,
                       data_format, preconv_counter):
  # Preconv pooling
  preConv_pool = preconv_kernel(prefix, inp, kHpc, kWpc, dHpc, dWpc, data_format, preconv_counter)#kHpc, kWpc, stride, data_format, preconv_counter)
  # conv1 + relu1
  conv1 = conv_kernel(prefix + '_conv1', preConv_pool, nIn, nOut, kH, kW, dH, dW, padType, data_format)
  # conv1-1 + relu1-1
  conv2 = conv_kernel(prefix + '_conv2', conv1, nOut, nOut, kH, kW, dH, dW, padType, data_format)
  # conv1-2 + relu1-2
  conv3 = conv_kernel(prefix + '_conv3', conv2, nOut, nOut, kH, kW, dH, dW, padType, data_format)
  # Pooling
  pool = max_pool_kernel(prefix + "_maxpool", conv3, kHp, kWp, dHp, dWp, data_format)

  return pool


def conv_pool(inp, nIn, nOut, kH, kW, dH, dW, kHp, kWp, dHp, dWp, padType, padding, pV, data_format):
  name = 'highres'
  with tf.name_scope(name) as scope:
    # conv1 + relu1
    conv1 = conv_kernel(name + '_conv1', inp, nIn, nOut, kH, kW, dH, dW, padType, data_format)
    # conv1-1 + relu1-1
    conv2 = conv_kernel(name + '_conv2', conv1, nOut, nOut, kH, kW, dH, dW, padType, data_format)
    # conv1-2 + relu1-2
    conv3 = conv_kernel(name + '_conv3', conv2, nOut, nOut, kH, kW, dH, dW, padType, data_format)
    # Pooling
    pool = max_pool_kernel(name + "_maxpool", conv3, kHp, kWp, dHp, dWp, data_format)

    return pool


def model_generator(num_classes,data_format='NCHW'): #'channels_first'):
  """Generator for Multi-scale CNN model

  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    num_classes: `int` number of possible classes for image classification.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
  """
  def model(inputs, is_training):
    """Creation of the model graph."""
    print("Yayyyyyyyyyyyyyyyyyyyyyayayayayaay")
    if data_format == 'NCHW': #'channels_first':
      images = tf.reshape(inputs, shape=[-1, 3, 1024, 1280])
    else:
      images = tf.reshape(inputs, shape=[-1, 1024, 1280, 3])

    nIn = 3  # Number of input channels
    preconv_counter = 1
    col1 = conv_pool(images, nIn, 16, 5, 5, 1, 1, 64, 64, 64, 64, 'SAME', True, 2, data_format)
    col2 = pc_par_conv_pool('res_2', images, 2, 2, 2, 2, nIn, 16, 5, 5, 1, 1, 32, 32, 32, 32, 'SAME', True, 2, data_format, preconv_counter)
    preconv_counter += 1
    col3 = pc_par_conv_pool('res_4', images, 4, 4, 4, 4, nIn, 16, 5, 5, 1, 1, 16, 16, 16, 16, 'SAME', True, 2, data_format, preconv_counter)
    preconv_counter += 1
    col4 = pc_par_conv_pool('res_8', images, 8, 8, 8, 8, nIn, 32, 5, 5, 1, 1, 8, 8, 8, 8, 'SAME', True, 2, data_format, preconv_counter)
    preconv_counter += 1
    col5 = pc_par_conv_pool('res_16', images, 16, 16, 16, 16, nIn, 32, 5, 5, 1, 1, 4, 4, 4, 4, 'SAME', True, 2, data_format, preconv_counter)
    preconv_counter += 1
    col6 = pc_par_conv_pool('res_32', images, 32, 32, 32, 32, nIn, 32, 5, 5, 1, 1, 2, 2, 2, 2, 'SAME', True, 2, data_format, preconv_counter)
    preconv_counter += 1
    col7 = pc_par_conv_pool('res_64', images, 64, 64, 64, 64, nIn, 64, 5, 5, 1, 1, 1, 1, 1, 1, 'SAME', True, 2, data_format, preconv_counter)

    # mergedPool
    if data_format == 'NCHW': #'channels_first':
      col_merge = tf.concat([col1, col2, col3, col4, col5, col6, col7], 1)
    else:
      col_merge = tf.concat([col1, col2, col3, col4, col5, col6, col7], 3)

    # mergedSummaryConv + relu-mergedSummaryConv
    mergedSummaryConv = conv_kernel('mergedSummaryConv', col_merge, 208, 1024, 1, 1, 1, 1, 'SAME', data_format=data_format)

    # poolMergedSummaryConv
    poolMergedSummaryConv = max_pool_kernel('mergedSummaryConv', mergedSummaryConv, 2, 2, 2, 2, data_format=data_format)

    # ip0 + relulp0
    resh1 = tf.reshape(poolMergedSummaryConv, [-1, 1024 * 10 * 8])
    ip0 = inner_product('ip0', resh1, 1024 * 10 * 8, 512)
    #kernel = tf.get_variable('ip3_weight', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer()) ####
    #biases = tf.Variable(tf.zeros(shape=[13], dtype=tf.float32), name='biases', trainable=True) ####
    #ip3 = tf.matmul(ip0, kernel, transpose_a=False, transpose_b=True) + biases ####
    ip3 = inner_product('ip3', ip0, 512, 13)

    return ip3

  model.default_image_size = 224
  return model


def mcnn(num_classes, data_format='NCHW'): #'channels_first'):
  """Returns the ResNet model for a given size and number of output classes."""
  
  return model_generator(num_classes, data_format)
