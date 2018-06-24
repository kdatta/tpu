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
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import multiscalecnn_preprocessing


def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    image = multiscalecnn_preprocessing.preprocess_image(
        image_bytes=image_bytes, is_training=False)
    return image

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


class HCIInput(object):
  """Generates HCI input fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the following format:

  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
    data_dir: `str` for the directory of the training and validation data;
        if 'null' (the literal string 'null', not None), then construct a null
        pipeline, consisting of empty images.
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    transpose_input: 'bool' for whether to use the double transpose trick
  """

  def __init__(self, is_training, data_dir, use_bfloat16, transpose_input=True):
    self.image_preprocessing_fn = multiscalecnn_preprocessing.preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.data_dir = data_dir
    self.transpose_input = transpose_input

  def dataset_parser(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'label':
            tf.FixedLenFeature([1], dtype=tf.int64),
        'image_raw':
            tf.FixedLenFeature([], dtype=tf.string)

    }
    parsed = tf.parse_single_example(value, keys_to_features)
    image = tf.decode_raw(parsed['image_raw'], tf.uint8) #decode_raw(bytes, out_type, little_endian=True, name=None)
    image = tf.reshape(image, [multiscalecnn_preprocessing.IMAGE_HEIGHT, multiscalecnn_preprocessing.IMAGE_WIDTH, 3])
    image = tf.cast(image, tf.float32) * (1.0 / 255.0)

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(parsed['label'], tf.int32)
    label = tf.reshape(label, []) #[1]
    print(label)
    if self.use_bfloat16:
      image = tf.cast(image, tf.bfloat16)

    return image, label

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    if self.data_dir == 'null':
      return self.input_fn_null(params)

    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    batch_size = params['batch_size']

    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(
        self.data_dir, 'slice-*' if self.is_training else 'test-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      # Number of bytes in the read buffer
      print ("Fetching data...")
      buffer_size = 32 * 1024 * 1280 * 3     # 16 images per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=8, sloppy=True))
    dataset = dataset.shuffle(buffer_size=32)#10000)#16)

    # Parse, preprocess, and batch the data in parallel
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            self.dataset_parser, batch_size=batch_size,
            num_parallel_batches=4))    # 8 == num_cores per host
            #drop_remainder=True)) not in tensorflow1.7

    # Transpose for performance on TPU
    if self.transpose_input:
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=8)

    def set_shapes(images, labels):
      """Statically set the batch_size dimension."""
      if self.transpose_input:
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([None, None, None, batch_size])))
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size])))
      else:
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size])))

      return images, labels

    # Assign static batch size dimension
    dataset = dataset.map(set_shapes)

    # Prefetch overlaps in-feed with training
    #dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=32)#16*1024*1280*3)
    return dataset

  def input_fn_null(self, params):
    """Input function which provides null (black) images."""
    batch_size = params['batch_size']
    dataset = tf.data.Dataset.range(1).repeat().map(self._get_null_input)
    dataset = dataset.prefetch(batch_size)

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.map(
        lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
        num_parallel_calls=8)

    dataset = dataset.prefetch(32)     # Prefetch overlaps in-feed with training
    tf.logging.info('Input dataset: %s', str(dataset))
    return dataset

  def _get_null_input(self, _):
    null_image = tf.zeros([224, 224, 3], tf.bfloat16
                          if self.use_bfloat16 else tf.float32)
    return (null_image, tf.constant(0, tf.int32))
