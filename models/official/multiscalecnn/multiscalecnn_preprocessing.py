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
"""ImageNet preprocessing for ResNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

IMAGE_HEIGHT = 724#1024 #724
IMAGE_WIDTH = 724#1280 #724
CROP_PADDING = 32
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_bytes, bbox]):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window)
    image = tf.image.convert_image_dtype(
        image, dtype=tf.float32)

    return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _normalize(image):
  """Normalize the image to zero mean and unit variance."""
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image = tf.reshape(image, [height, width, depth])
  image = tf.cast(image, tf.float32) * (1.0 / 255.0)
  offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
  image -= offset

  scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
  image /= scale
  return image


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


def preprocess_for_train(image_bytes):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """

  #image = _normalize(image_bytes)
  #image = _flip(image)
  #image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  pass
  #return image


def preprocess_for_eval(image_bytes):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  #image = _normalize(image_bytes)
  #image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  return image


def preprocess_image(image_bytes, is_training=False):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed image `Tensor`.
  """
  if is_training:
    return preprocess_for_train(image_bytes)
  else:
    return preprocess_for_eval(image_bytes)
