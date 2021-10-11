# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom neural network layers.

Low-level primitives such as custom convolution with custom initialization.
"""

from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf

from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


NCHW, NHWC = ('NCHW', 'NHWC')


# -----------------------------------------------------------------------------
# OPs / functions (those don't use any trainable variables).
# -----------------------------------------------------------------------------


def _check_order(order):
  if order not in (NCHW, NHWC):
    raise ValueError('Unsupported tensor order: %s.' % order)


def blur_pool(x, blur_filter=(1., 2., 1.), normalize=True, order=NHWC):
  """TBD."""
  blur_filter = np.array(blur_filter)
  if blur_filter.ndim == 1:
    blur_filter = blur_filter[:, np.newaxis] * blur_filter[np.newaxis, :]
  if normalize:
    blur_filter /= np.sum(blur_filter)
  blur_filter = blur_filter[:, :, np.newaxis, np.newaxis]
  blur_filter = tf.constant(blur_filter, dtype=tf.float32)
  channels = x.shape[1] if order == NCHW else x.shape[-1]
  blur_filter = tf.tile(blur_filter, [1, 1, channels, 1])
  strides = (1, 1, 2, 2) if order == NCHW else (1, 2, 2, 1)
  return tf.nn.depthwise_conv2d(
      input=x,
      filter=blur_filter,
      strides=strides,
      padding='SAME')


def downscale2d(x, n=2, pool_type='average', order=NHWC):
  """Box downscaling.

  Args:
    x: 4D tensor in order format.
    n: integer scale.
    pool_type: String, pooling method; one of {average, blur}.
    order: enum(NCHW, NHWC), the order of channels vs dimensions.

  Returns:
    4D tensor down scaled by a factor n.

  Raises:
    ValueError: if order not NCHW or NHWC.
  """
  _check_order(order)
  assert pool_type in ['average', 'blur']
  if n <= 1:
    return x
  if order == NCHW:
    pool2, pooln = [1, 1, 2, 2], [1, 1, n, n]
  else:
    pool2, pooln = [1, 2, 2, 1], [1, n, n, 1]
  if n % 2 == 0:
    if pool_type == 'average':
      x = tf.nn.avg_pool2d(x, pool2, pool2, 'VALID', order)
    elif pool_type == 'blur':
      x = blur_pool(x, order=order)
    return downscale2d(x, n // 2, order=order)
  # This shouldn't usually happen, unleas the downscale factor is odd and >= 3!
  return tf.nn.avg_pool2d(x, pooln, pooln, 'VALID', order)


def _upscale2d(x, n=2, order=NHWC):
  """Box upscaling (also called nearest neighbors).

  Args:
    x: 4D tensor in order format.
    n: integer scale (must be a power of 2).
    order: enum(NCHW, NHWC), the order of channels vs dimensions.

  Returns:
    4D tensor up scaled by a factor n.

  Raises:
    ValueError: if order not NCHW or NHWC.
  """
  _check_order(order)
  if n == 1:
    return x
  s = x.shape
  if order == NCHW:
    x = tf.reshape(x, [-1, s[1], s[2], s[3], 1])
    x = tf.tile(x, [1, 1, 1, n, n])
    x = tf.reshape(x, [-1, s[1], s[2] * n, s[3] * n])
  else:
    x = tf.tile(x, [1, 1, n, n])
    x = tf.reshape(x, [-1, s[1] * n, s[2] * n, s[3]])
  return x


def global_avg_pooling(x, keepdims=True, order=NHWC):
  """TBD"""
  axis = [2, 3] if order == NCHW else [1, 2]
  return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def flatten_spatial_dimensions(tensor, order=NHWC):
  """TBD"""
  assert order == NHWC, 'NCHW not yet supported (TBD).'
  batch_size = tf.dimension_value(tensor.shape[0])
  nc = tf.dimension_value(tensor.shape[-1])
  return tf.reshape(tensor, shape=(batch_size, -1, nc))


def remove_details2d(x, n=2, order=NHWC):
  """Removes box details by upscaling a downscaled image.

  Args:
    x: 4D tensor in order format.
    n: integer scale (must be a power of 2).
    order: enum(NCHW, NHWC), the order of channels vs dimensions.

  Returns:
    4D tensor image with removed details of size nxn.
  """
  if n == 1:
    return x
  return _upscale2d(downscale2d(x, n, order=order), n, order=order)


def blend_resolution(lores, hires, alpha):
  """Blends two images.

  Args:
      lores: 4D tensor, low resolution image.
      hires: 4D tensor, high resolution image.
      alpha: scalar tensor, 0 produces the low resolution, 1 the high one.

  Returns:
      4D tensor of blended images.
  """
  return lores + alpha * (hires - lores)


def _kaiming_scale(shape):
  # Shape is [kernel, kernel, fmaps_in, fmaps_out] or [in, out] for Conv/Dense.
  fan_in = np.prod(shape[:-1])
  return 1. / np.sqrt(fan_in)


# -----------------------------------------------------------------------------
# Custom keras initializers. (experimental)
# -----------------------------------------------------------------------------

class GlorotNormalInitializerWithGain(tf.keras.initializers.GlorotNormal):

  def __init__(self, gain=0.02, **kwargs):
    super(GlorotNormalInitializerWithGain, self).__init__(**kwargs)
    self.gain = gain

  def __call__(self, *args, **kwargs):
    weights = super(GlorotNormalInitializerWithGain, self).__call__(
        *args, **kwargs)
    return self.gain * weights


# -----------------------------------------------------------------------------
# Custom keras layers.
# -----------------------------------------------------------------------------


class ResBlock(tf.keras.layers.Layer):
  """Convolution block with resisudal connections (e.g. described in BigGAN)."""

  def __init__(self, fin, fout, conv_layer, act_layer, norm_layer, name=None,
               order=NHWC, simulate_shortcut_norm_bug=False,
               add_spatial_noise=False, **kwargs):
    super(ResBlock, self).__init__(name=name, **kwargs)
    f_middle = min(fin, fout)
    def _create_conv_block(num_filters, kernel_size, use_bias=True,
                          apply_norm=True, apply_activation=True,
                          block_name=None):
      """TBD."""
      conv_block = tf.keras.Sequential(name=block_name)
      if add_spatial_noise:
        conv_block.add(StyleGANSpatialNoise(name='spatial_noise', order=order))
      if (apply_norm  or simulate_shortcut_norm_bug) and norm_layer is not None:
        conv_block.add(norm_layer())
      if apply_activation:
        conv_block.add(act_layer())
      conv_block.add(
          conv_layer(filters=num_filters, kernel_size=kernel_size,
                     use_bias=use_bias, activation=None, name='conv2d'))
      return conv_block

    if fin != fout:
      self.shortcut = _create_conv_block(
          fout, 1, use_bias=False, apply_norm=False, apply_activation=False,
          block_name='Shortuct')
    else:
      self.shortcut = None
    self.conv_block0 = _create_conv_block(f_middle, 3, block_name='Conv0')
    self.conv_block1 = _create_conv_block(fout, 3, block_name='Conv1')

  def call(self, inputs, spatial_noise=None, training=None):
    if self.shortcut is not None:
      x_shortcut = self.shortcut(inputs, training=training)
    else:
      x_shortcut = inputs
    x = self.conv_block0(inputs, training=training)
    x = self.conv_block1(x, training=training)
    return x + x_shortcut


class ResBlockDown(tf.keras.layers.Layer):
  """Convolution downsample resblock (e.g. described in BigGAN)."""

  def __init__(self, fout, conv_layer, act_layer, norm_layer,
               pool_type='average', mul_factor=1., name=None, order=NHWC,
               **kwargs):
    super(ResBlockDown, self).__init__(name=name, **kwargs)
    self.fout = fout
    self.conv_layer = conv_layer
    self.act_layer = act_layer
    self.norm_layer = norm_layer
    self.pool_type = pool_type
    self.mul_factor = mul_factor
    self.order = order

  def build(self, input_shape):
    fin = input_shape[-1] if self.order == NHWC else input_shape[1]
    # Shortcut conv (no bias).
    self.conv_shortcut = self.conv_layer(
        filters=self.fout, kernel_size=1, use_bias=False, activation=None,
        name='conv_shortcut')
    # Conv block #0.
    self.norm0 = self.norm_layer() if self.norm_layer is not None else None
    self.act0 = self.act_layer()
    self.conv0 = self.conv_layer(filters=fin, kernel_size=3,
                                 use_bias=True, activation=None, name='Conv0')
    # Conv block #1.
    self.norm1 = self.norm_layer() if self.norm_layer is not None else None
    self.act1 = self.act_layer()
    self.conv1 = self.conv_layer(filters=self.fout, kernel_size=3,
                                 use_bias=True, activation=None, name='Conv1')
    # Downsampling layer/function.
    self.downsample2d = functools.partial(
        downscale2d, n=2, pool_type=self.pool_type, order=self.order)
    super(ResBlockDown, self).build(input_shape)

  def _maybe_normalize(self, apply_norm_fn, x, z_style, training):
    """Applies standard normalization or adaptive instance normalization (AdaIN), if any."""
    if apply_norm_fn is None:
      return x
    else:
      inputs = [x, z_style] if z_style is not None else x
      return apply_norm_fn(inputs, training=training)

  def call(self, inputs, z_style=None, training=None):
    x_shortcut = self.conv_shortcut(inputs, training=training)
    x_shortcut = self.downsample2d(x_shortcut)
    x = inputs
    x = self._maybe_normalize(self.norm0, x, z_style, training=training)
    x = self.act0(x)
    x = self.conv0(x, training=training)
    x = self._maybe_normalize(self.norm1, x, z_style, training=training)
    x = self.act1(x)
    x = self.conv1(x, training=training)
    x = self.downsample2d(x)
    return (x + x_shortcut) * self.mul_factor


class ResBlockUp(tf.keras.layers.Layer):
  """Convolution upsample resblock (e.g. described in BigGAN)."""

  def __init__(self, fout, conv_layer, act_layer, norm_layer,
               interpolation='bilinear', name=None, order=NHWC, **kwargs):
    super(ResBlockUp, self).__init__(name=name, **kwargs)
    self.fout = fout
    self.conv_layer = conv_layer
    self.act_layer = act_layer
    self.norm_layer = norm_layer
    self.interpolation = interpolation
    self.order = order

  def build(self, input_shape):
    # Shortcut conv (no bias).
    self.conv_shortcut = self.conv_layer(
        filters=self.fout, kernel_size=1, use_bias=False, activation=None,
        name='conv_shortcut')
    # Conv block #0.
    self.norm0 = self.norm_layer() if self.norm_layer is not None else None
    self.act0 = self.act_layer()
    self.conv0 = self.conv_layer(filters=self.fout, kernel_size=3,
                                 use_bias=True, activation=None, name='Conv0')
    # Conv block #1.
    self.norm1 = self.norm_layer() if self.norm_layer is not None else None
    self.act1 = self.act_layer()
    self.conv1 = self.conv_layer(filters=self.fout, kernel_size=3,
                                 use_bias=True, activation=None, name='Conv1')
    # Upsampling layer/function.
    data_format = 'channels_last' if self.order == NHWC else 'channels_last'
    self.upsample2d = tf.keras.layers.UpSampling2D(
        size=2, interpolation=self.interpolation, data_format=data_format)
    super(ResBlockUp, self).build(input_shape)

  def _maybe_normalize(self, apply_norm_fn, x, z_style, training):
    if apply_norm_fn is None:
      return x
    else:
      inputs = [x, z_style] if z_style is not None else x
      return apply_norm_fn(inputs, training=training)

  def call(self, inputs, z_style=None, training=None):
    x_shortcut = self.upsample2d(inputs)
    x_shortcut = self.conv_shortcut(x_shortcut, training=training)
    x = inputs
    x = self._maybe_normalize(self.norm0, x, z_style, training=training)
    x = self.act0(x)
    x = self.upsample2d(x)
    x = self.conv0(x, training=training)
    x = self._maybe_normalize(self.norm1, x, z_style, training=training)
    x = self.act1(x)
    x = self.conv1(x, training=training)
    return x + x_shortcut


class SPADEConvBlock(tf.keras.layers.Layer):
  """TBD."""

  def __init__(self, pre_norm_layer, act_layer, conv_layer, fout,
               spade_norm_conv_layer, projection_filters=128, kernel_size=3,
               use_bias=True, name=None, order=NHWC, **kwargs):
    super(SPADEConvBlock, self).__init__(**kwargs)
    self.spade_norm = SPADENormalization(
        pre_norm_layer,
        spade_norm_conv_layer,
        projection_filters=projection_filters,
        order=order,
        **kwargs)
    self.activation = act_layer() if act_layer is not None else None
    self.conv = conv_layer(
        filters=fout, kernel_size=kernel_size, strides=1, padding='SAME',
        use_bias=use_bias, activation=None,
        data_format='channels_last' if order == NHWC else 'channels_first')

  def call(self, inputs, training=None):
    x, x_cond = inputs
    x = self.spade_norm([x, x_cond], training=training)
    if self.activation is not None:
      x = self.activation(x)
    x = self.conv(x, training=training)
    return x


class SPADEResBlock(tf.keras.layers.Layer):
  """TBD."""

  def __init__(self, fin, fout, conv_layer, act_layer, pre_norm_layer,
               projection_filters=128, spade_norm_conv_layer=None,
               order=NHWC, **kwargs):
    super(SPADEResBlock, self).__init__(**kwargs)
    if spade_norm_conv_layer is None:
      spade_norm_conv_layer = conv_layer
    f_middle = min(fin, fout)
    # x_shortcut = x_init
    if fin != fout:
      # Shortcut doesn't have activation after norm, its conv has no bias and
      #  its kernel_size is 1.
      self.shortcut = SPADEConvBlock(
          pre_norm_layer, None, conv_layer, fout, spade_norm_conv_layer,
          projection_filters=projection_filters, kernel_size=1, use_bias=False,
          name='spade_block_shortcut')
    else:
      self.shortcut = None
    self.conv_block0 = SPADEConvBlock(
        pre_norm_layer, act_layer, conv_layer, f_middle, spade_norm_conv_layer,
        projection_filters=projection_filters, name='spade_block_conv0')
    self.conv_block1 = SPADEConvBlock(
        pre_norm_layer, act_layer, conv_layer, fout, spade_norm_conv_layer,
        projection_filters=projection_filters, name='spade_block_conv1')

  def call(self, inputs, training=None):
    x, x_cond = inputs
    if self.shortcut is not None:
      x_shortcut = self.shortcut([x, x_cond], training=training)
    else:
      x_shortcut = x
    x = self.conv_block0([x, x_cond], training=training)
    x = self.conv_block1([x, x_cond], training=training)
    return x + x_shortcut


class SPADENormalization(tf.keras.layers.Layer):
  """TBD."""

  # NOTE: SPADE_norm uses 3x3 conv when using sync_batch_norm and 5x5 conv --
  # when using instance_norm (but 5x5 seems a lot slower)!.
  def __init__(self, pre_norm_layer, conv_layer, projection_filters=128,
               kernel_size=3, order=NHWC, **kwargs):
    super(SPADENormalization, self).__init__(**kwargs)
    self.projection_filters = projection_filters
    self.order = order
    data_format = 'channels_last' if order == NHWC else 'channels_first'
    self.conv_layer = functools.partial(
        conv_layer, kernel_size=kernel_size, strides=1, padding='SAME',
        use_bias=True, data_format=data_format)
    self.pre_norm_layer = pre_norm_layer

  def build(self, input_shape):
    super(SPADENormalization, self).build(input_shape)
    # Extract shapes and compute downscale factor for the spatial input.
    x_shape, x_cond_shape = input_shape
    _, x_h, x_w, x_nc = x_shape
    _, x_cond_h, x_cond_w, _ = x_cond_shape
    assert x_h == x_w and x_cond_h == x_cond_w  # supports square resolutions
    self.downscale_factor = x_cond_h // x_h
    # Build conv layers.
    # The first conv uses ReLU activation.
    self.conv0 = self.conv_layer(
        filters=self.projection_filters, activation=tf.nn.relu,
        name='spade_projection_conv')
    self.gamma_conv = self.conv_layer(filters=x_nc, activation=None,
                                      name='gamma_conv')
    self.beta_conv = self.conv_layer(filters=x_nc, activation=None,
                                     name='beta_conv')
    self.feature_mod = FeatureModulation(self.pre_norm_layer, self.order,
                                         name='feature_mod')

  def call(self, inputs, training=None):
    x, x_cond = inputs
    # QUES: does downscaled2d() create a new tensor with each call() execution?
    x_cond_down = downscale2d(x_cond, n=self.downscale_factor, order=self.order)
    x_cond_down = self.conv0(x_cond_down, training=training)
    gamma_map = self.gamma_conv(x_cond_down, training=training)
    beta_map = self.beta_conv(x_cond_down, training=training)
    return self.feature_mod([x, gamma_map, beta_map], training=training)


class SelfAttention(tf.keras.layers.Layer):
  """Self-attention layer."""

  def __init__(self, fout, conv_layer, channel_multiplier=1./8., pool_size=2,
               order=NHWC, **kwargs):
    assert order == NHWC, 'NCHW not yet supported (TBD).'
    self.fout = fout
    self.conv_layer = conv_layer
    self.channel_multiplier = channel_multiplier
    self.pool_size = pool_size
    self.order = order
    super(SelfAttention, self).__init__(**kwargs)

  def build(self, input_shape):
    batch_size, height, width, fin = input_shape
    self.max_pool_layer = tf.keras.layers.MaxPool2D(
        pool_size=self.pool_size, strides=2, padding='SAME')
    num_inner_channels = max(int(self.channel_multiplier * self.fout), 1)

    self.f_conv = self.conv_layer(
        filters=num_inner_channels,
        kernel_size=1,
        strides=1,
        use_bias=False,
        activation=None,
        name='f_conv')  # [bs, h, w, c']
    self.g_conv = self.conv_layer(
        filters=num_inner_channels,
        kernel_size=1,
        strides=1,
        use_bias=False,
        activation=None,
        name='g_conv')  # [bs, h, w, c']
    self.h_conv = self.conv_layer(
        filters=num_inner_channels,
        kernel_size=1,
        strides=1,
        use_bias=False,
        activation=None,
        name='h_conv')  # [bs, h, w, c']
    self.reshape_attention_output = functools.partial(
        tf.reshape, shape=(batch_size, height, width, num_inner_channels))
    self.attention_conv = self.conv_layer(
        filters=fin,
        kernel_size=1,
        strides=1,
        use_bias=False,
        activation=None,
        name='attention_conv')  # [bs, h, w, c]
    self.attention_weight = self.add_weight(
        name='attention_weight',
        shape=(),
        dtype=self.dtype,
        initializer=tf.constant_initializer(0.),
        constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    super(SelfAttention, self).build(input_shape)

  def call(self, inputs, training=None):
    x = inputs
    # f branch.
    f_x = self.f_conv(x, training=training)  # [bs, h, w, c']
    f_x = self.max_pool_layer(f_x)
    flattened_f_x = flatten_spatial_dimensions(f_x)
    # g branch (no max pooling).
    g_x = self.g_conv(x, training=training)  # [bs, h, w, c']
    flattened_g_x = flatten_spatial_dimensions(g_x)
    # h branch.
    h_x = self.h_conv(x, training=training)  # [bs, h, w, c']
    h_x = self.max_pool_layer(h_x)
    flattened_h_x = flatten_spatial_dimensions(h_x)
    # Compute flattened attention map.
    attention_map = tf.matmul(flattened_g_x, flattened_f_x,
                              transpose_b=True)  # [bs, flat(h*w), flat(h*w)]
    attention_map = tf.nn.softmax(attention_map, axis=-1)
    # Weight the flattend h branch with the flattend attention map.
    flattened_out = tf.matmul(
        attention_map, flattened_h_x)  # [bs, flat(h*w), c']
    # Reshape the flattened weighted output back to 4D.
    out = self.reshape_attention_output(flattened_out)  # [bs, h, w, c']
    # Apply final convolution and add a shortcut.
    attention_conv_output = self.attention_conv(
        out, training=training)  # [bs, h, w, c]
    x = self.attention_weight * attention_conv_output + x
    return x


class DenseScaled(tf.keras.layers.Dense):
  """Learning rate scaled version of the tf.layers.Dense.
  """

  def __init__(self, units, gain=1, lr_mul=1, **kwargs):
    init_std = 1. / lr_mul
    super(DenseScaled, self).__init__(
        units=units,
        kernel_initializer=tf.random_normal_initializer(stddev=init_std),
        **kwargs)
    self.gain = gain
    self.lr_mul = lr_mul

  def build(self, input_shape):
    super(DenseScaled, self).build(input_shape)
    he_scale = _kaiming_scale(self.kernel.shape.as_list())
    self.runtime_coeff = tf.constant(self.gain * self.lr_mul * he_scale,
                                     dtype=self.kernel.dtype)

  def call(self, inputs):
    # QUES: does tf.matmul() create a new tensor with each call() execution?
    # outputs = tf.matmul(inputs, self.kernel * cur_scale)
    outputs = gen_math_ops.mat_mul(inputs, self.kernel * self.runtime_coeff)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is None:
      return outputs
    return self.activation(outputs)

  def set_gain(self, gain):
    self.gain = gain


class Conv2DScaled(tf.keras.layers.Conv2D):
  """Learning rate scaled version of the tf.layers.Conv2D.
  """

  def __init__(self, filters, padding='same', gain=1, lr_mul=1, **kwargs):
    self.gain = gain
    init_std = 1. / lr_mul
    super(Conv2DScaled, self).__init__(
        filters=filters,
        padding=padding,
        kernel_initializer=tf.random_normal_initializer(stddev=init_std),
        **kwargs)

  def build(self, input_shape):
    super(Conv2DScaled, self).build(input_shape)
    he_scale = _kaiming_scale(self.kernel.shape.as_list())
    self.runtime_coeff = tf.constant(self.gain * he_scale,
                                     dtype=self.kernel.dtype)

  def call(self, inputs):
    if self._recreate_conv_op(inputs):
      self._convolution_op = nn_ops.Convolution(
          inputs.get_shape(),
          filter_shape=self.kernel.shape,
          dilation_rate=self.dilation_rate,
          strides=self.strides,
          padding=self._padding_op,
          data_format=self._conv_op_data_format)
    outputs = self._convolution_op(inputs, self.kernel * self.runtime_coeff)
    if self.use_bias:
      fmt = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'
      outputs = nn.bias_add(outputs, self.bias, data_format=fmt)
    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def set_gain(self, gain):
    self.gain = gain


class DenseScaled2(tf.keras.layers.Dense):
  """Learning rate scaled version of the tf.layers.Dense.
  """

  def __init__(self, units, gain=1, lr_mul=1, **kwargs):
    init_std = 1. / lr_mul
    super(DenseScaled2, self).__init__(
        units=units,
        kernel_initializer=tf.random_normal_initializer(stddev=init_std),
        **kwargs)
    self.gain = gain
    self.lr_mul = lr_mul

  def build(self, input_shape):
    super(DenseScaled2, self).build(input_shape)
    he_scale = _kaiming_scale(self.kernel.shape.as_list())
    self.runtime_coeff = self.gain * self.lr_mul * he_scale

  def call(self, inputs):
    output = super(DenseScaled2, self).call(inputs)
    return self.runtime_coeff * output


class Conv2DScaled2(tf.keras.layers.Conv2D):
  """Learning rate scaled version of the tf.layers.Conv2D.
  """

  def __init__(self, filters, gain=1, lr_mul=1, **kwargs):
    self.gain = gain
    self.lr_mul = lr_mul
    init_std = 1. / lr_mul
    super(Conv2DScaled2, self).__init__(
        filters=filters,
        kernel_initializer=tf.random_normal_initializer(stddev=init_std),
        **kwargs)

  def build(self, input_shape):
    super(Conv2DScaled2, self).build(input_shape)
    he_scale = _kaiming_scale(self.kernel.shape.as_list())
    self.runtime_coeff = self.gain * self.lr_mul * he_scale

  def call(self, inputs):
    output = super(Conv2DScaled2, self).call(inputs)
    return self.runtime_coeff * output


class ParamFreeInstanceNorm(tf.keras.layers.Layer):
  """TBD."""

  def __init__(self, epsilon=1e-8, order=NHWC, name=None, **kwargs):
    self.reduce_axes = [1, 2] if order == NHWC else [2, 3]
    self.eps = tf.constant(epsilon, name='epsilon')
    super(ParamFreeInstanceNorm, self).__init__(name=name, **kwargs)

  def call(self, x):
    x -= tf.reduce_mean(x, axis=self.reduce_axes, keepdims=True)
    x *= tf.math.rsqrt(
        self.eps + tf.reduce_mean(
            tf.math.square(x), axis=self.reduce_axes, keepdims=True))
    return x


class FeatureModulation(tf.keras.layers.Layer):
  """TBD."""

  def __init__(self, pre_norm_layer, order=NHWC, name=None, **kwargs):
    super(FeatureModulation, self).__init__(name=name, **kwargs)
    self.norm_layer = pre_norm_layer()
    self.order = order

  def call(self, inputs, training=None):
    x, gamma_map, beta_map = inputs
    if self.norm_layer is not None:
      x = self.norm_layer(x, training=training)
    return x * (1 + gamma_map) + beta_map


class StyleModulation(tf.keras.layers.Layer):
  """TBD."""

  def __init__(self, dense_layer, param_free_norm=None, order=NHWC, **kwargs):
    super(StyleModulation, self).__init__(**kwargs)
    self.dense_layer = dense_layer
    self.order = order
    self.feature_mod = FeatureModulation(param_free_norm, order=order)

  def build(self, input_shape):  # Expects a pair of inputs (x, dlatents).
    super(StyleModulation, self).build(input_shape)
    x_shape, _ = input_shape
    channel_dim = -1 if self.order == NHWC else 1
    num_out_filters = 2 * x_shape[channel_dim]
    self.dense = self.dense_layer(units=num_out_filters, use_bias=True,
                                  name='dense')
    if self.order == NHWC:
      self.reshape = [-1, 2, 1, 1, x_shape[channel_dim]]
    else:
      self.reshape = [-1, 2, x_shape[channel_dim], 1, 1]

  def call(self, inputs, training=None):
    x, dlatents = inputs
    gammas_and_betas = self.dense(dlatents, training=training)
    gammas_and_betas = tf.reshape(gammas_and_betas, self.reshape)
    gamma_map = gammas_and_betas[:, 0]
    beta_map = gammas_and_betas[:, 1]
    return self.feature_mod([x, gamma_map, beta_map], training=training)


class StyleGANSpatialNoise(tf.keras.layers.Layer):
  """TBD."""

  def __init__(self, per_channel_noise_weight=True, order=NHWC, **kwargs):
    super(StyleGANSpatialNoise, self).__init__(**kwargs)
    self.order = order
    self.channel_dim = 3 if self.order == NHWC else 1
    self.per_channel_noise_weight = per_channel_noise_weight

  def build(self, input_shape):
    super(StyleGANSpatialNoise, self).build(input_shape)
    if self.per_channel_noise_weight:
      weight_shape = [1, 1, 1, 1]
      weight_shape[self.channel_dim] = input_shape[self.channel_dim]
    else:  # scalar weight for all channels as in StyleGAN2
      weight_shape = []
    self.noise_weight = self.add_weight(
        name='noise_weight', shape=weight_shape,
        initializer=tf.initializers.zeros(), trainable=True)

  def call(self, x, noise_map=None):
    if noise_map is None:
      noise_shape = x.shape.as_list().copy()
      noise_shape[self.channel_dim] = 1
      noise_map = tf.random.normal(noise_shape, dtype=x.dtype)
    else:
      noise_map = tf.dtypes.cast(self.noise_map, x.dtype)
    noise_weight = tf.dtypes.cast(self.noise_weight, x.dtype)
    return x + noise_weight * noise_map


class BiasLayer(tf.keras.layers.Layer):
  """TBD."""

  def __init__(self, lr_mul=1, order=NHWC, **kwargs):
    super(BiasLayer, self).__init__(**kwargs)
    self.lr_mul = lr_mul
    self.order = order

  def build(self, input_shape):
    super(BiasLayer, self).build(input_shape)
    # The following works for both 2D and 4D tensors.
    channel_dim = -1 if self.order == NHWC else 1
    bias_shape = [1] * len(input_shape)
    bias_shape[channel_dim] = input_shape[channel_dim]
    self.bias = self.add_weight(
        name='bias', shape=bias_shape, initializer=tf.initializers.zeros(),
        trainable=True)

  def call(self, inputs):
    return inputs + tf.dtypes.cast(self.bias * self.lr_mul, inputs.dtype)


# NOTE: this isn't completely faithful to styleGAN's released code.
class StyleGANLayerEpilogue(tf.keras.layers.Layer):
  """TBD."""

  def __init__(self, nf, conv_layer, dense_layer, act_layer, norm_layer,
               add_spatial_noise=True, use_style_mod=True, order=NHWC,
               **kwargs):
    super(StyleGANLayerEpilogue, self).__init__(**kwargs)
    self.add_spatial_noise = add_spatial_noise
    self.use_style_mod = use_style_mod
    if conv_layer is not None:
      data_format = 'channels_last' if order == NHWC else 'channels_first'
      self.conv = conv_layer(nf, kernel_size=1, strides=1, padding='SAME',
                             use_bias=False, activation=None,
                             data_format=data_format, name='conv2d')
    else:
      self.conv = None
    if add_spatial_noise:
      self.add_noise = StyleGANSpatialNoise(name='spatial_noise', order=order)
    self.add_bias = BiasLayer(order=order, name='bias_add')
    self.activation = act_layer()
    if use_style_mod:
      self.style_mod = StyleModulation(
          dense_layer, param_free_norm=norm_layer, order=order,
          name='style_mod')
    else:
      self.normalize = norm_layer()

  def call(self, inputs, training=None):  # Expects 2 inputs: {x, dlatents=None}
    x, dlatents = inputs
    if self.conv is not None:
      x = self.conv(x, training=training)
    if self.add_spatial_noise:
      x = self.add_noise(x, training=training)
    x = self.add_bias(x, training=training)
    x = self.activation(x)
    if self.use_style_mod:
      x = self.style_mod([x, dlatents], training=training)
    else:
      x = self.normalize(x, training=training)
    return x


class IdentityLayer(tf.keras.layers.Layer):
  """TBD."""

  def call(self, inputs):
    return inputs


# Copy of tensorflow 2.x InstanceNormalization code to use with tf 1.15.
# TODO: Need to replace with tf_addons implementation.
class InstanceNormalization(tf.keras.layers.Layer):
  """Instance normalization layer.

  Normalize the activations of the previous layer at each step,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  # Arguments
    axis: Integer, the axis that should be normalized
      (typically the features axis).
      For instance, after a `Conv2D` layer with
      `data_format="channels_first"`,
      set `axis=1` in `InstanceNormalization`.
      Setting `axis=None` will normalize all values in each
      instance of the batch.
      Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
      If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
      When the next layer is linear (also e.g. `nn.relu`),
      this can be disabled since the scaling
      will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.

  # Input shape
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a Sequential model.

  # Output shape
    Same shape as input.

  # References
    - [Layer Normalization](https://arxiv.org/abs/1607.06450)
    - [Instance Normalization: The Missing Ingredient for Fast Stylization](
    https://arxiv.org/abs/1607.08022)
  """

  def __init__(self,
               axis=None,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               **kwargs):
    super(InstanceNormalization, self).__init__(**kwargs)
    self.supports_masking = True
    self.axis = axis
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = tf.keras.initializers.get(beta_initializer)
    self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
    self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
    self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
    self.beta_constraint = tf.keras.constraints.get(beta_constraint)
    self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

  def build(self, input_shape):
    ndim = len(input_shape)
    if self.axis == 0:
      raise ValueError('Axis cannot be zero')

    if (self.axis is not None) and (ndim == 2):
      raise ValueError('Cannot specify axis for rank 1 tensor')

    self.input_spec = tf.keras.layers.InputSpec(ndim=ndim)

    if self.axis is None:
      shape = (1,)
    else:
      shape = (input_shape[self.axis],)

    if self.scale:
      self.gamma = self.add_weight(shape=shape,
                                   name='gamma',
                                   initializer=self.gamma_initializer,
                                   regularizer=self.gamma_regularizer,
                                   constraint=self.gamma_constraint)
    else:
      self.gamma = None
    if self.center:
      self.beta = self.add_weight(shape=shape,
                                  name='beta',
                                  initializer=self.beta_initializer,
                                  regularizer=self.beta_regularizer,
                                  constraint=self.beta_constraint)
    else:
      self.beta = None
    self.built = True

  def call(self, inputs, training=None):
    input_shape = tf.keras.backend.int_shape(inputs)
    reduction_axes = list(range(0, len(input_shape)))

    if self.axis is not None:
      del reduction_axes[self.axis]

    del reduction_axes[0]

    mean = tf.keras.backend.mean(inputs, reduction_axes, keepdims=True)
    stddev = tf.keras.backend.std(
        inputs, reduction_axes, keepdims=True) + self.epsilon
    normed = (inputs - mean) / stddev

    broadcast_shape = [1] * len(input_shape)
    if self.axis is not None:
      broadcast_shape[self.axis] = input_shape[self.axis]

    if self.scale:
      broadcast_gamma = tf.keras.backend.reshape(self.gamma, broadcast_shape)
      normed = normed * broadcast_gamma
    if self.center:
      broadcast_beta = tf.keras.backend.reshape(self.beta, broadcast_shape)
      normed = normed + broadcast_beta
    return normed

  def get_config(self):
    config = {
        'axis': self.axis,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': tf.keras.initializers.serialize(
            self.beta_initializer),
        'gamma_initializer': tf.keras.initializers.serialize(
            self.gamma_initializer),
        'beta_regularizer': tf.keras.regularizers.serialize(
            self.beta_regularizer),
        'gamma_regularizer': tf.keras.regularizers.serialize(
            self.gamma_regularizer),
        'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
        'gamma_constraint': tf.keras.constraints.serialize(
            self.gamma_constraint)
    }
    base_config = super(InstanceNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# -----------------------------------------------------------------------------
# TODOs section: the following are not yet converted into the new API!
# -----------------------------------------------------------------------------

# TODO(meshry): wrap in a tf.keras.layers.Layer class.
def channel_norm(x, order=NHWC):
  """Channel normalization.

  Args:
    x: 4D image tensor (in either NCHW or NHWC format).
    order: enum(NCHW, NHWC), the order of channels vs dimensions in the image
      tensor.

  Returns:
    nD tensor with normalized channels.

  Raises:
    ValueError: if order not NCHW or NHWC.
  """
  _check_order(order)
  return channel_norm_via_dim(x, 1 if order == NCHW else 3)


# TODO(meshry): wrap in a tf.keras.layers.Layer class.
def channel_norm_via_dim(x, channel_dim):
  """Channel normalization.

  Args:
    x: nD tensor with channels in dimension 'channel_dim'.
    channel_dim: The dimension of 'x' containing the channels.

  Returns:
    nD tensor with normalized channels.
  """
  return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), [channel_dim],
                                          keepdims=True) + 1e-8)


# TODO(meshry): wrap in a tf.keras.layers.Layer class.
def minibatch_mean_variance(x):
  """Computes the variance average.

  This is used by the discriminator as a form of batch discrimination.

  Args:
    x: nD tensor for which to compute variance average.

  Returns:
    a scalar, the mean variance of variable x.
  """
  mean = tf.reduce_mean(x, 0, keepdims=True)
  vals = tf.sqrt(tf.reduce_mean(tf.squared_difference(x, mean), 0) + 1e-8)
  vals = tf.reduce_mean(vals)
  return vals


# To disable grouping, set group_size to -1 or None
def minibatch_mean_stddev(x, group_size=4, num_new_feats=1, order=NHWC):
  """TBD."""
  if group_size is None or group_size == -1:
    group_size = np.inf
  # batch_size must be divisbile by (or smaller than) group size.
  x_shape = x.shape.as_list()
  if order == NHWC:
    batch_size, H, W, nf = x_shape
    channels_dim = -1
  else:
    batch_size, nf, H, W = x_shape
    channels_dim = 2  # Because there is an extra leading dim for groups.
  group_size = min(group_size, batch_size)
  # Let
  #   c: num_new_feats,
  #   C': C//c
  #   N': group_size,
  #   G: num_groups (i.e. N//N')
  # Reshape to [N', G, H, W, c, C']; N' = N//G, C' = C//num_new_feats
  if order == NHWC:
    y = tf.reshape(x, [group_size, -1, H, W, nf//num_new_feats, num_new_feats])
  else:
    y = tf.reshape(x, [group_size, -1, num_new_feats, nf//num_new_feats, H, W])
  # Subtract mean over group.
  y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [N',G,H,W,C',c]
  # Compute variance over group.
  y = tf.reduce_mean(tf.square(y), axis=0, keepdims=False)  # [G,H,W,C',c]
  # Compute stddev over group.
  y = tf.sqrt(y + 1e-8)  # [G,H,W,C',c]
  # Compute num_new_feats scalars per image group representing the mean stddev.
  if order == NHWC:
    y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [G,1,1,1,c]
    # Remove the channels dim.
    y = tf.reduce_mean(y, axis=3)  # [G, 1, 1, c]
    # Replicate over group and pixels.
    y = tf.tile(y, [group_size, H, W, 1])  # [N,H,W,c]
    channels_dim = 3
  else:
    y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)  # [G,c,1,1,1]
    # Remove the channels dim.
    y = tf.reduce_mean(y, axis=2)  # [G, c, 1, 1]
    # Replicate over group and pixels.
    y = tf.tile(y, [group_size, 1, H, W])  # [N,c,H,W]
    channels_dim = 1
  # Append the new c channels to x channels.
  return tf.concat([x, y], axis=channels_dim)


# TODO(meshry): wrap in a tf.keras.layers.Layer class.
def scalar_concat(x, scalar, order=NHWC):
  """Concatenates a scalar to a 4D tensor as an extra channel.

  Args:
    x: 4D image tensor (in either NCHW or NHWC format).
    scalar: a scalar to concatenate to the tensor.
    order: enum(NCHW, NHWC), the order of channels vs dimensions in the image
      tensor.

  Returns:
    a 4D tensor with one extra channel containing the value scalar at
     every position.

  Raises:
    ValueError: if order not NCHW or NHWC.
  """
  _check_order(order)
  s = tf.shape(x)
  if order == NCHW:
    return tf.concat([x, tf.ones([s[0], 1, s[2], s[3]]) * scalar], axis=1)
  else:
    return tf.concat([x, tf.ones([s[0], s[1], s[2], 1]) * scalar], axis=3)


# TODO(meshry): wrap in a tf.keras.layers.Layer class.
def self_attention(x, fout, conv_layer, channel_multiplier=1./8.,
                   pool_size=2, order=NHWC):
  assert order == NHWC, 'NCHW not yet supported (TBD).'
  batch_size, H, W, fin = x.shape
  max_pool_layer = functools.partial(
      tf.keras.layers.max_pooling2d, pool_size=pool_size, strides=2,
      padding='SAME')
  num_inner_channels = max(int(channel_multiplier * fout), 1)

  with tf.variable_scope('f_conv'):
    f_x = conv_layer(x, num_inner_channels, kernel_size=1, strides=1,
                     use_bias=True, activation=None)  # [bs, h, w, c']
    f_x = max_pool_layer(f_x)
    flattened_f_x = flatten_spatial_dimensions(f_x)
  with tf.variable_scope('g_conv'):
    g_x = conv_layer(x, num_inner_channels, kernel_size=1, strides=1,
                     use_bias=True, activation=None)  # [bs, h, w, c']
    flattened_g_x = flatten_spatial_dimensions(g_x)
  with tf.variable_scope('h_conv'):
    h_x = conv_layer(x, num_inner_channels, kernel_size=1, strides=1,
                     use_bias=True, activation=None)  # [bs, h, w, c']
    h_x = max_pool_layer(h_x)
    flattened_h_x = flatten_spatial_dimensions(h_x)

  attention_map = tf.matmul(flattened_g_x, flattened_f_x,
                            transpose_b=True)  # [bs, flat(h*w), flat(h*w)]
  attention_map = tf.nn.softmax(attention_map, axis=-1)
  flattened_out = tf.matmul(attention_map, flattened_h_x)  # [bs, flat(h*w), c']
  out = tf.reshape(flattened_out, shape=(batch_size, H, W, num_inner_channels))

  with tf.variable_scope('attention_conv'):
    attention_output = conv_layer(
        out, fin, kernel_size=1, strides=1, use_bias=True,
        activation=None)  # [bs, h, w, c]
  with tf.variable_scope('shortcut'):
    attention_weight = tf.get_variable(
        'attention_weight', shape=(), initializer=tf.constant_initializer(0.))
    x = attention_weight * attention_output + x

  return x

