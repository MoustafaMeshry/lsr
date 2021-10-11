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

## networks
"""Network models for image synthesis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import layers
from options import FLAGS as opts


ENCODER_NAME_SCOPE = 'Encoder'
GENERATOR_NAME_SCOPE = 'Generator'
DISCRIMINATOR_NAME_SCOPE = 'Discriminator'

# ------------------------------------------------------------------------------
# Auxiliary function.
# ------------------------------------------------------------------------------


def get_conv_layer(conv_type):
  """TBD."""
  if conv_type == 'scaled':
    return functools.partial(layers.Conv2DScaled, gain=np.sqrt(2))
  elif conv_type == 'spectral_norm':
    if not opts.faithful_spade_hacks:
      return functools.partial(layers.Conv2DSpectralNorm, padding='same')
    else:
      return functools.partial(
          layers.Conv2DSpectralNorm,
          padding='same',
          # kernel_initializer=layers.GlorotNormalInitializerWithGain(gain=0.02))
          kernel_initializer='glorot_normal')
  elif conv_type == 'weight_norm':
    return functools.partial(layers.Conv2DWeightNorm)
  elif conv_type == 'regular':
    if not opts.faithful_spade_hacks:
      return functools.partial(tf.keras.layers.Conv2D, padding='same')
    else:
      return functools.partial(
          tf.keras.layers.Conv2D,
          padding='same',
          # kernel_initializer=layers.GlorotNormalInitializerWithGain(gain=0.02))
          kernel_initializer='glorot_normal')
  else:
    raise ValueError('Invalid conv_type of "%s"!' % conv_type)


def get_dense_layer(layer_type):
  """TBD."""
  if layer_type == 'scaled':
    return functools.partial(layers.DenseScaled, gain=np.sqrt(2))
  elif layer_type == 'spectral_norm':
    if not opts.faithful_spade_hacks:
      return layers.DenseSpectralNorm
    else:
      return functools.partial(
          layers.DenseSpectralNorm,
          # kernel_initializer=layers.GlorotNormalInitializerWithGain(gain=0.02))
          kernel_initializer='glorot_normal')
  elif layer_type == 'weight_norm':
    return functools.partial(layers.DenseWeightNorm)
  elif layer_type == 'regular':
    if not opts.faithful_spade_hacks:
      return tf.keras.layers.Dense
    else:
      return functools.partial(
          tf.keras.layers.Dense,
          # kernel_initializer=layers.GlorotNormalInitializerWithGain(gain=0.02))
          kernel_initializer='glorot_normal')
  else:
    raise ValueError('Invalid layer_type of "%s"!' % layer_type)


def get_norm_layer(norm_type, param_free=False, order=layers.NHWC):
  """TBD."""
  if norm_type is None:
    return None
  if norm_type.startswith('param_free'):  # 'param_free[-_]*'
    param_free = True
    norm_type = norm_type[len('param_free_'):]

  if norm_type == 'instance_norm':
    center = scale = not param_free
    return functools.partial(
        tfa.layers.InstanceNormalization, center=center, scale=scale,
        # name='instance_normalization', epsilon=1e-5)
        name='instance_normalization')
  elif norm_type == 'layer_norm':
    center = scale = not param_free
    return functools.partial(
        layers.InstanceNormalization, center=center, scale=scale,
        # axis=None, name='layer_normalization', epsilon=1e-5)
        axis=None, name='layer_normalization')
    # return functools.partial(
    #     tf.keras.layers.LayerNormalization, center=center, scale=scale,
    #     name='layer_normalization')  # NOTE: bug with the TF implementation (see https://github.com/tensorflow/tensorflow/issues/39942)
  elif norm_type == 'pixel_norm':
    raise NotImplementedError('pixel/channel norm not yet wrapped in a class.')
  elif norm_type == 'batch_norm':
    center = scale = not param_free
    return functools.partial(
      tf.keras.layers.BatchNormalization, center=center, scale=scale,
      name='batch_norm')
  elif norm_type == 'sync_batch_norm':
    center = scale = not param_free
    return functools.partial(
      tf.keras.layers.experimental.SyncBatchNormalization, center=center,
      scale=scale, name='batch_norm')
      # scale=scale, momentum=0.1, name='batch_norm')
  elif norm_type == 'none':
    return functools.partial(layers.IdentityLayer, name='identity_layer')
  else:
    raise ValueError('Invalid norm_type value of "%s"!' % norm_type)


def get_upscale_layer(interpolation, order):
  """TBD."""
  assert interpolation in ['nearest', 'bilinear'], (
      'Invalid interpolation value of "%s"!' % interpolation)
  data_format = 'channels_last' if order == layers.NHWC else 'channels_first'
  return functools.partial(
      tf.keras.layers.UpSampling2D, size=2, data_format=data_format,
      interpolation=interpolation)


def get_activation_layer(nonlinearity):
  """TBD."""
  if nonlinearity == 'relu':
    return functools.partial(tf.keras.layers.ReLU)
  elif nonlinearity == 'lrelu':
    return functools.partial(tf.keras.layers.LeakyReLU, alpha=0.2)
  elif nonlinearity == 'prelu':
    return functools.partial(tf.keras.layers.PReLU)
  else:
    raise ValueError('Invalid nonlinearity value of "%s"!' % nonlinearity)


# ------------------------------------------------------------------------------
# Build wrappers for constructing D, E and G networks.
# ------------------------------------------------------------------------------


def build_discriminator(discriminator_arch,
                        nf=64,
                        num_downsamples=5,
                        conv_type='scaled',
                        norm_type='none',
                        nonlinearity='relu',
                        get_fmaps=False,
                        nf_max=512,
                        use_minibatch_stats=False,  # FIXME: not yet used.
                        name=None,
                        order=layers.NHWC):
  """Build the discriminator architecture as specified by --D_arch."""

  if discriminator_arch == 'resnet_stylegan2':
    # assert 2 ** (2 + num_downsamples) == opts.train_resolution
    return DiscriminatorStyleGAN2(
        resolution=opts.train_resolution,
        nf_start=nf,
        num_downsamples=num_downsamples,
        conv_type=conv_type,
        norm_type=norm_type,
        nonlinearity=nonlinearity,
        get_fmaps=get_fmaps,
        nf_max=nf_max,
        use_minibatch_stats=use_minibatch_stats,
        name=name,
        order=order)
  else:
    raise ValueError(
        'Unsupported discriminator architecture %s.' % discriminator_arch)


def build_encoder(encoder_arch,
                  nf=32,
                  conv_type='regular',
                  latent_size=512,
                  nf_max=512,
                  num_latents=1,
                  norm_type=None,
                  use_vae=False,
                  global_avg_pool=True,
                  add_fc_to_encoder=False,
                  normalize_latent=False,
                  name=None,
                  order=layers.NHWC):
  """Build the encoder network E as specified by --E_arch."""

  if encoder_arch == 'resnet':
    return EncoderResnet(
        nf_start=nf,
        num_downsamples=opts.E_num_downsamples,
        conv_type=conv_type,
        norm_type=norm_type,
        nonlinearity='relu',
        style_nc=num_latents*latent_size,
        nf_max=nf_max,
        vae_encoder=use_vae,
        global_avg_pool=global_avg_pool,
        add_linear_layer=add_fc_to_encoder,
        normalize_latent=normalize_latent,
        name=name,
        order=order)
  else:
    raise ValueError('Unsupported encoder architecture %s.' % encoder_arch)


def build_generator(g_arch,
                    nf=32,
                    output_nc=3,
                    output_nonlinearity=tf.math.tanh,
                    name=None,
                    order=layers.NHWC):
  """Build the generator network G as specified by --G_arch."""

  self_attention_layer_idx= -1 if not opts.use_self_attn else 2
  if g_arch == 'spade_gen':
    starting_res = opts.train_resolution // (2**opts.G_num_upsamples)
    assert starting_res * 2**opts.G_num_upsamples == opts.train_resolution, (
        '--train_resolution=%d is not compatible with --G_num_upsamples=%d' % (
            opts.train_resolution, opts.G_num_upsamples))
    return GeneratorSPADE(
        nf=nf,
        num_upsamples=opts.G_num_upsamples,
        conv_type=opts.conv_type,  # Note: should default to 'spectral_norm'.
        norm_type=opts.generator_norm_type,
        nonlinearity='lrelu',
        interpolation=opts.upsampling_mode,  # SPADE default is 'nearest'
        style_nc=opts.style_nc,
        starting_res=starting_res,
        nf_max=opts.g_nf_max,
        output_nc=output_nc,
        num_projection_filters=opts.spade_proj_filters,
        output_nonlinearity=output_nonlinearity,
        use_self_attention=False,
        self_attention_idx=-1,
        name=name,
        order=order)
  elif g_arch == 'unet_resnet':
    return UNetResnet(
        num_unet_blocks=opts.G_num_upsamples,
        num_bottleneck_blocks=opts.G_num_bottleneck_blocks,
        nf_start=nf,
        nf_max=opts.g_nf_max,
        output_nc=output_nc,
        conv_type=opts.conv_type,
        norm_type=opts.generator_norm_type,
        nonlinearity='lrelu',
        pool_type='blur',
        interpolation=opts.upsampling_mode,
        use_style_mod=True,
        output_nonlinearity=output_nonlinearity,
        use_skips=opts.use_skip_connections,
        self_attention_layer_idx=self_attention_layer_idx,
        name=name,
        order=order)
  elif g_arch == 'None':
    return tf.keras.Sequential(name=name)
  else:
    raise ValueError('Unsupported generator architecture %s.' % g_arch)


# ------------------------------------------------------------------------------
# Different encoder architectures.
# ------------------------------------------------------------------------------


class BaseEncoder(tf.keras.Model):
  """TBD."""

  def __init__(self,
               nf_start=64,
               num_downsamples=6,
               conv_type='scaled',
               norm_type='none',
               nonlinearity='relu',
               style_nc=512,
               nf_max=512,
               vae_encoder=False,
               global_avg_pool=True,
               add_linear_layer=False,
               normalize_latent=True,
               name=None,
               order=layers.NHWC,
               **kwargs):
    """Base encoder model."""
    super(BaseEncoder, self).__init__(name=name, **kwargs)
    self.nf_start = nf_start
    self.num_downsamples = num_downsamples
    self.style_nc = style_nc
    self.nf_max = nf_max
    self.vae_encoder = vae_encoder
    self.add_linear_layer = add_linear_layer
    self.normalize_latent = normalize_latent
    self.order = order
    # Some input arguments validation.
    assert not (vae_encoder and (
        normalize_latent or global_avg_pool)), (
            'If learning mu and sigma latents, then both `normalize_latent` and'
            '`global_avg_pool` should be False!')
    assert global_avg_pool or add_linear_layer or vae_encoder
    assert conv_type in ['scaled', 'spectral_norm', 'weight_norm', 'regular']
    assert nonlinearity in ['relu', 'lrelu', 'prelu']
    # Init layers and ops.
    self.conv_layer = get_conv_layer(conv_type)
    data_format = 'channels_last' if order == layers.NHWC else 'channels_first'
    # The following line is a hack to be consistent with SPADE authors' code;
    #  They use param-free norm ONLY with instance_norm in the discriminator!!
    if opts.faithful_spade_hacks:
      # Adding bias before instance normalization is meaningless.
      use_bias = False if 'instance_norm' in norm_type else True
    else:
      use_bias = True
    self.norm_layer = get_norm_layer(norm_type, order=order)
    self.conv_layer = functools.partial(
        self.conv_layer, use_bias=use_bias, padding='same',
        data_format=data_format)
    if opts.faithful_spade_hacks and conv_type == 'spectral_norm':
      self.dense_layer = get_dense_layer('regular')
    else:
      self.dense_layer = get_dense_layer(conv_type)
    self.act_layer = get_activation_layer(nonlinearity)

    self.blocks = []
    self.build_encoder_blocks()
    self.build_latent_layers(style_nc, vae_encoder, global_avg_pool,
                             add_linear_layer, normalize_latent)

  def build_encoder_blocks(self):
    pass

  def build_latent_layers(self, style_nc, vae_encoder, global_avg_pool,
                          add_linear_layer, normalize_latent):
    if global_avg_pool:
      self.blocks.append(functools.partial(
          layers.global_avg_pooling, keepdims=False, order=self.order))

    if vae_encoder:
      self.blocks.append(tf.keras.layers.Flatten())
      self.fc_mu = self.dense_layer(units=style_nc, name='dense_mu')
      self.fc_logvar = self.dense_layer(units=style_nc, name='dense_logvar')
    elif add_linear_layer:
      self.blocks.append(tf.keras.layers.Flatten())
      self.fc_z = self.dense_layer(units=style_nc, name='dense')
    else:
      self.fc_z = layers.IdentityLayer()

  def call(self, inputs, training=None):
    x = inputs
    for block_fn in self.blocks:
      if isinstance(block_fn, tf.keras.layers.Layer):
        x = block_fn(x, training=training)
      else:
        x = block_fn(x)
    if self.vae_encoder:
      z_mean = self.fc_mu(x, training=training)
      z_logvar = self.fc_logvar(x, training=training)
      z_logvar = tf.identity(z_logvar, name='z_logvar')
    else:
      z = self.fc_z(x, training=training)
      if self.normalize_latent:
        z = layers.channel_norm_via_dim(z, channel_dim=1)
      z_mean = z
      z_logvar = None
    z_mean = tf.identity(z_mean, name='z_mean')
    return z_mean, z_logvar
# ------------------------------------------------------------------------------


class EncoderResnet(BaseEncoder):
  """TBD."""

  def __init__(self,
               nf_start=64,
               num_downsamples=6,
               conv_type='scaled',
               norm_type='none',
               nonlinearity='relu',
               style_nc=512,
               nf_max=512,
               vae_encoder=False,
               global_avg_pool=True,
               add_linear_layer=True,
               normalize_latent=False,
               name=None,
               order=layers.NHWC,
               **kwargs):
    """Residual encoder model."""
    super(EncoderResnet, self).__init__(
        nf_start, num_downsamples, conv_type, norm_type, nonlinearity, style_nc,
        nf_max, vae_encoder, global_avg_pool, add_linear_layer,
        normalize_latent, name, order, **kwargs)

  def build_encoder_blocks(self):
    downscale_function = functools.partial(layers.downscale2d, n=2,
                                           order=self.order)
    nf = self.nf_start
    self.blocks.append(
        self.conv_layer(filters=nf, kernel_size=7, strides=1,
                        activation=self.act_layer(), name='from_rgb'))

    for i in range(self.num_downsamples - 1, -1, -1):
      if i == 0 and not (self.add_linear_layer or self.vae_encoder):
        # This is the last layer/block with no FC layer afterwards.
        next_nf = self.style_nc
      else:
        next_nf = min(nf * 2, self.nf_max)

      self.blocks.append(downscale_function)
      self.blocks.append(layers.ResBlock(
          nf, next_nf, self.conv_layer, self.act_layer, self.norm_layer,
          name=('downsample_resblock_%d' % i), order=self.order))
      nf = next_nf
    # Apply final activation after all Resnet blocks.
    self.blocks.append(self.act_layer())

# ------------------------------------------------------------------------------
# Different generator architectures.
# ------------------------------------------------------------------------------


class GeneratorSPADE(tf.keras.Model):
  """GauGAN (SPADE) generator module."""

  def __init__(self,
               nf=64,
               num_upsamples=5,
               conv_type='spectral_norm',
               norm_type='sync_batch_norm',  # MUST be param-free normalalization!
               nonlinearity='lrelu',  # Author's code use leaky_relu although the implementation details in the paper specifies relu.
               interpolation='nearest',
               style_nc=256,
               starting_res=8,
               nf_max=1024,
               output_nc=3,
               num_projection_filters=128,
               output_nonlinearity=tf.math.tanh,
               use_self_attention=False,
               self_attention_idx=-1,
               name=None,
               order=layers.NHWC,
               **kwargs):
    """Initializes the GauGAN (SPADE) generator module.

    Args:
      nf: Integer, number of filters at the last layer of the decoder (or in the
        first layer of a corresponding encoder).
      num_upsamples: Integer, number of upsampling blocks.
      conv_type: String, one of {scaled, spectral_norm, weight_norm, regular}.
      norm_type: String, normalization method; one of {instance, pixel}.
      nonlinearity: String, one of {relu, lrelu}.
      interpolation: String, one of {nearest, bilinear} used for upsampling.
      style_nc: Integer, dimensionality of the input style vector. A default
        value of zero means there is no input style and instead a convolution
        is performed on a downsampling of the input spatial map.
      starting_res: Integer, starting resolution at the first block.
      nf_max: Integer, max number of feature maps.
      output_nc: Integer, number of output channels.
      output_nonlinearity: callable, nonlinearity function to apply on the
        output channels.
      use_self_attention: Boolean, whether to apply self attention or not.
      self_attention_idx: Integer, layer index after which to insert self
        attention. This value is ignored if 'use_self_attention' is set to
        False. A default value of -1 means to insert self attention after
        a layer index of 'num_upsamples // 2'.
      name: String, optional name for the network.
      order: enum(NCHW, NHWC), the order of channels vs dimensions in an image.
      **kwargs: accepts tf.keras.Model arguments.
    """

    super(GeneratorSPADE, self).__init__(name=name, **kwargs)
    assert conv_type in ['scaled', 'spectral_norm', 'weight_norm', 'regular']
    assert interpolation in ['nearest', 'bilinear']
    assert nonlinearity in ['relu', 'lrelu']
    self.nf = nf
    # Set self_attention_idx to default value if needed.
    if use_self_attention and self_attention_idx == -1:
      self_attention_idx = num_upsamples // 2
    # Init layers and ops.
    self.num_projection_filters = num_projection_filters
    self.conv_layer = get_conv_layer(conv_type)
    if conv_type == 'scaled':
      self.to_rgb_conv_layer = functools.partial(self.conv_layer, gain=1)
      self.dense_layer = get_dense_layer(conv_type)
    elif opts.faithful_spade_hacks and conv_type == 'spectral_norm':
      self.conv_layer = functools.partial(self.conv_layer, use_bias=False)
      self.to_rgb_conv_layer = get_conv_layer('regular')
      self.dense_layer = get_dense_layer('regular')
    else:
      self.to_rgb_conv_layer = functools.partial(self.conv_layer, use_bias=True)
      self.dense_layer = get_dense_layer(conv_type)
    if opts.faithful_spade_hacks:
      spade_norm_conv_layer = functools.partial(
          tf.keras.layers.Conv2D,
          padding='same',
          use_bias=True,
          kernel_initializer='glorot_normal')
    else:
      spade_norm_conv_layer = functools.partial(self.conv_layer, use_bias=True)
    self.norm_layer = get_norm_layer(norm_type, param_free=True, order=order)
    self.act_layer = get_activation_layer(nonlinearity)
    self.upscale_layer = get_upscale_layer(interpolation, order)

    def _nf(layer_idx):
      return min(
          self.nf * (1 << (num_upsamples - layer_idx - 1)), nf_max)
    # Each entry of self.needs_spatial_input is a boolean to indicate whether
    #  the corresponding block in self.blocks needs an extra spatial map input.
    self.needs_spatial_input = []
    self.blocks = []
    nf = _nf(layer_idx=-1)
    if style_nc:
      filters_out = starting_res * starting_res * nf
      self.needs_spatial_input.append(False)
      self.blocks.append(self.dense_layer(filters_out))
      if order == layers.NHWC:
        shape_init = (-1, starting_res, starting_res, nf)
      else:
        shape_init = (-1, nf, starting_res, starting_res)
      self.needs_spatial_input.append(False)
      self.blocks.append(functools.partial(tf.reshape, shape=shape_init))
      self.needs_spatial_input.append(True)
      self.blocks.append(layers.SPADEResBlock(
          nf, nf, self.conv_layer, self.act_layer, self.norm_layer,
          projection_filters=self.num_projection_filters,
          spade_norm_conv_layer=spade_norm_conv_layer, order=order))
    else:
      raise NotImplementedError('TODO: support no style inputs as detailed '
                                'in the SPADE paper')
    # Decoder/upsampling layers.
    for layer_idx in range(num_upsamples):
      nf, next_nf = _nf(layer_idx - 1), _nf(layer_idx)
      self.needs_spatial_input.append(False)
      self.blocks.append(self.upscale_layer())
      self.needs_spatial_input.append(True)
      self.blocks.append(layers.SPADEResBlock(
          nf, next_nf, self.conv_layer, self.act_layer, self.norm_layer,
          projection_filters=self.num_projection_filters,
          spade_norm_conv_layer=spade_norm_conv_layer, order=order,
          name='spade_upsample_block_%d' % layer_idx))
      if use_self_attention and layer_idx == self_attention_idx:
        attention_layer = lambda x: x
        self.needs_spatial_input.append(False)
        self.blocks.append(attention_layer)
        raise NotImplementedError('TODO: self-attention not yet supported in '
                                  'TF 1.15!')
    # Convert to RGB output.
    self.needs_spatial_input.append(False)
    self.blocks.append(self.act_layer())
    self.to_rgb = self.to_rgb_conv_layer(output_nc, kernel_size=3, strides=1,
                                         padding='same', use_bias=True,
                                         activation=output_nonlinearity)

  def call(self, x_in, z_style=None, training=None):
    """TBD: add documentation."""
    x = z_style if z_style is not None else x_in
    for pass_spatial_input, block_fn in zip(self.needs_spatial_input,
                                            self.blocks):
      x = [x, x_in] if pass_spatial_input else x
      if isinstance(block_fn, tf.keras.layers.Layer):
        x = block_fn(x, training=training)
      else:
        x = block_fn(x)
    output = self.to_rgb(x, training=training)
    output = tf.identity(output, name='output')
    return output, x
# ------------------------------------------------------------------------------


class BaseUNet(tf.keras.Model):
  """TBD."""

  def __init__(self,
               encoder_layer,
               decoder_layer,
               bottleneck_layer=None,
               add_from_rgb_conv=True,
               num_unet_blocks=5,
               num_bottleneck_blocks=0,
               nf_start=64,
               nf_max=512,
               output_nc=3,
               conv_type='scaled',
               nonlinearity='lrelu',
               conv_kernel_size=3,
               output_nonlinearity=tf.math.tanh,
               use_skips=True,
               self_attention_layer_idx=-1,
               order=layers.NHWC,
               **kwargs):
    """TBD."""
    super(BaseUNet, self).__init__(**kwargs)
    self.use_skips = use_skips
    self.self_attention_layer_idx = self_attention_layer_idx
    self.encoder_blocks = []
    self.bottleneck_blocks = []
    self.decoder_blocks = []
    # Initialize convlution and activation layers.
    conv_layer = get_conv_layer(conv_type)
    if conv_type == 'scaled':
      to_rgb_conv_layer = functools.partial(conv_layer, gain=1)
    else:
      to_rgb_conv_layer = conv_layer
    act_layer = get_activation_layer(nonlinearity)
    # Input convolution (e.g from_rgb_conv).
    nf = nf_start
    if add_from_rgb_conv:
      self.input_conv = conv_layer(
          filters=nf_start, kernel_size=conv_kernel_size, strides=1,
          padding='SAME', use_bias=True, activation=act_layer(),
          name='input_conv')
    else:
      self.input_conv = None
    filters = []
    # Encoder blocks.
    for layer_idx in range(num_unet_blocks - 1, -1, -1):
      filters.append(nf)
      next_nf = min(nf * 2, nf_max)
      self.encoder_blocks.append(encoder_layer(
          next_nf,
          layer_idx,
          apply_self_attention=(layer_idx == self.self_attention_layer_idx),
          name='Downsample_block_%d' % layer_idx))
      nf = next_nf
    # Bottleneck blocks (if any).
    if num_bottleneck_blocks > 0:
      assert bottleneck_layer is not None
      for layer_idx in range(num_bottleneck_blocks):
        self.bottleneck_blocks.append(bottleneck_layer(
            nf, layer_idx, name='Bottleneck_block_%d' % layer_idx))
    # Decoder blocks.
    filters = filters[::-1]
    for layer_idx in range(num_unet_blocks):
      self.decoder_blocks.append(decoder_layer(
          filters[layer_idx],
          layer_idx,
          apply_self_attention=(layer_idx == self.self_attention_layer_idx),
          name='Upsample_block_%d' % layer_idx))
    # Output convolution.
    self.output_conv = to_rgb_conv_layer(
        filters=output_nc, kernel_size=conv_kernel_size, strides=1,
        padding='SAME', use_bias=True, activation=output_nonlinearity,
        name='output_conv')

  def call(self, x, z_style=None, training=None):
    skips = []
    if self.input_conv is not None:
      x = self.input_conv(x, training=training)
    for block_fn in self.encoder_blocks:
      x = block_fn(x, training=training)
      skips.append(x)
    for block_fn in self.bottleneck_blocks:
      x = block_fn(x, training=training)
    if self.use_skips:
      skips = skips[::-1]
    else:
      skips = [None] * len(self.decoder_blocks)
    # for i, (block_fn, skip) in enumerate(zip(self.decoder_blocks, skips)):
    for i, block_fn in enumerate(self.decoder_blocks):
      skip = skips[i + 1] if i < len(self.decoder_blocks) - 1 else None
      x = block_fn([x, z_style, skip], training=training)
    output = self.output_conv(x, training=training)
    return output, x
# ------------------------------------------------------------------------------


class ResnetEncoderBlock(tf.keras.layers.Layer):
  """Simple encoder block that applies a 2-strided convolution."""

  def __init__(self,
               nf,
               layer_idx,
               conv_type='scaled',  # should default to spectral_norm.
               norm_type=None,
               nonlinearity='relu',
               pool_type='blur',  # {average, blur}
               apply_self_attention=False,
               name=None,
               order=layers.NHWC,
               **kwargs):
    """TBD."""
    super(ResnetEncoderBlock, self).__init__(name=name, **kwargs)
    self.nf = nf
    self.layer_idx = layer_idx
    self.conv_layer = get_conv_layer(conv_type)
    self.norm_layer = get_norm_layer(
        norm_type, param_free=not opts.use_parametric_norm_in_unet_encoder,
        order=order)
    self.act_layer = get_activation_layer(nonlinearity)
    self.pool_type = pool_type
    self.apply_self_attention = apply_self_attention
    self.order = order
    if order == layers.NHWC:
      self.data_format = 'channels_last'
    else:
      self.data_format = 'channels_last'

  def build(self, input_shape):
    self.blocks = []
    if self.apply_self_attention:
      self.blocks.append(
          layers.SelfAttention(self.nf, self.conv_layer, order=self.order))
    # Residual block.
    self.blocks.append(layers.ResBlockDown(
        self.nf, self.conv_layer, self.act_layer, self.norm_layer,
        pool_type=self.pool_type, name='Resblock_down_%d' % self.layer_idx,
        order=self.order))
    super(ResnetEncoderBlock, self).build(input_shape)

  def call(self, x, training=None):
    for block_fn in self.blocks:
      x = block_fn(x, training=training)
    return x


class ResnetBottleneckBlock(layers.ResBlock):
  """Simple decoder block that applies upsampling followed by a convlution."""

  def __init__(self,
               nf,
               layer_idx,
               conv_type='scaled',
               norm_type='instance_norm',
               nonlinearity='lrelu',
               name=None,
               order=layers.NHWC,
               **kwargs):
    """TBD."""
    del layer_idx
    self.conv_layer = get_conv_layer(conv_type)
    self.norm_layer = get_norm_layer(
        norm_type, param_free=not opts.use_parametric_norm_in_unet_encoder,
        order=order)
    self.act_layer = get_activation_layer(nonlinearity)
    self.order = order
    super(ResnetBottleneckBlock, self).__init__(
        nf, nf, self.conv_layer, self.act_layer, self.norm_layer, name=name,
        order=order, **kwargs)


class ResnetDecoderBlock(tf.keras.layers.Layer):
  """Simple decoder block that applies upsampling followed by a convlution."""

  def __init__(self,
               nf,
               layer_idx,
               conv_type='scaled',
               norm_type='instance_norm',
               nonlinearity='lrelu',
               interpolation='bilinear',
               use_style_mod=True,
               apply_self_attention=False,
               name=None,
               order=layers.NHWC,
               **kwargs):
    """TBD."""
    super(ResnetDecoderBlock, self).__init__(name=name, **kwargs)
    self.nf = nf
    self.layer_idx = layer_idx
    self.conv_layer = get_conv_layer(conv_type)
    norm_layer = get_norm_layer(
        norm_type, param_free=use_style_mod, order=order)
    if use_style_mod:
      dense_layer = get_dense_layer(conv_type)
      self.norm_layer = functools.partial(
          layers.StyleModulation, dense_layer, param_free_norm=norm_layer,
          order=order, name='style_mod')
    else:
      self.norm_layer = norm_layer
    self.act_layer = get_activation_layer(nonlinearity)
    self.interpolation = interpolation
    self.use_style_mod = use_style_mod
    self.apply_self_attention = apply_self_attention
    self.order = order
    if order == layers.NHWC:
      self.data_format = 'channels_last'
    else:
      self.data_format = 'channels_last'

  def build(self, input_shape):
    # Residual block.
    self.resblock_up_fn = layers.ResBlockUp(
        self.nf, self.conv_layer, self.act_layer, self.norm_layer,
        interpolation=self.interpolation,
        name='Resblock_up_%d' % self.layer_idx, order=self.order)
    self.concatenate_skip = tf.keras.layers.Concatenate(axis=-1)
    if self.apply_self_attention:
      self.attn_layer = layers.SelfAttention(
          self.nf, self.conv_layer, order=self.order)
    else:
      self.attn_layer = None
    super(ResnetDecoderBlock, self).build(input_shape)

  def call(self, inputs, training=None):  # Expects 3 inputs.
    x, z_style, skip = inputs
    assert self.use_style_mod == (z_style is not None)
    x = self.resblock_up_fn(x, z_style, training=training)
    if self.attn_layer is not None:
      x = self.attn_layer(x, training=training)
    if skip is not None:
      x = self.concatenate_skip([x, skip])
    return x


class UNetResnet(BaseUNet):
  """TBD."""

  def __init__(self,
               num_unet_blocks=5,
               num_bottleneck_blocks=0,
               nf_start=64,
               nf_max=512,
               output_nc=3,
               conv_type='scaled',
               norm_type='instance_norm',
               nonlinearity='lrelu',
               pool_type='blur',  # {average, blur}
               interpolation='bilinear',
               use_style_mod=True,
               output_nonlinearity=tf.math.tanh,
               use_skips=True,
               self_attention_layer_idx=-1,
               order=layers.NHWC,
               **kwargs):
    """TBD."""
    encoder_layer = functools.partial(
        ResnetEncoderBlock, conv_type=conv_type, norm_type=norm_type,
        nonlinearity=nonlinearity, pool_type=pool_type, order=order)
    decoder_layer = functools.partial(
        ResnetDecoderBlock, conv_type=conv_type, norm_type=norm_type,
        nonlinearity=nonlinearity, interpolation=interpolation,
        use_style_mod=use_style_mod, order=order)
    bottleneck_layer = functools.partial(
        ResnetBottleneckBlock, conv_type=conv_type, norm_type=norm_type,
        nonlinearity=nonlinearity, order=order)
    super(UNetResnet, self).__init__(
        encoder_layer=encoder_layer,
        decoder_layer=decoder_layer,
        bottleneck_layer=bottleneck_layer,
        add_from_rgb_conv=False,
        num_unet_blocks=num_unet_blocks,
        num_bottleneck_blocks=num_bottleneck_blocks,
        nf_start=nf_start,
        nf_max=nf_max,
        output_nc=output_nc,
        conv_type=conv_type,
        conv_kernel_size=3,
        nonlinearity=nonlinearity,
        output_nonlinearity=output_nonlinearity,
        use_skips=use_skips,
        self_attention_layer_idx=self_attention_layer_idx,
        order=order,
        **kwargs)


# ------------------------------------------------------------------------------
# Different discriminator architectures.
# ------------------------------------------------------------------------------


class BaseDiscriminator(tf.keras.Model):
  """TBD."""

  def __init__(self,
               nf_start=64,
               num_downsamples=6,
               conv_type='scaled',
               norm_type='none',
               nonlinearity='relu',
               get_fmaps=False,
               nf_max=512,
               use_minibatch_stats=False,  # FIXME: not yet used.
               name=None,
               order=layers.NHWC,
               **kwargs):
    """Base discriminator model."""
    super(BaseDiscriminator, self).__init__(name=name, **kwargs)
    self.nf_start = nf_start
    self.num_downsamples = num_downsamples
    self.get_fmaps = get_fmaps
    self.nf_max = nf_max
    self.use_minibatch_stats = use_minibatch_stats
    self.order = order
    # Some input arguments validation.
    assert conv_type in ['scaled', 'spectral_norm', 'weight_norm', 'regular']
    assert nonlinearity in ['relu', 'lrelu', 'prelu']
    # Init layers and ops.
    self.conv_layer = get_conv_layer(conv_type)
    data_format = 'channels_last' if order == layers.NHWC else 'channels_first'
    self.conv_layer = functools.partial(self.conv_layer, use_bias=True,
                                        padding='same', data_format=data_format)
    self.dense_layer = get_dense_layer(conv_type)
    self.norm_layer = get_norm_layer(norm_type, order=order)
    self.act_layer = get_activation_layer(nonlinearity)

    self.blocks = []
    self.blocks_last_layer_flag = []
    self.compute_logits = None

  def build(self, input_shape):
    self.build_discriminator_blocks()

  def build_discriminator_blocks(self):
    pass

  def call(self, x, x_cond=None, training=None):
    # Concatenate extra conditioning input, if any.
    if x_cond is not None:
      channel_axis = 3 if self.order == layers.NHWC else 1
      x = tf.concat([x, x_cond], axis=channel_axis)

    disc_fmaps = []
    for keep_layer, layer in zip(self.blocks_last_layer_flag, self.blocks):
      if isinstance(layer, tf.keras.layers.Layer):
        x = layer(x, training=training)
      else:
        x = layer(x)
      if self.get_fmaps and keep_layer:
        disc_fmaps.append(x)
    # TODO(meshry): apply mini_batch stats and concatenation here if enabled.
    y = self.compute_logits(x, training=training)
    y = tf.identity(y, name='logits')
    disc_fmaps.append(y)
    return disc_fmaps
# ------------------------------------------------------------------------------


class DiscriminatorStyleGAN2(BaseDiscriminator):
  """TBD."""

  def __init__(self,
               resolution=256,
               nf_start=64,
               num_downsamples=6,
               conv_type='scaled',
               norm_type='none',
               nonlinearity='lrelu',
               get_fmaps=False,
               nf_max=512,
               use_minibatch_stats=True,
               name=None,
               order=layers.NHWC,
               **kwargs):
    self.order = order
    self.resolution_log2 = int(np.log2(resolution))
    # assert 2 ** self.resolution_log2 == resolution
    # assert num_downsamples == self.resolution_log2 - 2
    super(DiscriminatorStyleGAN2, self).__init__(
        nf_start=nf_start, num_downsamples=num_downsamples, conv_type=conv_type,
        norm_type=norm_type, nonlinearity=nonlinearity, get_fmaps=get_fmaps,
        nf_max=nf_max, use_minibatch_stats=use_minibatch_stats, name=name,
        order=order, **kwargs)

  def build_discriminator_blocks(self):
    downscale_function = functools.partial(layers.downscale2d, n=2,
                                           order=self.order)
    nf = self.nf_start
    self.blocks.append(
        self.conv_layer(filters=nf, kernel_size=1, strides=1, use_bias=True,
                        activation=self.act_layer(), name='from_rgb'))
    self.blocks_last_layer_flag.append(True)

    # Downsample blocks.
    for i in range(self.num_downsamples - 1, -1, -1):
      next_nf = min(nf * 2, self.nf_max)
      # self.blocks.append(downscale_function)
      # self.blocks_last_layer_flag.append(False)
      self.blocks.append(layers.ResBlockDown(
          next_nf, self.conv_layer, self.act_layer, self.norm_layer,
          pool_type='blur', mul_factor=1./np.sqrt(2),
          name=('Downsample_resblock_%d' % i), order=self.order))
      self.blocks_last_layer_flag.append(True)
      nf = next_nf

    # Apply final activation after all Resnet blocks.
    self.blocks.append(self.act_layer())
    self.blocks_last_layer_flag.append(False)
    if self.use_minibatch_stats:
      self.blocks.append(functools.partial(
          layers.minibatch_mean_stddev, order=self.order))
      self.blocks_last_layer_flag.append(False)

    # Conv
    self.blocks.append(self.conv_layer(
        filters=nf, kernel_size=3, strides=1, use_bias=True,
        activation=self.act_layer(), name='conv_4x4'))
    self.blocks_last_layer_flag.append(True)

    # Dense layer 4x4xnf -> nf
    self.blocks.append(tf.keras.layers.Flatten())
    self.blocks_last_layer_flag.append(False)
    self.blocks.append(self.dense_layer(
        units=nf, use_bias=True, activation=self.act_layer(), name='dense'))
    self.blocks_last_layer_flag.append(False)
    self.blocks.append(functools.partial(tf.reshape, shape=[-1, 1, 1, nf]))
    self.blocks_last_layer_flag.append(True)

    # Compute logits:
    self.compute_logits = tf.keras.Sequential(name='Disc_logits')
    # self.compute_logits.add(self.dense_layer(
    #     units=1, use_bias=True, activation=None, name='dense_logits'))
    self.compute_logits.add(self.conv_layer(
        filters=1, kernel_size=1, strides=1, use_bias=True,
        activation=None, name='conv2d_logits'))
# ------------------------------------------------------------------------------


class MultiScaleDiscriminator(tf.keras.Model):
  """TBD."""

  def __init__(self,
               num_scales=3,
               discriminator_arch='patch_gan',
               nf_start=64,
               num_downsamples=3,
               conv_type='scaled',
               norm_type='none',
               nonlinearity='lrelu',
               get_fmaps=False,
               nf_max=512,
               use_minibatch_stats=False,
               name='n_scale_disc',
               order=layers.NHWC,
               **kwargs):
    super(MultiScaleDiscriminator, self).__init__(name=name, **kwargs)
    self.get_fmaps = get_fmaps
    self.order = order
    discriminators = []
    for i in range(num_scales):
      discriminators.append(build_discriminator(
          discriminator_arch=discriminator_arch,
          nf=nf_start,
          num_downsamples=num_downsamples,
          conv_type=conv_type,
          norm_type=norm_type,
          nonlinearity=nonlinearity,
          get_fmaps=get_fmaps,
          use_minibatch_stats=use_minibatch_stats,
          name=('D_scale%d' % i),
          order=order))
    self.discriminators = discriminators

  def call(self, x, x_cond=None, training=None):
    if x_cond is not None:
      x = tf.concat([x, x_cond], axis=3)
    responses = []
    for ii, disc in enumerate(self.discriminators):
      # Pass x_cond=None since x_cond is concatenated above!
      responses.append(disc(x, x_cond=None, training=training))
      if ii != len(self.discriminators) - 1:
        x = layers.downscale2d(x, n=2, order=self.order)
    return responses
# ------------------------------------------------------------------------------

