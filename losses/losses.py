"""Loss functions for deep neural networks and GANs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os.path as osp
import tensorflow as tf
from losses.vgg_loss import VGGLoss
from losses.vgg_face_loss import VGGFaceLoss


class LPIPS(object):

  def __init__(
      self, net='alex_lin',
      frozen_graphs_parent_dir='/vulcanscratch/mmeshry/third_party/lpips'):
    """Frozen graphs can be downloaded from the following link:
        http://rail.eecs.berkeley.edu/models/lpips
    """
    frozen_graph_name = {
      'alex_lin': 'net-lin_alex_v0.1_27.pb',
      'alex': 'net_alex_v0.1_27.pb',
      'vgg_lin': 'net-lin_vgg_v0.1_27.pb',
      'vgg': 'net_vgg_v0.1_27.pb',
    }
    frozen_graph_path = osp.join(
        frozen_graphs_parent_dir, frozen_graph_name[net])
    inputs=['0:0', '1:0']
    outputs = {
      'alex_lin': 'Reshape_10:0',
      'alex': 'Add_20:0',
      'vgg_lin': 'Reshape_10:0',
      'vgg': 'Add_36:0',
    }
    with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())

    wrapped_import = tf.compat.v1.wrap_function(
        lambda: tf.compat.v1.import_graph_def(graph_def, name=''), [])
    imported_graph = wrapped_import.graph
    # layers = [op.name for op in imported_graph.get_operations()]
    # print('DBG: len(layers) = ', len(layers))
    # for i, layer in enumerate(layers):
    #   print(i, layer)
    self.func = wrapped_import.prune(
        tf.nest.map_structure(imported_graph.as_graph_element, inputs),
        tf.nest.map_structure(imported_graph.as_graph_element, outputs[net]))

  def __call__(self, x, y, axis=None):
    """Computes LPIPS loss value."""
    if not tf.is_tensor(x):
      x = tf.convert_to_tensor(x)
    if not tf.is_tensor(y):
      y = tf.convert_to_tensor(y)
    # Convert to NCHW format, which is the required format by AlexNet.
    x = tf.transpose(x, [0, 3, 1, 2])
    y = tf.transpose(y, [0, 3, 1, 2])
    lpips = self.func(x, y)
    if len(lpips.shape) > 1:
      return tf.math.reduce_mean(lpips, axis=axis)
    elif axis is not None:  # This is for net={alex,vgg}!
      # TODO: Debug the weird behavior of using net={alex,vgg} which returns a
      #  scalar.
      raise NotImplementedError(
          '`alex` and `vgg` from scratch return a scalar, not per-example loss '
          ' while my below attempts result in g_loss much larger than the sum '
          'of individual losses.')
      # NOTE: the returned scalar is either the sum of mean over the mini-batch.
      #  So, I need to figure this out fist.
      bs = x.shape.as_list()[0]
      # return tf.ones(shape=(bs, 1), dtype=lpips.dtype) * lpips  #  * bs
      # return tf.ones(shape=(bs, 1), dtype=lpips.dtype) * lpips / bs
      return tf.reshape(lpips, [-1, 1])  # * bs
    else:
      return lpips


def gradient_penalty_loss(y_xy, xy, iwass_target=1, iwass_lambda=10, axis=None):
  """TBD."""
  grad = tf.gradients(tf.reduce_sum(y_xy), [xy])[0]
  grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]) + 1e-8)
  loss_gp = tf.reduce_mean(
      tf.square(grad_norm - iwass_target),
      axis=axis) * iwass_lambda / iwass_target**2
  return loss_gp


def kl_loss(mean, logvar, axis=None):
  """TBD."""
  loss = 0.5 * tf.reduce_sum(
      tf.square(mean) + tf.exp(logvar) - 1. - logvar, axis=-1, keepdims=True)
  # return tf.reduce_mean(loss)
  return tf.reduce_sum(loss, axis=axis)  # just to match DRIT implementation


# TODO: fix the formula for L2 norm!! (and merge fix with the main lib.)
def l2_regularize(x, axis=None):
  """TBD."""
  return tf.reduce_mean(tf.square(x), axis=axis)


# TODO: fix the formula for L1 norm!! (and merge fix with the main lib.)
def l1_loss(x, y, axis=None):
  """TBD."""
  return tf.reduce_mean(tf.abs(x - y), axis=axis)


def disc_feature_loss(x_fmaps, y_fmaps, axis=None):
  """TBD."""
  assert len(x_fmaps) == len(y_fmaps), 'Inconsistent input feature maps.'
  loss = 0.
  for x_layer, y_layer in zip(x_fmaps, y_fmaps):
    loss += l1_loss(x_layer, y_layer, axis=axis)
  return loss


def multiscale_disc_feature_loss(x_multiscale_fmaps,
                                 y_multiscale_fmaps,
                                 axis=None):
  """TBD."""
  assert len(x_multiscale_fmaps) == len(y_multiscale_fmaps)
  num_discriminators = len(x_multiscale_fmaps)
  loss = 0.
  for x_cur_scale_fmaps, y_cur_scale_fmaps in zip(x_multiscale_fmaps,
                                                  y_multiscale_fmaps):
    loss += disc_feature_loss(x_cur_scale_fmaps, y_cur_scale_fmaps, axis=axis)
  return loss / num_discriminators


def hinge_gan_loss(disc_response, is_real, allow_negative_g_loss, axis=None):
  """TBD."""
  if not allow_negative_g_loss:
    sign = -1 if is_real else 1
    # The following works for both regular and patchGAN discriminators.
    loss = tf.reduce_mean(
        tf.maximum(0., 1 + sign * disc_response),
        axis=axis)
  else:
    assert is_real, 'Generator loss must always aim for real!'
    loss = tf.reduce_mean(-disc_response, axis=axis)
  return loss


def lsgan_loss(disc_response, is_real, axis=None):
  """TBD."""
  gt_label = 1 if is_real else 0
  # The following works for both regular and patchGAN discriminators
  loss = tf.reduce_mean(tf.square(disc_response - gt_label), axis=axis)
  return loss


def logistic_gan_saturating(disc_response, is_real, axis=None):
  """TBD."""
  sign = -1 if is_real else 1
  # Computes the following formula:
  #   if real: log(1 - logistic(disc_real_response))
  #   if fake: log(logistic(disc_fake_response))
  loss = tf.reduce_mean(sign * tf.nn.softplus(disc_response), axis=axis)
  return loss


def logistic_gan_nonsaturating(disc_response, is_real, axis=None):
  """TBD."""
  sign = -1 if is_real else 1
  # Computes the following formula:
  #   if real: -log(logistic(disc_real_response))
  #   if fake: -log(1 - logistic(disc_fake_response))
  loss = tf.reduce_mean(tf.nn.softplus(sign * disc_response), axis=axis)
  return loss


def multiscale_discriminator_loss(discs_responses,
                                  is_real,
                                  loss_type='hinge_gan',
                                  allow_negative_g_loss=False,
                                  axis=None):
  """TBD."""
  if allow_negative_g_loss:
    assert loss_type == 'hinge_gan', (
        '--allow_negative_g_loss is currently supported for '
        'hinge_gan loss only!')
  num_discriminators = len(discs_responses)
  loss = 0
  if loss_type == 'lsgan':
    gan_loss_fn = lsgan_loss
  elif loss_type == 'hinge_gan':
    gan_loss_fn = functools.partial(
        hinge_gan_loss, allow_negative_g_loss=allow_negative_g_loss)
  elif loss_type == 'logistic_saturating':
    gan_loss_fn = logistic_gan_saturating
  elif loss_type == 'logistic_nonsaturating':
    gan_loss_fn = logistic_gan_nonsaturating
  elif loss_type == 'wgan':
    raise NotImplementedError('WGAN loss not yet implemented!')
  elif loss_type == 'wgan_gp':
    raise NotImplementedError('WGAN-GP loss not yet implemented!')
  else:
    raise ValueError('--d_loss_type=%s is not supported' % loss_type)
  for i in range(num_discriminators):
    curr_response = discs_responses[i][-1]
    loss += gan_loss_fn(curr_response, is_real, axis=axis)
  return loss / num_discriminators

