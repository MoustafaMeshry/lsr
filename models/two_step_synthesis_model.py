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


from data import voxceleb_data_provider as voxceleb
from enum import Enum
from losses import losses
from models import model_utils
from models.base_model import *
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union
import functools
import layers
import networks
import numpy as np
import tensorflow as tf
import utils


class TwoStepSynthesisModel(BaseHeadSynthesisModel):

  def __init__(self,
               config: Dict,
               order: Enum=layers.NHWC):
    """
    Initializes a few-shot talking head synthesis model.

    This function should be called within `strategy.scope()` if using
      tf.disribute.Strategy.

    Args:
      - config: dictionary, model configuration and options (command line
         flags).
      - order: Enum, one of {layers.NHWC or NCHW} to specify the channel format
         of image tensors.
    """
    super(TwoStepSynthesisModel, self).__init__(config, order)
    ## Build networks.
    starting_res = self.config.train_resolution // (
        2**self.config.G_num_upsamples)
    assert starting_res * 2**self.config.G_num_upsamples == \
        self.config.train_resolution, (
            '--train_resolution=%d is not compatible with --G_num_upsamples=%d'
            % (self.config.train_resolution, self.config.G_num_upsamples))
    # Build Generator.
    if self.config.output_nonlinearity in [None, 'none', 'None']:
      output_nonlinearity = None
    elif self.config.output_nonlinearity == 'tanh':
      output_nonlinearity = tf.nn.tanh
    else:
      raise ValueError('Unsupported value of %s for --output_nonlinearity!' % (
          self.config.output_nonlinearity))

    layout_output_nc = self.config.num_segmentation_classes
    if self.config.multi_task_training:
      layout_output_nc += self.config.output_nc
    self._layout_generator = networks.build_generator(
        g_arch=self.config.segmap_G_arch,
        nf=self.config.segmap_g_nf,
        output_nc=layout_output_nc,
        output_nonlinearity=None,
        name=LAYOUT_GENERATOR_NAME_SCOPE,
        order=layers.NHWC)
    self._generator = networks.build_generator(
        g_arch=self.config.G_arch,
        nf=self.config.g_nf,
        output_nc=self.config.output_nc,
        output_nonlinearity=output_nonlinearity,
        name=GENERATOR_NAME_SCOPE,
        order=layers.NHWC)
    # Build Encoder.
    self._encoder = networks.build_encoder(
        self.config.E_arch,
        conv_type=self.config.e_conv_type,
        nf=self.config.e_nf,
        latent_size=self.config.num_latent_embeddings * self.config.style_nc,
        norm_type=self.config.encoder_norm_type,
        use_vae=self.config.use_vae,
        global_avg_pool=self.config.encoder_global_avg_pool,
        add_fc_to_encoder=self.config.add_fc_to_encoder,
        normalize_latent=self.config.normalize_latent,
        name=ENCODER_NAME_SCOPE,
        order=self.order)
    if self.config.separate_layout_encoder:
      assert self.config.num_latent_embeddings == 1
      self._layout_encoder = networks.build_encoder(
          self.config.segmap_E_arch,
          conv_type=self.config.e_conv_type,
          nf=self.config.segmap_e_nf,
          latent_size=self.config.style_nc,
          norm_type=self.config.encoder_norm_type,
          use_vae=self.config.use_vae,
          global_avg_pool=self.config.encoder_global_avg_pool,
          add_fc_to_encoder=self.config.add_fc_to_encoder,
          normalize_latent=self.config.normalize_latent,
          name=LAYOUT_ENCODER_NAME_SCOPE,
          order=self.order)
    # Placeholders for other train/finetune networks and optimizers.
    self._discriminator = None
    self._vgg_loss = None
    self._vgg_face_loss = None
    self._lpips_loss = None
    self._g_optimizer = None
    self._g_layout_optimizer = None
    self._d_optimizer = None
    # Eval metrics:
    self.eval_metrics_dict = {}
    self.eval_metrics_dict['mean_absolute_error'] = \
        tf.keras.metrics.MeanAbsoluteError(name='mean_l1')
    self.eval_metrics_dict['mean_psnr'] = tf.keras.metrics.Mean(
        name='mean_psnr')
    self.eval_metrics_dict['mean_ssim'] = tf.keras.metrics.Mean(
        name='mean_ssim')
    for loss_key in [
        'loss_d_real', 'loss_d_fake', 'loss_d', 'loss_g_gan', 'loss_g_feat',
        'loss_g_vgg', 'loss_g_l1', 'loss_g_lpips', 'loss_g_kl', 'loss_z_l2',
        'loss_z_layout_l2', 'loss_g_vgg_face_recon_and_id',
        'loss_g_segmentation', 'loss_g']:
      self.eval_metrics_dict['losses/%s' % loss_key] = tf.keras.metrics.Mean(
          name=loss_key)

  def init_extra_train_and_eval_networks(self):
    """Initializes train losses, networks and optimizers.
    
    This function should be called within `strategy.scope()` if using
      tf.distribute.Strategy. It initilizes the following components:
      - self._vgg_loss: initialized only if --w_loss_vgg > 0.
      - self._vgg_face_loss: initialized only if --w_loss_vgg_face_recon > 0 or
         --w_loss_identity > 0.
      - self._lpips_loss: initialized only if --w_loss_lpips > 0.
      - self._discriminator network: initialized only if --w_loss_gan > 0 or
          --w_loss_feat > 0.
    """
    if self.train_and_eval_networks_initialized:
      return
    # Build VGG network.
    if self._vgg_loss is None and self.config.w_loss_vgg:
      self._vgg_loss = losses.VGGLoss()
    # Build VGGFace network.
    if self._vgg_face_loss is None and (
          self.config.w_loss_vgg_face_recon or self.config.w_loss_identity):
      self._vgg_face_loss = losses.VGGFaceLoss()
    # Build AlexNet network for LPIPS.
    if self._lpips_loss is None and self.config.w_loss_lpips:
      self._lpips_loss = losses.LPIPS(net=self.config.lpips_net_type)
    # Build discriminator.
    if self._discriminator is None and (
          self.config.w_loss_gan or self.config.w_loss_feat):
      self._discriminator = networks.MultiScaleDiscriminator(
          num_scales=self.config.num_disc_scales,
          discriminator_arch=self.config.D_arch,
          nf_start=self.config.d_nf,
          num_downsamples=self.config.D_num_downsamples,
          conv_type=self.config.d_conv_type,
          norm_type=self.config.discriminator_norm_type,
          nonlinearity='lrelu',
          get_fmaps=True,
          nf_max=512,
          use_minibatch_stats=False,
          name=DISCRIMINATOR_NAME_SCOPE,
          order=layers.NHWC)
    self.train_and_eval_networks_initialized = True

  def create_optimizers(
      self,
      lr_warmstart_steps: int,
      decay_start_step: int,
      decay_end_step: int,
      decay_num_intervals: int,
      starting_step: Optional[int]=0,
      lr_mul_factor: Optional[float]=1.) -> Dict[Text,
                                                 tf.keras.optimizers.Optimizer]:
    """Initializes optimizers for training.

    This function should be called within `strategy.scope()` if using
      tf.distribute.Strategy. It initilizes the following components:
      - self._g_optimizer: this optimizer handles all inference variables (e.g.
          encoder and generator).
      - self._g_layout_optimizer: initilized only if --separate_layout_optimizer
          is set to true. This optimizer handles the segmentation loss
          optimization.
      - self._d_optimizer: initilized only if self._discriminator was
          initialized. This optimizer handles discriminator variables.

    Args:
      - lr_warmstart_steps: int, number of steps to apply learning rate warmup.
      - decay_start_step: int, train step at which to start learning rate decay.
      - decay_end_step: int, train step at which to end learning rate decay.
      - decay_num_intervals: int, factor by which to decay the learning rate;
          final learning rate = initial_learning_rate / `decay_num_intervals`.
      - starting_step: int, the train starting step. This is zero when training
         from scratch, or the loaded train step for finetuning a pre-trained
         model.
      - lr_mul_factor: optional float, multiplier factor for the learning rate;
          mainly used to increase the learning rate w.r.t the number of gpus.

    Returns:
      A dictionary with all the otpimizers of the model training.
    """
    if self.optimizers_initialized:
      return self.get_optimizers()
    # Build optimizer and learning rate schedule for the generator/encoder.
    g_lr_callback = functools.partial(
        model_utils.get_learning_rate_schedule,
        lr_init=tf.constant(lr_mul_factor * self.config.g_lr, tf.dtypes.float32),
        global_step=tf.compat.v1.train.get_or_create_global_step(),
        lr_warmstart_steps=tf.constant(lr_warmstart_steps, tf.dtypes.float32),
        decay_start_step=tf.constant(decay_start_step, tf.dtypes.float32),
        decay_end_step=tf.constant(decay_end_step, tf.dtypes.float32),
        decay_num_intervals=tf.constant(decay_num_intervals),
        starting_step=starting_step,
        name='g_lr')
    self._g_optimizer = tf.keras.optimizers.Adam(
        learning_rate=g_lr_callback,
        beta_1=self.config.adam_beta1,
        beta_2=self.config.adam_beta2,
        name='Adam_opt_G')
    if (self.config.separate_layout_optimizer and
        self.config.w_loss_segmentation > 0):
      self._g_layout_optimizer = tf.keras.optimizers.Adam(
        learning_rate=g_lr_callback,
        beta_1=self.config.adam_beta1,
        beta_2=self.config.adam_beta2,
        name='Adam_opt_G_layout')
    # Build optimizer and learning rate schedule for the discriminator.
    d_lr_callback = functools.partial(
        model_utils.get_learning_rate_schedule,
        lr_init=tf.constant(lr_mul_factor * self.config.d_lr, tf.dtypes.float32),
        global_step=tf.compat.v1.train.get_or_create_global_step(),
        lr_warmstart_steps=tf.constant(lr_warmstart_steps, tf.dtypes.float32),
        decay_start_step=tf.constant(decay_start_step, tf.dtypes.float32),
        decay_end_step=tf.constant(decay_end_step, tf.dtypes.float32),
        decay_num_intervals=tf.constant(decay_num_intervals),
        starting_step=starting_step,
        name='d_lr')
    self._d_optimizer = tf.keras.optimizers.Adam(
        learning_rate=d_lr_callback,
        beta_1=self.config.adam_beta1,
        beta_2=self.config.adam_beta2,
        name='Adam_opt_D')
    self.optimizers_initialized = True
    return self.get_optimizers()

  def compute_losses(
      self,
      real: tf.Tensor,
      segmap_gt: tf.Tensor,
      outputs_dict: Dict[Text, tf.Tensor],
      training: bool,
      fg_mask: Union[float, tf.Tensor]=1.,
      conditional_inputs: Optional[tf.Tensor]=None,
      gradient_tape: Optional[tf.GradientTape]=None) -> Dict[Text, tf.Tensor]:
    """Computes and returns per-example losses of a mini-batch.

    Args:
      - real: 4D tensor, the ground truth output.
      - segmap_gt: 4D tensor, the label map of the semantic segmentation (
         shape = [batch_size, H, W, 1]).
      - outputs_dict: dict string->tf.Tensor, all intermediate and final outputs.
      - training: boolean, whether or not to run the networks in training mode.
      - fg_mask: Optional 4D tensor, a mask image to apply to the final output
         and ground truth. Default is a scalar 1, which leaves the output and
         ground truth unchanged.
      - conditional_inputs: Optional 4D tensor, the conditional input the
         generator network. This is used for the conditional discriminator.
      - gradient_tape: Optional tf.GradientTape, tensorflow's gradient_tape
         for gradient penalty computation (if any).

    Returns:
      A dictionary (str->tf.Tensor), the value of each entry is a 1-D tensor
       of length equal to the mini-batch size, representing the per-example loss
       values.
    """
    fake = outputs_dict['output']
    z_style, z_layout = outputs_dict['latents']
    # Watch discriminator inputs if needed for gradient penalty.
    if training and (self.config.r1_gamma or self.config.r2_gamma):
      assert gradient_tape is not None and self._discriminator is not None
      if self.config.r1_gamma:
        gradient_tape.watch(real)
      if self.config.r2_gamma:
        gradient_tape.watch(fake)
    # Apply foreground mask.
    real = real * fg_mask
    fake = fake * fg_mask

    per_example_loss_dict = {}
    # Compute discriminator logits if needed.
    loss_d_real = loss_d_fake = loss_d = tf.zeros([])
    if self.config.w_loss_gan or self.config.w_loss_feat:
      if self.config.use_conditional_disc:
        assert conditional_inputs is not None
      disc_real_featmaps = self._discriminator(
          real, conditional_inputs, training=training)
      disc_fake_featmaps = self._discriminator(
          fake, conditional_inputs, training=training)
    else:
      disc_real_featmaps = disc_fake_featmaps = []

    # Generator losses.
    per_example_loss_g = 0
    # GAN loss.
    if self.config.w_loss_gan > 0:
      per_example_loss_d_real = losses.multiscale_discriminator_loss(
          disc_real_featmaps, is_real=True, loss_type=self.config.d_loss_type,
          axis=(1, 2, 3))
      per_example_loss_d_fake = losses.multiscale_discriminator_loss(
          disc_fake_featmaps, is_real=False, loss_type=self.config.d_loss_type,
          axis=(1, 2, 3))
      per_example_loss_d = per_example_loss_d_real + per_example_loss_d_fake
      per_example_loss_dict['loss_d_real'] = per_example_loss_d_real
      per_example_loss_dict['loss_d_fake'] = per_example_loss_d_fake
      per_example_loss_dict['loss_d'] = per_example_loss_d

      # Generator GAN loss.
      if self.config.d_loss_type == 'hinge_gan':
        allow_negative_g_loss = True
      else:
        allow_negative_g_loss = False
      per_example_loss_g_gan = losses.multiscale_discriminator_loss(
          disc_fake_featmaps, is_real=True, loss_type=self.config.d_loss_type,
          allow_negative_g_loss=allow_negative_g_loss, axis=(1, 2, 3))
      per_example_loss_dict['loss_g_gan'] = (
          self.config.w_loss_gan * per_example_loss_g_gan)
      per_example_loss_g += per_example_loss_dict['loss_g_gan']

    # Discriminator-based perceptual/feature loss.
    if self.config.w_loss_feat > 0:
      per_example_loss_g_feat = losses.multiscale_disc_feature_loss(
          disc_real_featmaps, disc_fake_featmaps, axis=(1, 2, 3))
      per_example_loss_dict['loss_g_feat'] = (
          self.config.w_loss_feat * per_example_loss_g_feat)
      per_example_loss_g += per_example_loss_dict['loss_g_feat']

    # VGG loss.
    per_example_loss_g_vgg = 0
    if self.config.w_loss_vgg:
      per_example_loss_g_vgg = self._vgg_loss(real, fake, axis=(1, 2, 3))
      per_example_loss_dict['loss_g_vgg'] = (
          self.config.w_loss_vgg * per_example_loss_g_vgg)
      per_example_loss_g += per_example_loss_dict['loss_g_vgg']

    # L1 loss.
    per_example_loss_g_l1 = 0
    if self.config.w_loss_l1:
      per_example_loss_g_l1 = losses.l1_loss(fake, real, axis=(1, 2, 3))
      per_example_loss_dict['loss_g_l1'] = (
          self.config.w_loss_l1 * per_example_loss_g_l1)
      per_example_loss_g += per_example_loss_dict['loss_g_l1']

    # LPIPS loss
    per_example_loss_g_lpips = 0
    if self.config.w_loss_lpips:
      per_example_loss_g_lpips = self._lpips_loss(real, fake, axis=(1, 2, 3))
      per_example_loss_dict['loss_g_lpips'] = (
          self.config.w_loss_lpips * per_example_loss_g_lpips)
      per_example_loss_g += per_example_loss_dict['loss_g_lpips']

    # Latent l2 regularization.
    if self.config.w_loss_z_l2:
      per_example_loss_z_l2 = losses.l2_regularize(z_style, axis=1)
      per_example_loss_dict['loss_z_l2'] = (
          self.config.w_loss_z_l2 * per_example_loss_z_l2)
      per_example_loss_g += per_example_loss_dict['loss_z_l2']

    if self.config.w_loss_z_layout_l2:
      assert z_layout is not None
      per_example_loss_z_layout_l2 = losses.l2_regularize(z_layout, axis=1)
      per_example_loss_dict['loss_z_layout_l2'] = (
          self.config.w_loss_z_layout_l2 * per_example_loss_z_layout_l2)
      per_example_loss_g += per_example_loss_dict['loss_z_layout_l2']
    
    # vgg_face-based reconstruction loss.
    per_example_loss_g_vgg_face_recon = 0
    per_example_loss_g_identity = 0
    if self.config.w_loss_vgg_face_recon or self.config.w_loss_identity:
      layer_idxs = ()
      layer_weights = ()
      if self.config.w_loss_vgg_face_recon:
        layer_idxs += (1, 6, 11, 18, 25)
        layer_weights += tuple(self.config.w_loss_vgg_face_recon * np.ones(
            len(layer_idxs)))
      if self.config.w_loss_identity:
        layer_idxs += (-3,)
        layer_weights += (self.config.w_loss_identity,)
      per_example_loss_g_vgg_face_recon = self._vgg_face_loss(
          real, fake, layer_idxs=layer_idxs, layer_weights=layer_weights,
          axis=(1, 2, 3))
      per_example_loss_dict[
          'loss_g_vgg_face_recon_and_id'] = per_example_loss_g_vgg_face_recon
      per_example_loss_g += per_example_loss_dict[
          'loss_g_vgg_face_recon_and_id']

    # Semantic segmentation loss.
    per_example_loss_g_segmentation = 0
    if self.config.w_loss_segmentation:
      segmap_logits = outputs_dict['segmap_logits']
      per_example_loss_g_segmentation = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=segmap_gt, logits=segmap_logits),
          axis=(1, 2))
      per_example_loss_dict[
          'loss_g_segmentation'] = (
              self.config.w_loss_segmentation * per_example_loss_g_segmentation)
      # Add the segmentation loss to the generator loss only if not using a
      #  separate optimizer for the segmentation tower.
      if not self.config.separate_layout_optimizer:
        per_example_loss_g += per_example_loss_dict[
            'loss_g_segmentation']

    # Combined loss for the generator and encoder.
    per_example_loss_dict['loss_g'] = per_example_loss_g

    # Gradient Penalty:
    if training:
      d_vars = self._discriminator.trainable_variables
      if self.config.r1_gamma > 0:
        r1_gamma = self.config.r1_gamma
        real_grads = gradient_tape.gradient(
            per_example_loss_dict['loss_d_real'], [real])[0]
        r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
        per_example_loss_dict['real_gp'] = r1_gamma * 0.5 * r1_penalty
        per_example_loss_dict['loss_d'] += per_example_loss_dict['real_gp']
      if self.config.r2_gamma > 0:
        r2_gamma = self.config.r2_gamma
        fake_grads = gradient_tape.gradient(
            per_example_loss_dict['loss_d_fake'], [fake])[0]
        r2_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1, 2, 3])
        per_example_loss_dict['fake_gp'] = r2_gamma * 0.5 * r2_penalty
        per_example_loss_dict['loss_d'] += per_example_loss_dict['fake_gp']

    return per_example_loss_dict

  def compute_latents(
      self,
      encoder_inputs: tf.Tensor,
      num_few_shots: int,
      training: bool,
      use_vae: bool=False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Computes layout and style latents given the input to the encoder(s).

    Args:
      - encoder_inputs: 4D or 5D tensor, the input the encoder network.
      - num_few_shots: integer, number of few-shot inputs to the encoder.
      - training: boolean, whether or not to run the networks in training mode.
      - use_vae: boolean, whether the encoder is variational or not. If use_vae
         is true AND training is true, then noise sampled from N(0,1) is added
         to the standard deviaiton of the style latent.

    Returns: a 2-tuple represeting the style and layout latents respectively.
    """
    assert not use_vae, 'VAE-based few-shot encoder is not supported.'
    assert (self.config.num_latent_embeddings == 2) or (
        self.config.separate_layout_encoder)
    z_layout = None
    if num_few_shots == 0:
      z, _ = self._encoder(encoder_inputs, training=training)
      if self.config.separate_layout_encoder:
        z_layout, _ = self._layout_encoder(encoder_inputs, training=training)
    elif num_few_shots == 1 and len(encoder_inputs.shape) > 4:
      z, _ = self._encoder(tf.squeeze(encoder_inputs, axis=[1]),
                           training=training)
      if self.config.separate_layout_encoder:
        z_layout, _ = self._layout_encoder(tf.squeeze(encoder_inputs, axis=[1]),
                                           training=training)
    else:
      dims = encoder_inputs.shape
      # Convert to a 4D tensor by re-arraning to cancel the few-shot dimension.
      encoder_inputs = tf.reshape(
          encoder_inputs, [-1, dims[2], dims[3], dims[4]])
      z, _ = self._encoder(encoder_inputs, training=training)
      # Reshape back to a 5D tensor to retrieve the few-shot dimension.
      z = tf.reshape(
          z, [-1, num_few_shots,
              self.config.num_latent_embeddings * self.config.style_nc])
      # Compute a single latent by reduce_mean over the few-shot dimension.
      z = tf.reduce_mean(z, axis=1, keepdims=False)
      if self.config.separate_layout_encoder:
        z_layout, _ = self._layout_encoder(encoder_inputs, training=training)
        # Reshape back to a 5D tensor to retrieve the few-shot dimension.
        z_layout = tf.reshape(
            z_layout, [-1, num_few_shots, self.config.style_nc])
        # Compute a single latent by reduce_mean over the few-shot dimension.
        z_layout = tf.reduce_mean(z_layout, axis=1, keepdims=False)
    if self.config.separate_layout_encoder:
      z_style = z
    else:
      z_style, z_layout = tf.split(z, num_or_size_splits=2, axis=-1)
    assert z_layout is not None
    return z_style, z_layout

  def process(
      self,
      encoder_inputs: tf.Tensor,
      conditional_inputs: tf.Tensor,
      training: bool,
      precomputed_latents: Optional[Tuple[tf.Tensor, ...]]=None,
      ) -> Dict[Text, Union[tf.Tensor, Tuple[tf.Tensor, ...]]]:
    """Runs the forward pass and returns all intermediate and final outputs.

    Args:
      - encoder_inputs: 4D or 5D tensor, the input the encoder network.
      - conditional_inputs: 4D tensor, the input the generator network.
      - training: boolean, whether or not to run the networks in training mode.
      - precomputed_latents: Optional 2-tuple of tf.Tensor, pre-computed latent
         codes for the input mini-batch. If not None, then the encoder network
         is not run, and the pre-computed latents are used instead. If a single,
         latent is being used, then the 2nd element in the tuple is None.

    Returns: a dictionary holding all intermediate and final outputs.
    """
    if precomputed_latents is not None:
      assert encoder_inputs is None
      z_style, z_layout = precomputed_latents
    else:
      z_style, z_layout = self.compute_latents(
          encoder_inputs=encoder_inputs,
          num_few_shots=self.config.K,
          training=training)
    segmap_logits, _ = self._layout_generator(
        conditional_inputs, z_layout, training=training)
    if self.config.multi_task_training:
      segmap_logits, aux_frame_recon = tf.split(
          segmap_logits,
          num_or_size_splits=[self.config.num_segmentation_classes,
                              self.config.output_nc],
          axis=-1)
      if self.config.output_nonlinearity == 'tanh':
        aux_frame_recon = tf.nn.tanh(aux_frame_recon)
    else:
      aux_frame_recon = None
    segmap_prob_map = tf.nn.softmax(segmap_logits, axis=-1)
    channel_axis = 3 if self.order == layers.NHWC else 1
    segmap_label_map = tf.math.argmax(segmap_logits, axis=channel_axis,
                                      output_type=tf.dtypes.int32)
    if self.config.concat_contours_to_layout:
      spatial_input = tf.concat([conditional_inputs, segmap_prob_map],
                                axis=channel_axis)
    else:
      spatial_input = segmap_prob_map
    output, _ = self._generator(spatial_input, z_style, training=training)
    return {
        'latents': (z_style, z_layout),
        'segmap_logits': segmap_logits,
        'segmap_prob_map': segmap_prob_map,
        'segmap_label_map': segmap_label_map,
        'aux_frame_recon': aux_frame_recon,
        'output': output,
    }

  def train(
      self,
      inputs_dict: Dict[Text, tf.Tensor],
      global_batch_size: int,
      train_g_step: bool=True,
      train_d_step: bool=True) -> Tuple[
          Dict[Text, tf.Tensor], Dict[Text, Any], Tuple[Dict[Text, Any], ...]]:
    """Runs a train step over the input mini/sub-mini batch.

    Runs the training step over the input minibatch and aggregates the train
     losses over the "global" batch size.

    Args:
      - inputs_dict: dictionary of strings->tensors representing an input
         minibatch.
      - global_batch_size: integer representing the "global" minibatch size,
         which is equal to one minibatch_size * num_gpus.
      - train_g_step: boolean, whether to update the generator weights or not.
      - train_d_step: boolean, whether to update the discriminator weights or
         not.

    Returns: a 3-tuple:
      - loss_dict: dictionary of all train losses aggregated according to the
         global batch size.
      - outputs_dict: dict string->tf.Tensor, all intermediate and final outputs.
      - summaries: A 3-tuple representing scalara, image and text summaries.
         returend by _add_summaries(). See _add_summaries() for more details.
    """
    (encoder_inputs, conditional_inputs, real, target_segmap,
     basename, precomputed_latents) = self.parse_inputs(
        inputs_dict,
        model_utils.Mode.TRAIN,
        augmentation=self.config.data_augmentation)
    with tf.GradientTape(persistent=True) as tape:
      outputs_dict = self.process(
          encoder_inputs, conditional_inputs, training=True,
          precomputed_latents=precomputed_latents)

      # Compute losses.
      fg_mask = utils.get_foreground_mask(
          conditional_inputs, self.config.synthesize_background)
      per_example_loss_dict = self.compute_losses(
          real,
          target_segmap,
          outputs_dict,
          training=True,
          fg_mask=fg_mask,
          conditional_inputs=conditional_inputs,
          gradient_tape=tape)

      # Aggregate losses over minibatch.
      loss_dict = {}
      for loss_key, loss_val in per_example_loss_dict.items():
        loss_dict[loss_key] = tf.nn.compute_average_loss(
            per_example_loss=loss_val, global_batch_size=global_batch_size)

    # Apply generator train step.
    if train_g_step:
      loss_g = loss_dict['loss_g']
      g_layout_vars = self._layout_generator.trainable_variables
      g_vars = self._generator.trainable_variables
      e_vars = self._encoder.trainable_variables
      subject_ft = (self.config.run_mode in ['subject_finetuning', 'infer'])
      if self.config.separate_layout_encoder:
        e_layout_vars = self._layout_encoder.trainable_variables
      else:
        e_layout_vars = []
      if not subject_ft:
        g_vars_all = g_layout_vars + e_layout_vars + g_vars + e_vars
      else:
        g_vars_all = []
        if self.config.finetune_g: g_vars_all += g_vars
        if self.config.finetune_g_layout: g_vars_all += g_layout_vars
        if self.config.finetune_e: g_vars_all += e_vars
        if self.config.finetune_e_layout: g_vars_all += e_layout_vars
      if len(g_vars_all):
        train_g_op = model_utils.get_train_op(
            tape, self._g_optimizer, loss_g, g_vars_all,
            grad_clip_abs_val=self.config.grad_clip_abs_val)
      if self._g_layout_optimizer is not None:
        loss_g_segmentation = loss_dict['loss_g_segmentation']
        g_layout_vars_all = []
        if not subject_ft or self.config.finetune_g_layout:
          g_layout_vars_all += g_layout_vars
        if self.config.separate_layout_encoder:
          e_layout_vars_to_add = e_layout_vars
        else:
          e_layout_vars_to_add = e_vars
        if not subject_ft or self.config.finetune_e_layout:
          g_layout_vars_all += e_layout_vars_to_add
        if len(g_layout_vars_all):
          train_g_layout_op = model_utils.get_train_op(
            tape, self._g_layout_optimizer, loss_g_segmentation,
            g_layout_vars_all, grad_clip_abs_val=self.config.grad_clip_abs_val)
    # Apply discriminator train step.
    if train_d_step:
      if self._discriminator is not None:
        loss_d = loss_dict['loss_d']
        d_vars = self._discriminator.trainable_variables
        if not subject_ft or self.config.finetune_d:
          train_d_op = model_utils.get_train_op(
              tape, self._d_optimizer, loss_d, d_vars,
              grad_clip_abs_val=self.config.grad_clip_abs_val)

    # Add tensorboard summaries.
    summaries = self._add_summaries(
        encoder_inputs=encoder_inputs,
        target_landmarks=conditional_inputs,
        target_segmap=target_segmap,
        real=real,
        outputs_dict=outputs_dict,
        person_id=inputs_dict[voxceleb.PERSON_ID_KEY],
        video_id=inputs_dict[voxceleb.VIDEO_PART_ID_KEY],
        video_part_id=inputs_dict[voxceleb.VIDEO_PART_ID_KEY],
        fg_mask=fg_mask,
        input_basename=basename)

    # Return train losses, outputs and train summaries.
    return (
        loss_dict, outputs_dict, summaries)

  def evaluate(
      self,
      inputs_dict: Dict[Text, tf.Tensor],
      global_batch_size: int) -> Tuple[
          Dict[Text, tf.Tensor], Dict[Text, Any], Tuple[Dict[Text, Any], ...]]:
    """Runs an evaluation step and updates evaluation metrics.

    Runs the evaluation step over the input minibatch and aggregates the eval
     losses over the "global" batch size. A side effect of this method is
     updating the state of evaluation metrics in self.eval_metrics_dict.

    Args:
      - inputs_dict: dictionary of strings->tensors representing an input
         minibatch.
      - global_batch_size: integer representing the "global" minibatch size,
         which is equal to one minibatch_size * num_gpus.

    Returns: a 3-tuple:
      - loss_dict: dictionary of all train losses aggregated according to the
         global batch size.
      - outputs_dict: dict string->tf.Tensor, all intermediate and final outputs.
      - summaries: A 3-tuple representing scalara, image and text summaries.
         returend by _add_summaries(). See _add_summaries() for more details.
    """
    (encoder_inputs, conditional_inputs, real, target_segmap,
     basename, precomputed_latents) = self.parse_inputs(
        inputs_dict, model_utils.Mode.EVAL)
    outputs_dict = self.process(
        encoder_inputs, conditional_inputs, training=False,
        precomputed_latents=precomputed_latents)
    fake = outputs_dict['output']

    # Compute losses.
    fg_mask = utils.get_foreground_mask(
        conditional_inputs, self.config.synthesize_background)
    per_example_loss_dict = self.compute_losses(
        real,
        target_segmap,
        outputs_dict,
        training=False,
        fg_mask=fg_mask,
        conditional_inputs=conditional_inputs)

    # Aggregate losses over minibatch.
    loss_dict = {}
    for loss_key, loss_val in per_example_loss_dict.items():
      loss_dict[loss_key] = tf.nn.compute_average_loss(
          per_example_loss=loss_val, global_batch_size=global_batch_size)

    # Add tensorboard summaries.
    summaries = self._add_summaries(
        encoder_inputs=encoder_inputs,
        target_landmarks=conditional_inputs,
        target_segmap=target_segmap,
        real=real,
        outputs_dict=outputs_dict,
        person_id=inputs_dict[voxceleb.PERSON_ID_KEY],
        video_id=inputs_dict[voxceleb.VIDEO_PART_ID_KEY],
        video_part_id=inputs_dict[voxceleb.VIDEO_PART_ID_KEY],
        fg_mask=fg_mask,
        input_basename=basename)

    if real.shape[0] > 0:
      real_denormalized = utils.denormalize_image(real)
      fake_denormalized = utils.denormalize_image(fake)
      self.eval_metrics_dict['mean_absolute_error'].update_state(
          real_denormalized * fg_mask, fake_denormalized * fg_mask)
      self.eval_metrics_dict['mean_psnr'].update_state(
          tf.image.psnr(
              real_denormalized * fg_mask, fake_denormalized * fg_mask, 255))
      self.eval_metrics_dict['mean_ssim'].update_state(
          tf.image.ssim(
              real_denormalized * fg_mask, fake_denormalized * fg_mask, 255))

      for loss_key, loss_val in per_example_loss_dict.items():
        self.eval_metrics_dict['losses/%s' % loss_key].update_state(
            tf.reduce_mean(loss_val, axis=None))

    return loss_dict, outputs_dict, summaries

  def get_networks(self) -> Dict[Text, Union[
      tf.keras.Model, tf.keras.layers.Layer]]:
    """Returns a dictionary with all networks and submodules of the model."""
    networks_dict = {
        'layout_generator': self._layout_generator,
        'generator': self._generator,
        'encoder': self._encoder
    }
    if self.config.separate_layout_encoder:
      networks_dict['layout_encoder'] = self._layout_encoder
    if self._discriminator is not None:
      networks_dict['discriminator'] = self._discriminator
    return networks_dict

  def get_optimizers(self) -> Dict[Text, tf.keras.optimizers.Optimizer]:
    """Returns a dictionary with all the otpimizers of the model training."""
    optimizers_dict = {'g_optimizer': self._g_optimizer}
    if self._discriminator is not None:
      optimizers_dict['d_optimizer'] = self._d_optimizer
    if self._g_layout_optimizer is not None:
      optimizers_dict['g_layout_optimizer'] = self._g_layout_optimizer
    return optimizers_dict
