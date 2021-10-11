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
from models import model_utils
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union
import layers
import tensorflow as tf


GENERATOR_NAME_SCOPE = 'image_gen'
LAYOUT_GENERATOR_NAME_SCOPE = 'layout_gen'
DISCRIMINATOR_NAME_SCOPE = 'discriminator'
LAYOUT_ENCODER_NAME_SCOPE = 'layout_enc'
ENCODER_NAME_SCOPE = 'image_enc'
BACKGROUND_IDX = 0


class BaseHeadSynthesisModel(object):

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
    self.config = config
    self.concat_landmarks_to_encoder_input = \
        config.concat_conditional_inputs_to_encoder
    self.order = order
    if self.order != layers.NHWC:
      raise NotImplementedError('NCHW format not yet supported!')
    self.train_and_eval_networks_initialized = False
    self.optimizers_initialized = False

  def init_extra_train_and_eval_networks(self):
    """Initializes train losses, networks and optimizers.
    
    This function should be called within `strategy.scope()` if using
      tf.distribute.Strategy.
    """
    pass

  def load_loss_pretrained_weights(self):
    """Loads pre-trained weights for networks used for loss computation."""
    if self._vgg_face_loss is not None:
      resolution = int(self.config.train_resolution)
      input_shape = (None, resolution , resolution, 3)
      self._vgg_face_loss.load_pretrained_weights(input_shape=input_shape)
    return

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
      tf.distribute.Strategy.

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
    pass

  def parse_inputs(self,
                   inputs_dict: Dict[Text, tf.Tensor],
                   mode: Enum=model_utils.Mode.TRAIN,
                   augmentation: bool=False) -> Tuple[tf.Tensor, ...]:
    """Parses the input dataset into the required few-shot inputs.
    
    Given an input dicionary for a mini-batch, this function constructs the
     inputs to the encoder and the generator, as well as the ground truth output
     for training/evaluation.
    """
    # Parse mode-agnostic inputs.
    person_id = inputs_dict[voxceleb.PERSON_ID_KEY]
    video_id = inputs_dict[voxceleb.VIDEO_ID_KEY]
    video_part_id = inputs_dict[voxceleb.VIDEO_PART_ID_KEY]
    frames_few_shots = inputs_dict[voxceleb.FRAMES_KEY]
    frame_target = inputs_dict[voxceleb.TARGET_FRAME_KEY]
    contours_few_shots = inputs_dict[voxceleb.CONTOURS_KEY]
    contour_target = inputs_dict[voxceleb.TARGET_CONTOUR_KEY]
    segmaps_few_shots = inputs_dict[voxceleb.SEGMAPS_KEY]
    segmap_target = inputs_dict[voxceleb.TARGET_SEGMAP_KEY]
    # Cast segmentation label maps to int32
    segmaps_few_shots = tf.dtypes.cast(segmaps_few_shots, tf.dtypes.int32)
    segmap_target = tf.dtypes.cast(segmap_target, tf.dtypes.int32)

    conditional_inputs = contour_target
    z_style = inputs_dict['z_style'] if 'z_style' in inputs_dict else None
    z_layout = inputs_dict['z_layout'] if 'z_layout' in inputs_dict else None
    if z_style is not None or z_layout is not None:
      precomputed_latents = (z_style, z_layout)
    else:
      precomputed_latents = None
    basename = tf.strings.join((person_id, video_id, video_part_id),
                               separator='-')
    channel_axis = 3 if self.order == layers.NHWC else 1
    if precomputed_latents is None:
      encoder_inputs = frames_few_shots
      if self.concat_landmarks_to_encoder_input:
        encoder_inputs = tf.concat((encoder_inputs, contours_few_shots),
                                   axis=channel_axis + 1)
    else:
      encoder_inputs = None

    # Parse mode-specific inputs.
    if mode == model_utils.Mode.TRAIN or mode == model_utils.Mode.EVAL:
      x_gt = frame_target
      assert not augmentation, 'No augmentation supported yet!'
      return (encoder_inputs, conditional_inputs, x_gt, segmap_target, basename,
              precomputed_latents)
    elif mode == model_utils.Mode.PREDICT:
      return encoder_inputs, conditional_inputs, basename, precomputed_latents
    else:
      raise ValueError('Unsupported mode %s; must be one of '
                       '{TRAIN, EVAL, PREDICT}.' % str(mode))

  def _add_summaries(
      self,
      encoder_inputs: tf.Tensor,
      target_landmarks: tf.Tensor,
      target_segmap: tf.Tensor,
      real: tf.Tensor,
      outputs_dict: Dict[Text, tf.Tensor],
      fg_mask: Union[float, tf.Tensor]=1.,
      person_id: Optional[tf.Tensor]=None,
      video_id: Optional[tf.Tensor]=None,
      video_part_id: Optional[tf.Tensor]=None,
      input_basename: Optional[Union[Text, tf.Tensor]]=None,
      visualize_rgb: Optional[bool]=True) -> Tuple[Dict[Text, tf.Tensor], ...]:
    """Prepares tensorboard summaries for training/evaluation.

    This method takes all inputs, ground truth and intermediate outputs and
     prepares image/scalar/text tensorboard summaries to visualize the training
     or evaluation.

    Args:
      - encoder_inputs: 4D tensor, the input to the encoder network.
      - target_landmarks: 4D tensor, the input the generator network.
      - target_segmap: 4D tensor, the label map of the semantic segmentation (
         shape = [batch_size, H, W, 1]).
      - real: 4D tensor, the ground truth output.
      - outputs_dict: dict string->tf.Tensor, all intermediate and final
         outputs.
      - fg_mask: Optional 4D tensor, a mask image to apply to the final output
         and ground truth. Default is a scalar 1, which leaves the output and
         ground truth unmasked.
      - person_id: Optional text tensor, person_id for each example.
      - video_id: Optional text tensor, video_id for each example.
      - video_part_id: Optional text tensor, video_part_id for each example.
      - input_basename: Optional text, basenames/base paths for each input in
         the minibatch.
      - visualize_rgb: Optional bool, whether or not to visualize RGB output.

    Returns:
      A 3-tuple: (scalar_summaries, image_summaries, text_summaries); each is a
      dictionary of str->tf.Tensor.
    """
    scalar_summaries_dict = {}
    image_summaries_dict = {}
    text_summaries_dict = {}
    # Retrieve outputs.
    fake = outputs_dict['output']
    
    # Tensorboard text summaries.
    if person_id is not None:
      text_summaries_dict['person_id'] = person_id
    if video_id is not None:
      text_summaries_dict['video_id'] = video_id
    if video_part_id is not None:
      text_summaries_dict['video_part_id'] = video_part_id
    if input_basename is not None:
      text_summaries_dict['basename'] = input_basename

    # Visualize few-shot inputs and target rgb frames.
    if encoder_inputs is not None:
      few_shot_inputs = tf.slice(
          encoder_inputs, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 3])
      num_viz_shots = min(
          5, tf.compat.v1.dimension_value(few_shot_inputs.shape[1]))
      few_shot_splits = tf.split(few_shot_inputs, self.config.K, axis=1)
      few_shot_splits = few_shot_splits[:num_viz_shots]
      few_shot_splits = [tf.squeeze(x, axis=1) for x in few_shot_splits]
      input_and_target_frames = few_shot_splits
      input_and_target_frames.append(real)
      few_shot_tuple_viz = tf.concat(input_and_target_frames, axis=2)
      image_summaries_dict['few_shot_inputs_and_target'] = few_shot_tuple_viz

    # Add IO tuple visualization.
    io_tuple_items = []
    io_tuple_items.append(target_landmarks)
    if target_segmap is not None:
      segmap_out_label_map = outputs_dict['segmap_label_map']
      num_seg_classes = self.config.num_segmentation_classes
      segmap_out_vis = model_utils.visualize_label_map(
          segmap_out_label_map, num_seg_classes=num_seg_classes)
      segmap_gt_vis = model_utils.visualize_label_map(
          target_segmap, num_seg_classes=num_seg_classes)
      io_tuple_items.append(segmap_out_vis)
      io_tuple_items.append(segmap_gt_vis)

    if visualize_rgb:
      if not self.config.synthesize_background:
        io_tuple_items.append(tf.clip_by_value(fake, -1, 1))
      io_tuple_items.append(tf.clip_by_value(fake * fg_mask, -1, 1))
      io_tuple_items.append(real * fg_mask)
    # Concatenate along width.
    io_tuple = tf.concat(io_tuple_items, axis=2)
    image_summaries_dict['io_tuple'] = io_tuple

    return scalar_summaries_dict, image_summaries_dict, text_summaries_dict

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
    pass

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
    pass

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
    pass

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
    pass

  @tf.function
  def train_distributed(
      self,
      strategy: tf.distribute.Strategy,
      dist_inputs_dict: Dict[Text, Any],
      global_batch_size: int,
      train_g_step: bool=True,
      train_d_step: bool=True) -> Tuple[
          Dict[Text, tf.Tensor], Dict[Text, Any], Tuple[Dict[Text, Any], ...]]:
    """Runs a distributed train step and aggregates losses across replicas.

    Runs the train step over the global minibatch and aggregates the train
     losses across different replicas.

    Args:
      - strategy: tf.distribute.Strategy to be used for building strategy-aware
         networks.
      - dist_inputs_dict: dictionary of strings->tensors representing an input
         minibatch to be distributed across replicas.
      - global_batch_size: integer representing the "global" minibatch size,
         which is equal to one minibatch_size * num_gpus.
      - train_g_step: boolean, whether to update the generator weights or not.
      - train_d_step: boolean, whether to update the discriminator weights or
         not.

    Returns: a 3-tuple:
      - loss_dict: dictionary of all train losses aggregated properly across
         different replicas (i.e over the global batch size).
      - outputs_dict: dict string->PerReplica object, all intermediate and
        final outputs, but not aggregated (concatenated) across replicas.
      - summaries: A 3-tuple representing scalara, image and text summaries.
         returend by _add_summaries(). See _add_summaries() for more details.
         Summary tensors are PerReplica objects that are not aggregated
         (concatenated) across replicas.
    """
    (per_replica_loss_dict, per_replica_outputs_dict,
     per_replica_summaries) = strategy.run(
         self.train, args=(
             dist_inputs_dict, global_batch_size, train_g_step, train_d_step))
    loss_dict = {}
    for loss_key, loss_val in per_replica_loss_dict.items():
      loss_dict[loss_key] = strategy.reduce(
          tf.distribute.ReduceOp.SUM, loss_val, axis=None)
    return loss_dict, per_replica_outputs_dict, per_replica_summaries

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
    pass

  @tf.function
  def evaluate_distributed(
      self,
      strategy: tf.distribute.Strategy,
      dist_inputs_dict: Dict[Text, Any],
      global_batch_size: int) -> Tuple[
          Dict[Text, tf.Tensor], Dict[Text, Any], Tuple[Dict[Text, Any], ...]]:
    """Runs a distributed evaluation step and aggregates losses across replicas.

    Runs the evaluation step over the global minibatch and aggregates the eval
     losses across different replicas. A side effect of this method is
     updating the state of evaluation metrics in self.eval_metrics_dict.

    Args:
      - strategy: tf.distribute.Strategy to be used for building strategy-aware
         networks.
      - dist_inputs_dict: dictionary of strings->tensors representing an input
         minibatch to be distributed across replicas.
      - global_batch_size: integer representing the "global" minibatch size,
         which is equal to one minibatch_size * num_gpus.

    Returns: a 3-tuple:
      - loss_dict: dictionary of all train losses aggregated properly across
         different replicas (i.e over the global batch size).
      - outputs_dict: dict string->PerReplica object, all intermediate and
        final outputs, but not aggregated (concatenated) across replicas.
      - summaries: A 3-tuple representing scalara, image and text summaries.
         returend by _add_summaries(). See _add_summaries() for more details.
         Summary tensors are PerReplica objects that are not aggregated
         (concatenated) across replicas.
    """
    (per_replica_loss_dict, per_replica_outputs_dict,
     per_replica_summaries) = strategy.run(
         self.evaluate, args=(dist_inputs_dict, global_batch_size))
    loss_dict = {}
    for loss_key, loss_val in per_replica_loss_dict.items():
      loss_dict[loss_key] = strategy.reduce(
          tf.distribute.ReduceOp.SUM, loss_val, axis=None)
    return loss_dict, per_replica_outputs_dict, per_replica_summaries

  def get_networks(self) -> Dict[Text, Union[
      tf.keras.Model, tf.keras.layers.Layer]]:
    """Returns a dictionary with all networks and submodules of the model."""
    pass

  def get_optimizers(self) -> Dict[Text, tf.keras.optimizers.Optimizer]:
    """Returns a dictionary with all the otpimizers of the model training."""
    pass

  def reset_eval_metrics(self):
    """Resets the internal state of all evaluation metrics."""
    for metric in self.eval_metrics_dict.values():
      metric.reset_states()
