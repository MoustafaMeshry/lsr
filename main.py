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


from absl import app
from data import voxceleb_data_provider as voxceleb
from data import voxceleb_data_provider_raw as voxceleb_raw
from data import voxceleb_data_provider_std
from models import layout_pretraining_model
from models import two_step_synthesis_model
from models import model_utils
from options import FLAGS as opts
from options import list_configs
from PIL import Image
import imageio
import glob
import numpy as np
import os
import os.path as osp
import tensorflow as tf
import time
import utils


# Note: _create_model should be called within a strategy.scop()
def _create_model(model_type, config):
  """Creates a model instance."""
  if model_type == 'two_step_syn':
    model = two_step_synthesis_model.TwoStepSynthesisModel(config=config)
  elif model_type == 'pretrain_layout':
    model = layout_pretraining_model.LayoutPretrainingModel(config=config)
  else:
    raise ValueError('Unsupported --model_type value of %s.' % opts.model_type)
  return model


def _load_eval_model(model_dir, strategy):
  """Loads inference weights from a checkpoint."""
  # Construct graph and ops for training and evaluation.
  with strategy.scope():
    model = _create_model(opts.model_type, opts)
    model.init_extra_train_and_eval_networks()

  #  Create checkpoing manager.
  checkpoint = tf.train.Checkpoint(
      **model.get_networks())

  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=model_dir,
      max_to_keep=opts.max_checkpoints_to_keep)

  #  Restore pre-trained models or old checkpoints, if any.
  if not checkpoint_manager.latest_checkpoint:
    print(f'Error! No checkpoint found at {model_dir}.')
    return None
  checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()

  return model


def _train_step(model,
                strategy,
                inputs_dict,
                global_batch_size,
                first_step: bool,
                train_g_step: bool,
                save_summaries: bool,
                global_step: tf.Variable,
                log_info: bool,
                log_str: str='',
                timer: tf.estimator.SecondOrStepTimer=None):
  """
  Runs one training iteration of the model.
  """
  # First call to @tf.functions needs to construct the full graph, hence
  #  needs to run both G and D train steps.
  if not opts.alternate_G_D_training or first_step_flag:
    loss_dict, outputs_dict, summaries = model.train_distributed(
        strategy,
        inputs_dict,
        global_batch_size,
        train_g_step=train_g_step,
        train_d_step=True)
  else:
    if train_g_step:
      loss_dict, outputs_dict, summaries = model.train_distributed(
          strategy,
          inputs_dict,
          global_batch_size,
          train_g_step=True,
          train_d_step=False)
    loss_dict, outputs_dict, summaries = model.train_distributed(
        strategy,
        inputs_dict,
        global_batch_size,
        train_g_step=False,
        train_d_step=True)

  # Write summaries.
  if save_summaries:
    tf.summary.experimental.set_step(step=global_step.numpy())
    # tf.summary.scalar('iterations', global_step.numpy())
    for optimizer_name, optimizer in model.get_optimizers().items():
      tf.summary.scalar(f'learning_rate/{optimizer_name}',
                        optimizer.learning_rate.numpy())

    # Write loss summaries.
    for key, value in loss_dict.items():
      tf.summary.scalar('losses/' + key, value)

    # Write other scalar, image and text summaries.
    scalar_summaries, image_summaries, text_summaries = summaries
    # Scalar summaries.
    for key, value in scalar_summaries.items():
      if strategy.num_replicas_in_sync > 1:
        value = value.values[0]
      tf.summary.scalar(key, value)
    # Image summaries.
    for key, value in image_summaries.items():
      if strategy.num_replicas_in_sync > 1:
        value = value.values[0]
      tf.summary.image(key, (value + 1.) / 2.)
    # Text summaries.
    for key, value in text_summaries.items():
      if strategy.num_replicas_in_sync > 1:
        value = value.values[0]
      tf.summary.text(key, value)

  # Log losses.
  if log_info:
    for i, (key, value) in enumerate(loss_dict.items()):
      log_str += (', ' if i > 0 else '') + f'{key}={value:.3f}'
    print(log_str)

  # Log steps/sec.
  if timer is not None and timer.should_trigger_for_step(global_step.numpy()):
    elapsed_time, elapsed_steps = timer.update_last_triggered_step(
        global_step.numpy())
    if elapsed_time is not None:
      steps_per_second = elapsed_steps / elapsed_time
      tf.summary.scalar(
          'steps/sec', steps_per_second, step=global_step)
  return loss_dict, outputs_dict, summaries


def _evaluate_and_save_output(model,
                              dist_eval_dataset,
                              global_batch_size,
                              strategy,
                              output_dir):
  """
  Evaluates a `model` using a given dataset and saves the output to `output_dir`
  """
  os.makedirs(output_dir, exist_ok=True)
  # Eval loop.
  ex_idx = 0
  for batch_idx, eval_input_dict in enumerate(dist_eval_dataset):
    if batch_idx % 10 == 0:
      print('Evaluating minibatch #%04d.' % batch_idx)
    eval_loss_dict, _, eval_summaries = model.evaluate_distributed(
        strategy,
        eval_input_dict,
        global_batch_size)
    _, image_summaries, text_summaries = eval_summaries
    basename_tensors = text_summaries['basename']
    if strategy.num_replicas_in_sync > 1:
      basename_tensors = basename_tensors.values
    else:
      basename_tensors = [basename_tensors]
    for key, value in image_summaries.items():
      if not 'io_tuple' in key:  # Hack to save io_tuples only!
        continue
      if strategy.num_replicas_in_sync > 1:
        value = value.values
      else:
        value = [value]
      for replica_minibatch_basenames, replica_minibatch_values in zip(
            basename_tensors,  value):
        for basename, io_tuple_out in zip(
              replica_minibatch_basenames, replica_minibatch_values):
          out_file_path = os.path.join(
              output_dir, '%s_%05d_%s.png' % (
                  basename.numpy().decode('utf-8'), ex_idx, key))
          ex_idx += 1
          with tf.compat.v1.gfile.Open(out_file_path, 'wb') as f:
            f.write(utils.to_png_numpy(np.squeeze(io_tuple_out)))

  # Log eval metrics summary.
  eval_summary_str = ''
  eval_summary_str += ('Evaluation metrics summary:\n'
                       '---------------------------\n')
  for key, value in model.eval_metrics_dict.items():
    eval_summary_str += 'eval_metrics/%s = %.4f\n' % (key, value.result())
  return eval_summary_str


def evaluate(model_dir,
             batch_size_per_gpu,
             dataset_name,
             dataset_parent_dir,
             strategy=None):
  """
  Given the train directory `model_dir`, loads the trained model, prepares the
   eval/test dataset and evaluates the loaded model.
  """

  if strategy is None:
    strategy = tf.distribute.MirroredStrategy()
  print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
  val_set_dir = os.path.join(model_dir, 'val_set-K=%d' % opts.K)

  global_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync

  # Prepare input data
  person_id = '*'
  video_id = '*'
  video_part_id = '*'
  # NOTE: The `train` portion of the hold-out validation set is used for both
  #  the few-shot inputs as well as for subject fine-tuning (if any). Note that
  #  both the train and test portions used below were not part of the original
  #  training dataset and all identities used here were not seen during the
  #  model training.
  train_data_source_pattern = osp.join(dataset_parent_dir, 'train',
                                       person_id, video_id, video_part_id)
  test_data_source_pattern = osp.join(dataset_parent_dir, 'test',
                                      person_id, video_id, video_part_id)
  eval_dataset = voxceleb_data_provider_std.provide_data(
      train_data_source_pattern=train_data_source_pattern,
      test_data_source_pattern=test_data_source_pattern,
      k_frames=opts.K,
      max_frames=-1,
      use_segmaps=opts.use_segmaps,
      batch_size=global_batch_size,
      is_training=False,
      shuffle=False)
  dist_eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)

  model = _load_eval_model(model_dir, strategy)
  if model is None:
    print(f'Error! Failed to load model from {model_dir}.')
    return
  eval_metrics_summary = _evaluate_and_save_output(
      model, dist_eval_dataset, global_batch_size, strategy, val_set_dir)
  print(eval_metrics_summary)


def map_person_few_shot_dataset_to_latents(person_few_shot_dataset,
                                           target_dataset,
                                           model,
                                           precomputed_latents=None):
  """TBD."""
  if precomputed_latents is not None:
    z_style, z_layout = precomputed_latents
  else:
    first_batch_dict = next(iter(person_few_shot_dataset))
    (encoder_inputs, conditional_inputs, real, target_segmap,
     basename, _) = model.parse_inputs(
       first_batch_dict, model_utils.Mode.EVAL)
    z_style, z_layout = model.compute_latents(
        encoder_inputs=encoder_inputs,
        num_few_shots=model.config.K,
        training=False)
    precomputed_latents = (z_style, z_layout)

  def _few_shot_to_latent_mapper(ex_dict):
    ret_dict = {}
    few_shot_keys = [
        voxceleb.SEGMAPS_KEY, voxceleb.CONTOURS_KEY, voxceleb.FRAMES_KEY]
    ret_dict['z_style'] = z_style
    ret_dict['z_layout'] = z_layout
    for k, v in ex_dict.items():
      if k in few_shot_keys:
        ret_dict[k] = []
      else:
        ret_dict[k] = v
    return ret_dict

  dataset = target_dataset.map(
      _few_shot_to_latent_mapper,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  ds_options = tf.data.Options()
  ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  dataset = dataset.with_options(ds_options)

  return dataset, precomputed_latents


def infer_single_example(parent_train_dir,
                         config,
                         source_dir,
                         target_dir,
                         batch_size_per_gpu,
                         finetune=True,
                         max_finetune_steps=-1,
                         log_info_every_n_steps=10,
                         save_summaries_every_n_steps=np.inf,
                         evaluate_metrics=False,
                         save_subject_checkpoint=False,
                         decay_lr=True,
                         output_dir=None,
                         num_id_toks=3,
                         strategy=None):
  """TBD."""
  # Initialize tf.distribute.Strategy and other training variables.
  if strategy is None:
    strategy = tf.distribute.MirroredStrategy()
  print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
  global_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync
  enable_summaries = (save_summaries_every_n_steps > 0 and
                      save_summaries_every_n_steps < np.inf)
  src_id = '_'.join(source_dir.split('/')[-num_id_toks:])
  target_id = '_'.join(target_dir.split('/')[-num_id_toks:])
  suffix = f'K{config.K:02}-' + ('ft' if finetune else 'meta')
  if output_dir is None:
    output_dir = osp.join(parent_train_dir, 'inference')
  output_dir = osp.join(output_dir, f'{src_id}--{target_id}--{suffix}')
  os.makedirs(output_dir, exist_ok=True)

  with strategy.scope():
    model = _create_model(config.model_type, config)
    model.init_extra_train_and_eval_networks()
    global_step = tf.compat.v1.train.get_or_create_global_step()
  # Load pre-trained weights, and create checkpoint manager.
  checkpoint = tf.train.Checkpoint(
      global_step=global_step,
      train_examples_count=tf.Variable(0, dtype=tf.int64, trainable=False),
      **model.get_networks())

  warmstart_ckpt_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=parent_train_dir,
      max_to_keep=config.max_checkpoints_to_keep)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=output_dir,
      max_to_keep=1)
  if ckpt_manager.latest_checkpoint:
    print('Waring: Restoring person fine-tuned model weights from {}'.format(
        ckpt_manager.latest_checkpoint))
    status = checkpoint.restore(warmstart_ckpt_manager.latest_checkpoint)
    status.expect_partial()
  else:
    assert warmstart_ckpt_manager.latest_checkpoint
    print('Restoring meta-learned model weights from {}'.format(
        warmstart_ckpt_manager.latest_checkpoint))
    status = checkpoint.restore(warmstart_ckpt_manager.latest_checkpoint)
    status.expect_partial()

  few_shot_dataset = voxceleb_raw.provide_data(
      data_source_pattern=source_dir,
      use_segmaps=config.use_segmaps,
      batch_size=1,
      k_frames=config.K,
      max_sequence_frames=config.K,  # Fine-tune using a deterministic set of K frames out of the 32 available frames.
      num_concatenations=-1,  # Not needed argument, should remove that.
      is_training=False,
      max_examples=1,
      shuffle=False)

  train_dataset = voxceleb_data_provider_std.provide_data(
      train_data_source_pattern=source_dir,
      test_data_source_pattern=source_dir,
      k_frames=1,  # Hack to reduce unnecessary I/O.
      max_frames=-1,
      use_segmaps=config.use_segmaps,
      batch_size=1,
      is_training=False,
      shuffle=False,
      same_video_eval=False)
  train_dataset, latents_tuple = map_person_few_shot_dataset_to_latents(
      few_shot_dataset, train_dataset, model)
  train_dataset = train_dataset.repeat(-1).shuffle(64)

  eval_dataset = voxceleb_data_provider_std.provide_data(
      train_data_source_pattern=target_dir,
      test_data_source_pattern=target_dir,
      k_frames=1,  # Hack to reduce unnecessary I/O.
      max_frames=-1,
      use_segmaps=config.use_segmaps,
      batch_size=1,
      is_training=False,
      shuffle=False,
      same_video_eval=False)
  eval_dataset, _ = map_person_few_shot_dataset_to_latents(
      few_shot_dataset, eval_dataset, model, precomputed_latents=latents_tuple)
  ds_options = tf.data.Options()
  ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  few_shot_dataset = few_shot_dataset.with_options(ds_options)
  train_dataset = train_dataset.with_options(ds_options)
  eval_dataset = eval_dataset.with_options(ds_options)

  # Distribute datasets for a multi-gpu setting, if any.
  dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  dist_eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)

  if finetune:
    if max_finetune_steps is None:
      if config.K < 4:
        max_finetune_steps = 24
      elif config.K < 8:
        max_finetune_steps = 60
      elif config.K < 16:
        max_finetune_steps = 100
      elif config.K < 32:
        max_finetune_steps = 200
      else:
        max_finetune_steps = 400
    # Set up a summary writer.
    if enable_summaries:
      finetune_summary_writer = tf.summary.create_file_writer(output_dir)
      finetune_summary_writer.set_as_default()
    # Load train step and example counts from the pre-trained model.
    prev_global_step = utils.load_variable_from_checkpoint(
        parent_train_dir, 'global_step')
    assert prev_global_step is not None
    # Configure learning rate schedule.
    lr_warmstart_steps = 0
    if decay_lr:
      decay_start_step = prev_global_step + max_finetune_steps // 2
    else:
      decay_start_step = prev_global_step + max_finetune_steps
    decay_end_step = prev_global_step + max_finetune_steps
    decay_num_intervals = config.decay_num_intervals
    if config.scale_lr_with_num_gpus:
      lr_mul_factor = strategy.num_replicas_in_sync
    else:
      lr_mul_factor = 1
    # Build model.
    with strategy.scope():
      optimizers = model.create_optimizers(
          lr_warmstart_steps,
          decay_start_step,
          decay_end_step,
          decay_num_intervals,
          starting_step=prev_global_step,
          lr_mul_factor=lr_mul_factor)
      timer = tf.estimator.SecondOrStepTimer(every_steps=100)
      timer.update_last_triggered_step(global_step.numpy())

    # Load pre-trained weights for auxiliary netwrosk (e.g. VGGFace).
    model.load_loss_pretrained_weights()

    # Fine-tune loop.
    first_step_flag = True
    for batch_idx, train_input_dict in enumerate(dist_train_dataset):
      if batch_idx >= max_finetune_steps:
        break
      # print(f'DBG: batch_idx={batch_idx}, global_step={global_step.numpy()}')
      log_info = (batch_idx % log_info_every_n_steps) == 0
      log_str = ''
      if log_info:
        log_str += (
            f'[Fine-tune step={batch_idx + 1}; '
            f'global_step={global_step.numpy()}: ')
      if first_step_flag:
        train_g_step = True
      else:
        train_g_step = (batch_idx % config.disc_steps_per_g) == 0
      save_summaries = (batch_idx % save_summaries_every_n_steps) == 0
      _train_step(model=model,
                  strategy=strategy,
                  inputs_dict=train_input_dict,
                  global_batch_size=global_batch_size,
                  first_step=first_step_flag,
                  train_g_step=train_g_step,
                  save_summaries=(enable_summaries and save_summaries),
                  global_step=global_step,
                  log_info=log_info,
                  log_str=log_str,
                  timer=timer)
      if first_step_flag:
        first_step_flag = False

      # Increment global step and train examples count.
      checkpoint.train_examples_count.assign_add(global_batch_size)
      tf.compat.v1.assign_add(global_step, 1)

    # Save subject checkpoint.
    if save_subject_checkpoint:
      # Save checkpoint.
      tf.compat.v1.logging.info('Saving checkpoint at step %d to %s.' % (
          global_step.numpy(), output_dir))
      ckpt_manager.save(checkpoint_number=global_step.numpy())
    if enable_summaries:
      finetune_summary_writer.close()

  _evaluate_and_save_output(
      model, dist_eval_dataset, global_batch_size, strategy, output_dir)
  return


def train(
    train_dir,
    batch_size_per_gpu,
    num_train_epochs,
    dataset_name,
    dataset_parent_dir,
    strategy=None):
  """TBD."""

  # ----------------------------------------------------------------------------
  # Print train configuration.
  # ----------------------------------------------------------------------------

  os.makedirs(train_dir, exist_ok=True)
  configs = list_configs()
  print(configs)
  with open(os.path.join(train_dir, 'run_config.txt'), 'a') as f:
    f.write(configs)
    f.write('# =============================================================\n')

  # ----------------------------------------------------------------------------
  # Initialize tf.distribute.Strategy and other training variables.
  # ----------------------------------------------------------------------------

  if strategy is None:
    strategy = tf.distribute.MirroredStrategy()
  print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  global_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync

  # ----------------------------------------------------------------------------
  # Prepare input data
  # ----------------------------------------------------------------------------

  train_dataset = voxceleb.provide_data(
      data_source_pattern=osp.join(dataset_parent_dir, opts.trainset_pattern),
      use_segmaps = opts.use_segmaps,
      batch_size=global_batch_size,
      k_frames=opts.K,
      num_concatenations=opts.num_frame_concatenations,
      is_training=True,
      shuffle=True)
  dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)

  train_summary_writer = tf.summary.create_file_writer(train_dir)
  train_summary_writer.set_as_default()

  # ----------------------------------------------------------------------------
  # Construct graph and ops for training and evaluation.
  # ----------------------------------------------------------------------------

  lr_warmstart_steps = opts.lr_warmstart_steps
  # decay_start_step = (opts.num_train_epochs - opts.num_lr_decay_epochs) * (
  #     opts.trainset_size // global_batch_size)
  # decay_end_step = (
  #     opts.num_train_epochs * opts.trainset_size // global_batch_size)
  prev_global_step = utils.load_variable_from_checkpoint(
      train_dir, 'global_step')
  prev_train_examples_count = utils.load_variable_from_checkpoint(
      train_dir, 'train_examples_count')
  if prev_global_step is None:
    prev_global_step = 0
  if prev_train_examples_count is None:
    prev_train_examples_count = 0
  total_train_examples = opts.total_k_examples * 1000
  num_lr_decay_examples = opts.num_lr_decay_k_examples * 1000
  decay_start_example = total_train_examples - num_lr_decay_examples
  remaining_steps_to_decay = (
      (decay_start_example - prev_train_examples_count) // global_batch_size)
  decay_start_step = prev_global_step + remaining_steps_to_decay
  remaining_steps_to_terminate = (
      (total_train_examples - prev_train_examples_count) // global_batch_size)
  decay_end_step = prev_global_step + remaining_steps_to_terminate
  decay_num_intervals = opts.decay_num_intervals

  if opts.scale_lr_with_num_gpus:
    lr_mul_factor = strategy.num_replicas_in_sync
  else:
    lr_mul_factor = 1

  with strategy.scope():
    model = _create_model(opts.model_type, opts)
    model.init_extra_train_and_eval_networks()
    optimizers = model.create_optimizers(
        lr_warmstart_steps,
        decay_start_step,
        decay_end_step,
        decay_num_intervals,
        lr_mul_factor)
    if opts.model_type == 'pretrain_layout':
      g_optimizer = optimizers['g_layout_optimizer']
    else:
      g_optimizer = optimizers['g_optimizer']
    if 'd_optimizer' in optimizers:
      d_optimizer = optimizers['d_optimizer']
    else:
      d_optimizer = None
    tf.compat.v1.logging.info('Creating Timer ...')
    global_step = tf.compat.v1.train.get_or_create_global_step()
    timer = tf.estimator.SecondOrStepTimer(every_steps=100)
    timer.update_last_triggered_step(global_step.numpy())

  # ----------------------------------------------------------------------------
  #  Create checkpoing manager.
  # ----------------------------------------------------------------------------

  tf.compat.v1.logging.info('Creating checkpoint ...')
  epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)
  train_examples_count_var = tf.Variable(0, dtype=tf.int64, trainable=False)
  checkpoint = tf.train.Checkpoint(
      **model.get_optimizers(),
      global_step=global_step,
      epoch=epoch_var,
      train_examples_count=train_examples_count_var,
      training_finished=tf.Variable(False, dtype=tf.bool, trainable=False),
      **model.get_networks())

  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=train_dir,
      max_to_keep=opts.max_checkpoints_to_keep,
      keep_checkpoint_every_n_hours=opts.keep_checkpoint_every_n_hours)

  # ----------------------------------------------------------------------------
  #  Restore pre-trained models or old checkpoints, if any.
  # ----------------------------------------------------------------------------

  if checkpoint_manager.latest_checkpoint:
    print('Restoring model weights from {}'.format(
        checkpoint_manager.latest_checkpoint))
    # checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
    # status.assert_consumed()
  elif opts.warmup_checkpoint:
    warmup_ckpt = tf.train.Checkpoint(
        # **model.get_optimizers(),
        global_step=global_step,
        epoch=epoch_var,
        train_examples_count=train_examples_count_var,
        # training_finished=tf.Variable(False, dtype=tf.bool, trainable=False),
        **model.get_networks())
    warmup_ckpt_manager = tf.train.CheckpointManager(
        warmup_ckpt,
        directory=opts.warmup_checkpoint,
      max_to_keep=1)
    assert warmup_ckpt_manager.latest_checkpoint, (
        f'No latest ckpt for  --warmup_checkpoint={opts.warmup_checkpoint}')
    print('*** Warmstarting model weights from {}'.format(
        warmup_ckpt_manager.latest_checkpoint))
    status = warmup_ckpt.restore(warmup_ckpt_manager.latest_checkpoint)
    status.expect_partial()
  else:
    print('Initializing networks from scratch!')

  # Load pre-trained weights for auxiliary netwrosk (e.g. VGGFace).
  model.load_loss_pretrained_weights()

  if train_examples_count_var.numpy() >= opts.total_k_examples * 1000:
    print('Model has already trained for --total_k_examples=%d.' % (
        opts.total_k_examples))
    return

  # ----------------------------------------------------------------------------
  # Main loop.
  # ----------------------------------------------------------------------------

  st_time = time.time()
  steps_per_second = -1
  tf.summary.experimental.set_step(step=global_step.numpy())
  first_step_flag = True
  # Alternatively, you can use (global_step.numpy() < opts.max_steps).
  # while epoch_var.numpy() < num_train_epochs:
  while train_examples_count_var.numpy() < opts.total_k_examples * 1000:
    for batch_idx, train_input_dict in enumerate(dist_train_dataset):
      log_info = (global_step.numpy() % opts.log_info_every_n_steps) == 0
      log_str = ''
      if log_info:
        d_iter = 'N/A' if d_optimizer is None else str(
            d_optimizer.iterations.numpy())
        log_str += (
            f'[EPOCH {epoch_var.numpy() + 1}; i_batch={batch_idx}; '
            f'global_step={global_step.numpy()} '
            f'num_k_examples={train_examples_count_var.numpy() // 1000} '
            f'(g_iter={g_optimizer.iterations.numpy()}, d_iter={d_iter})]: ')

      if first_step_flag:
        train_g_step = True
      else:
        train_g_step = (batch_idx % opts.disc_steps_per_g) == 0

      # First call to @tf.functions needs to construct the full graph, hence
      #  needs to run both G and D train steps.
      if not opts.alternate_G_D_training or first_step_flag:
        loss_dict, _, summaries = model.train_distributed(
            strategy,
            train_input_dict,
            global_batch_size,
            train_g_step=train_g_step,
            train_d_step=True)
      else:
        if train_g_step:
          loss_dict, _, summaries = model.train_distributed(
              strategy,
              train_input_dict,
              global_batch_size,
              train_g_step=True,
              train_d_step=False)
        loss_dict, _, summaries = model.train_distributed(
            strategy,
            train_input_dict,
            global_batch_size,
            train_g_step=False,
            train_d_step=True)

      if first_step_flag:
        for network_name, network in model.get_networks().items():
          model_utils.print_model_summary(network, network_name, list_vars=True)
        first_step_flag = False

      # Increment train examples count.
      train_examples_count_var.assign_add(global_batch_size)

      # Write summaries.
      if global_step.numpy() % opts.save_summaries_every_n_steps == 0:
        tf.summary.experimental.set_step(step=global_step.numpy())
        # tf.summary.scalar('iterations', global_step.numpy())
        for optimizer_name, optimizer in optimizers.items():
          tf.summary.scalar(f'learning_rate/{optimizer_name}',
                            optimizer.learning_rate.numpy())

        # Write loss summaries.
        for key, value in loss_dict.items():
          tf.summary.scalar('losses/' + key, value)

        # Write other scalar, image and text summaries.
        scalar_summaries, image_summaries, text_summaries = summaries
        # Scalar summaries.
        for key, value in scalar_summaries.items():
          if strategy.num_replicas_in_sync > 1:
            value = value.values[0]
          tf.summary.scalar(key, value)
        # Image summaries.
        for key, value in image_summaries.items():
          if strategy.num_replicas_in_sync > 1:
            value = value.values[0]
          tf.summary.image(key, (value + 1.) / 2.)
        # Text summaries.
        for key, value in text_summaries.items():
          if strategy.num_replicas_in_sync > 1:
            value = value.values[0]
          tf.summary.text(key, value)

      # Log losses.
      if log_info:
        for i, (key, value) in enumerate(loss_dict.items()):
          log_str += (', ' if i > 0 else '') + f'{key}={value:.3f}'
        print(log_str)

      # Log steps/sec.
      if timer.should_trigger_for_step(global_step.numpy()):
        elapsed_time, elapsed_steps = timer.update_last_triggered_step(
            global_step.numpy())
        if elapsed_time is not None:
          steps_per_second = elapsed_steps / elapsed_time
          tf.summary.scalar(
              'steps/sec', steps_per_second, step=global_step)

      # Increment global_step.
      tf.compat.v1.assign_add(global_step, 1)

      # Save checkpoint.
      if (global_step.numpy() - 1) % opts.save_checkpoint_every_n_steps == 0:
        tf.compat.v1.logging.info('Saving checkpoint at step %d to %s.' % (
            global_step.numpy(), train_dir))
        checkpoint_manager.save(
            checkpoint_number=global_step.numpy())

      if (train_examples_count_var.numpy() // opts.trainset_size) > (
            epoch_var.numpy()):
        # Increment epoch.
        epoch_var.assign_add(1)
        break

  # Assign training_finished variable to True after training is finished,
  #  save the last checkpoint and close summary writer.
  checkpoint.training_finished.assign(True)
  checkpoint_manager.save(checkpoint_number=global_step.numpy())
  train_summary_writer.close()

  # Print overall training time.
  total_time = time.time() - st_time
  print('Total runtime for %d K examples (%d epochs) = %s.' % (
      opts.total_k_examples, epoch_var.numpy(), total_time))


def main(argv):
  print('TF.__version__ = {}'.format(tf.__version__))

  if opts.run_mode == 'train':
    num_train_epochs = opts.total_k_examples * 1000 // opts.trainset_size
    if opts.total_k_examples * 1000 % opts.trainset_size:
      num_train_epochs += 1
    train(train_dir=opts.train_dir,
          batch_size_per_gpu=opts.batch_size,
          num_train_epochs=num_train_epochs,  # opts.num_train_epochs,
          dataset_name=opts.dataset_name,
          dataset_parent_dir=opts.dataset_parent_dir)
  elif opts.run_mode == 'eval':
    evaluate(
        model_dir=opts.train_dir,
        batch_size_per_gpu=opts.batch_size,
        dataset_name=opts.dataset_name,
        dataset_parent_dir=opts.std_eval_subset_dir)
  elif opts.run_mode == 'infer':
    infer_single_example(
        parent_train_dir=opts.train_dir,
        config=opts,
        source_dir=opts.source_subject_dir,
        target_dir=opts.driver_subject_dir,
        batch_size_per_gpu=opts.batch_size,
        finetune=opts.few_shot_finetuning,
        max_finetune_steps=opts.max_finetune_steps,  # Set to -1 to auto-choose based on --K.
        log_info_every_n_steps=10,
        save_summaries_every_n_steps=opts.save_summaries_every_n_steps,
        save_subject_checkpoint=opts.save_subject_checkpoint,
        decay_lr=opts.decay_lr,
        output_dir=None,
        strategy=None)
  else:
    raise ValueError('--run_mode=%s not supported!' % opts.run_mode)


if __name__ == '__main__':
  app.run(main)
