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

"""Train and eval options."""

from absl import flags

FLAGS = flags.FLAGS


# ------------------------------------------------------------------------------
# Train and eval flags
# ------------------------------------------------------------------------------

## Experiment path and description.
flags.DEFINE_string(
    'train_dir', './_train/'
    'sanity_test', 'Directory for model training.')
flags.DEFINE_string('experiment_description', '',
                    'Optional description for the model being trained.')
# ------------------------------------------------------------------------------

## Generic
flags.DEFINE_string('run_mode', 'train', "{'train', 'eval', 'infer'}")
flags.DEFINE_integer('batch_size', 4, 'Batch size for training.')
flags.DEFINE_string('warmup_checkpoint', '',
                    'Load pre-trained weights from the specified path!')
# ------------------------------------------------------------------------------

## Dataset
flags.DEFINE_string('dataset_name', 'voxceleb2', 'name ID for a dataset.')
flags.DEFINE_string(
    'dataset_parent_dir',
    './datasets/preprocessed_voxceleb2/tfrecords',
    'Directory containing generated tfrecord dataset.')
flags.DEFINE_string(
    'eval_dataset_parent_dir',
    './_datasets/voxceleb2_processed/tfrecords',
    'Directory containing generated tfrecord dataset.')
flags.DEFINE_string(
    'trainset_pattern', 'vox2_dev*.tfrecords', 'trainset sstable name pattern.')
flags.DEFINE_string(
    'evalset_pattern', 'vox2b_test*.tfrecords', 'trainset sstable name pattern.')
flags.DEFINE_integer('trainset_size', 1090194, 'Size of the training set.')
# ------------------------------------------------------------------------------

## Training configuration.
flags.DEFINE_string('model_type', 'two_step_syn',
                    'Model type: {pretrain_layout, two_step_syn}.')
flags.DEFINE_boolean('multi_task_training', True, 'Whether to reconstruct RGB '
                     'images while training the segmentation network or not.')
flags.DEFINE_integer('train_resolution', 224,
                     'Crop train images to this resolution.')
flags.DEFINE_integer('K', 4, 'K-shot learning.')
flags.DEFINE_integer('num_frame_concatenations', 10, 'Number of concatenated '
                     'frames in each dataset example.')
flags.DEFINE_boolean('use_segmaps', True,
                     'Pprocess and provide segmaps in the input data pipeline.')
flags.DEFINE_boolean('data_augmentation', False, 'Whether to apply data '
                     'augmentation during training or not.')
# flags.DEFINE_integer('num_train_epochs', 50, 'Number of training epochs.')
# flags.DEFINE_integer('num_lr_decay_epochs', 10, 'Number of epochs during which '
#                      'to decay the learning rate. Set to 0 to disable.')
flags.DEFINE_integer('total_k_examples', 16350,
                     'Number of training examples in kilo.')
flags.DEFINE_integer('num_lr_decay_k_examples', 0, 'Number of examples (in '
                     'kilo) during which to decay the learning rate. Set to 0 '
                     'to disable.')
# flags.DEFINE_integer('decay_lrn_step', 50000,
#                      'Start step for learning rate decay.')
flags.DEFINE_integer('decay_num_intervals', 100, 'Number of decay intervals.')
flags.DEFINE_integer('lr_warmstart_steps', 500, 'Number of steps for learning '
                     'rate warmup. Set to 0 to disable.')
flags.DEFINE_boolean('separate_layout_optimizer', True, 'Whether or not to use'
                     ' a separate optimizer for the segmentation loss.')
flags.DEFINE_float('adam_beta1', 0.0, 'beta1 for adam optimizer.')
flags.DEFINE_float('adam_beta2', 0.999, 'beta2 for adam optimizer.')
flags.DEFINE_float('d_lr', 0.001, 'Learning rate for the discriminator.')
flags.DEFINE_float('g_lr', 0.001, 'Learning rate for the generator.')
flags.DEFINE_float('e_lr', 0.001, 'Learning rate for appearance encoder.')
flags.DEFINE_boolean('scale_lr_with_num_gpus', False, 'Whether to multiply the '
                     'learning rate by the number of gpus or not.')
flags.DEFINE_boolean('alternate_G_D_training', False, 'Whether to apply G, D '
                     'train steps simultaneously or subsequently!')
flags.DEFINE_integer('disc_steps_per_g', 1,
                     'Number of discriminator train steps per generator step.')
flags.DEFINE_float('grad_clip_abs_val', 1,
                   'Absolute value for gradient clipping. Set to -1 to disable')
# ------------------------------------------------------------------------------

## Loss weights
flags.DEFINE_string('lpips_net_type', 'vgg_lin', 'Type for *training* mode not '
                    'evaluation; One of {alex, alex_lin, vgg, vgg_lin}.')
flags.DEFINE_float('w_loss_vgg', 0.1, 'VGG loss weight.')  # was set to 10.
flags.DEFINE_float('w_loss_lpips', 0, 'LPIPS loss weight.')
flags.DEFINE_float('w_loss_segmentation', 0., 'Cross-entropy loss weight.')
flags.DEFINE_float('w_loss_z_l2', 1., '')
flags.DEFINE_float('w_loss_z_layout_l2', 1., '')
flags.DEFINE_float('w_loss_z_cyclic', 0,  # 80.
                   'Weight for cyclic reconstruction of the latent z.')
flags.DEFINE_float('w_loss_identity', 1,  # 70., 300.
                   'Weight for VGGFace/FaceNet-based identity loss.')
flags.DEFINE_float('w_loss_vgg_face_recon', 0.002,  # 70., 300.
                   'Weight for VGGFace/FaceNet-based perceptual loss.')
flags.DEFINE_float('w_loss_feat', 0, 'Feature loss weight (from pix2pixHD).')
flags.DEFINE_float('w_loss_l1', 1., 'L1 loss weight.')
flags.DEFINE_float('w_loss_gan', 1., 'Adversarial loss weight.')
flags.DEFINE_float('w_loss_z_gan', 1., 'Z adversarial loss weight.')
flags.DEFINE_float('w_loss_kl', 0, 'KL divergence weight.')  # 0.01
flags.DEFINE_float('r1_gamma', 10.0, 'weight for disc real gradient penalty.')
flags.DEFINE_float('r2_gamma', 10.0, 'weight for disc fake gradient penalty.')
# ------------------------------------------------------------------------------

## Checkpoints and logging configuration
flags.DEFINE_integer('log_info_every_n_steps', 1<<7,
                     'Log training info and losses after each N steps.')
flags.DEFINE_integer('save_summaries_every_n_steps', 1<<8,
                     'Save tensorboard summaries after each N train steps.')
flags.DEFINE_integer('save_checkpoint_every_n_steps', 1<<11,
                     'Save a checkpoint every N train steps.')
flags.DEFINE_integer('keep_checkpoint_every_n_hours', 6,
                     'Periodic interval for keeping checkpoints.')
flags.DEFINE_integer('max_checkpoints_to_keep', 3,
                     'Max number of checkpoints to keep.')
# -----------------------------------------------------------------------------


## Architecture configurations
## Generator:
flags.DEFINE_integer('output_nc', 3,
                     'Number of channels for the generated image.')
flags.DEFINE_string('output_nonlinearity', 'None', 'Activation for the output '
                    'convolution layer.')
flags.DEFINE_integer('num_segmentation_classes', 19,
                     'Number of classes in the semantic segmentation map.')
flags.DEFINE_integer('spade_proj_filters', 64,  # paper default is 128!
                     'Number of projection filters in a SPADE block.')
flags.DEFINE_boolean('synthesize_background', True,
                     'Whether to synthesize the background or mask it out.')
flags.DEFINE_integer('g_nf', 32,
                     'num filters in the first/last layers of U-net.')
flags.DEFINE_integer('segmap_g_nf', 32,
                     'num filters in the first/last layers of the segmap '
                     'generator.')
flags.DEFINE_integer('g_nf_max', 512, 'max number of filters in the generator.')
flags.DEFINE_integer('G_num_downsamples', 5,
                     'Num down/up sampling blocks for the generator network G.')
flags.DEFINE_integer('G_num_upsamples', 5,
                     'Num down/up sampling blocks for the generator network G.')
flags.DEFINE_integer('G_num_bottleneck_blocks', 0,  # 5
                     'Num residual blocks for the generator network G.')
flags.DEFINE_string('G_arch', 'spade_gen',
                    'Architecture type for the generator.')
flags.DEFINE_string('segmap_G_arch', 'unet_resnet',
                    'Architecture type for the generator.')
flags.DEFINE_boolean('concat_contours_to_layout', False,
                     'Concatenate contours to predicted spatial layouts.')
flags.DEFINE_string('generator_norm_type', 'instance_norm',
                    '{instance_norm, layer_norm, batch_norm, sync_batch_norm}.')
flags.DEFINE_string('upsampling_mode', 'bilinear', 'One of {bilinear, nearest}.')
flags.DEFINE_boolean('use_parametric_norm_in_unet_encoder', False, '')
flags.DEFINE_string('conv_type', 'scaled', '{scaled, spectral_norm, '
                    'weight_norm, regular}.')
flags.DEFINE_boolean('use_self_attn', False, 'Use self-attention layers.')
flags.DEFINE_boolean('use_skip_connections', True, 'use skip connections')
## Encoder:
flags.DEFINE_boolean('separate_layout_encoder', True, 'Whether or not to use'
                     ' a separate encoder for the layout generation tower.')
flags.DEFINE_integer('style_nc', 512, 'Size of latent style vector.')
flags.DEFINE_integer('num_latent_embeddings', 1,
                     'Number of latent vectors to encode from an image.')
flags.DEFINE_integer('e_nf', 32,
                     'num filters in the first layers of the style encoder E.')
flags.DEFINE_integer('segmap_e_nf', 32,
                     'num filters in the first layers of the segmap style '
                     'encoder E.')
flags.DEFINE_integer('E_num_downsamples', 6,
                     'num downsample blocks for the style encoder E.')
flags.DEFINE_boolean('use_vae', False, 'Whether to use VAE encoder or not.')
flags.DEFINE_string('encoder_norm_type', 'none', 'Normalization layer type '
                     'for the encoder; {None, `instance_norm`, `pixel_norm`, '
                     '`layer_norm`, `batch_norm`, `sync_batch_norm`, `none`}.')
flags.DEFINE_boolean('encoder_global_avg_pool', False, 'Adds a global average '
                     'pool layer after the encoder downsample layers.')
flags.DEFINE_boolean('add_fc_to_encoder', True, 'Adds a fully connected layer '
                     'at the end of the encoder.')
flags.DEFINE_boolean('normalize_latent', False, 'Normalize style latent to unit'
                     ' norm.')
flags.DEFINE_string('E_arch', 'resnet',
                    'Architecture type for the style encoder E.')
flags.DEFINE_string('segmap_E_arch', 'resnet',
                    'Architecture type for the style encoder E.')
flags.DEFINE_boolean('concat_conditional_inputs_to_encoder', True, '')
flags.DEFINE_string('e_conv_type', 'scaled', 'Convolution type for the '
                    'encoder. One of {scaled, spectral_norm, '
                    'weight_norm, regular}.')
## Discriminator:
flags.DEFINE_integer('d_nf', 32,
                     'num filters in the first layers of the discriminator D.')
flags.DEFINE_integer('num_disc_scales', 1,
                     'num scales for the multiscale discriminators.')
flags.DEFINE_integer('D_num_downsamples', 6,
                     'num downsample blocks for the discriminator D.')
flags.DEFINE_string('discriminator_norm_type', 'none', 'Normalization layer type '
                     'for the disc; {None, `instance_norm`, `pixel_norm`, '
                     '`layer_norm`, `batch_norm`, `sync_batch_norm`, `none`}.')
flags.DEFINE_string('D_arch', 'resnet_stylegan2', 'Architecture type for the '
                    'discriminator D; {`resnet`, `patch_disc`}.')
flags.DEFINE_string('d_conv_type', 'scaled', 'Convolution type for the '
                    'discriminator. One of {scaled, spectral_norm, '
                    'weight_norm, regular}.')
flags.DEFINE_boolean('use_conditional_disc', True, '')
flags.DEFINE_string('d_loss_type', 'logistic_nonsaturating',
                    '{lsgan, hinge_gan, non_saturating}')
# ------------------------------------------------------------------------------

## Inference and subject fine-tuning flags:
flags.DEFINE_string('source_subject_dir',
                    './_datasets/sample_fsth_eval_subset_processed/train/id00017/OLguY5ofUrY/combined',
                    'Directory with source identitiy data.')
flags.DEFINE_string('driver_subject_dir',
                    './_datasets/sample_fsth_eval_subset_processed/test/id00017/OLguY5ofUrY/combined',
                    'Directory with driver identity.')
flags.DEFINE_string('std_eval_subset_dir',
                    './_datasets/sample_fsth_eval_subset_processed',
                    'Directory with the processed standard test subset.')
flags.DEFINE_integer('max_finetune_steps', -1,
                     'Number of fine-tune setps at inference.')
flags.DEFINE_boolean('few_shot_finetuning', False,
                     'Apply subject-specific fine-tuning at inference.')
flags.DEFINE_string('eval_subset_parent_dir',
                    './datasets/eval_subset_reorg',
                    'Parent directory for the per-subject evaluation sets.')
flags.DEFINE_string('finetune_subset', 'train', 'Subset name for fine-tuning.')
flags.DEFINE_string('test_subset', 'test',
                    'Subset name for fine-tuning evaluation.')
flags.DEFINE_string('subject_pattern', '*',
                    'Target subject(s) for per-subject fine-tuning and '
                    'evaluation. Set to `*` for all subjects.')
flags.DEFINE_string('video_pattern', '*', 'Target video for per-subject '
                    'fine-tuning and evalation.')
flags.DEFINE_boolean('finetune_d', True, 'Finetune discriminator.')
flags.DEFINE_boolean('finetune_g', True, 'Finetune image generator.')
flags.DEFINE_boolean('finetune_e', False, 'Finetune style encoder.')
flags.DEFINE_boolean('finetune_g_layout', True, 'Finetune layout generator.')
flags.DEFINE_boolean('finetune_e_layout', False, 'Finetune layout encoder.')
flags.DEFINE_boolean('finetune_latent', False, 'Finetune input latents.')
flags.DEFINE_boolean('decay_lr', True, 'Wether or not to decay learning '
                     'rates during subject fine-tuning.')
flags.DEFINE_integer('evaluate_every_n_steps', -1, 'Periodic evaluation every '
                     'n steps during subject fune-tuning (set to -1 to '
                     'disable).')
flags.DEFINE_boolean('evaluate_dense', False, 'Wether or not to evaluate the '
                     'dense subset after subject fine-tuning.')
flags.DEFINE_boolean('save_subject_checkpoint', False, 'Wether or not to save '
                     'the fine-tuned checkpoint for target subjects.')
flags.DEFINE_string('target_subject_id', None,
                    'Target subject ID for cross-subject reenactment.')
flags.DEFINE_string('target_video_id', None,
                    'Target video ID for cross-subject reenactment.')
# ------------------------------------------------------------------------------

## Other flags:
flags.DEFINE_boolean('faithful_spade_hacks', False,
                     'Apply hacks to be faithful to SPADE official code;'
                     ' - apply **param-free** instance norm in E, D.'
                     ' - add a redundant SPADEResBlock w/o upsampling when '
                     '    train resolution is <= 256.'
                     ' - if specifying spectral_norm, ONLY use spectral norm'
                     '    for conv2d layers that are followed by a norm layer.'
                     '    Otherwise, (e.g. dense layers, from_rgb, to_rgb, ...)'
                     '    use regular conv.')

# -----------------------------------------------------------------------------
# Some validation and assertions
# -----------------------------------------------------------------------------


def validate_options():
  assert FLAGS.model_type in ['two_step_syn', 'pretrain_layout']


# -----------------------------------------------------------------------------
# Print all options
# -----------------------------------------------------------------------------
def list_configs():
  """Print run configuration."""
  configs = ('# Run flags/options from options.py:\n'
             '# ----------------------------------\n')

  configs += ('## Experiment path and description:\n'
              '## --------------------------------\n')
  configs += 'train_dir="%s"\n' % FLAGS.train_dir
  configs += 'experiment_description="%s"\n' % FLAGS.experiment_description
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Generic:\n'
              '## --------\n')
  configs += 'run_mode="%s"\n' % FLAGS.run_mode
  configs += 'batch_size=%d\n' % FLAGS.batch_size
  configs += 'warmup_checkpoint="%s"\n' % FLAGS.warmup_checkpoint
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Dataset:\n'
              '## --------\n')
  configs += 'dataset_name="%s"\n' % FLAGS.dataset_name
  configs += 'dataset_parent_dir="%s"\n' % FLAGS.dataset_parent_dir
  configs += 'eval_dataset_parent_dir="%s"\n' % FLAGS.eval_dataset_parent_dir
  configs += 'trainset_pattern="%s"\n' % FLAGS.trainset_pattern
  configs += 'evalset_pattern="%s"\n' % FLAGS.evalset_pattern
  configs += 'trainset_size=%d\n' % FLAGS.trainset_size
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Training configuration:\n'
              '## -----------------------\n')
  configs += 'model_type="%s"\n' % FLAGS.model_type
  configs += 'multi_task_training=%s\n' % str(FLAGS.multi_task_training).lower()
  configs += 'train_resolution=%d\n' % FLAGS.train_resolution
  configs += 'K=%d\n' % FLAGS.K
  configs += 'num_frame_concatenations=%d\n' % FLAGS.num_frame_concatenations
  configs += 'use_segmaps=%s\n' % str(FLAGS.use_segmaps).lower()
  configs += 'data_augmentation=%s\n' % str(FLAGS.data_augmentation).lower()
  # configs += 'num_train_epochs=%d\n' % FLAGS.num_train_epochs
  # configs += 'num_lr_decay_epochs=%d\n' % FLAGS.num_lr_decay_epochs
  configs += 'total_k_examples=%d\n' % FLAGS.total_k_examples
  configs += 'num_lr_decay_k_examples=%d\n' % FLAGS.num_lr_decay_k_examples
  configs += 'decay_num_intervals=%d\n' % FLAGS.decay_num_intervals
  configs += 'lr_warmstart_steps=%d\n' % FLAGS.lr_warmstart_steps
  configs += 'separate_layout_optimizer=%s\n' % str(
      FLAGS.separate_layout_optimizer).lower()
  configs += 'adam_beta1=%f\n' % FLAGS.adam_beta1
  configs += 'adam_beta2=%f\n' % FLAGS.adam_beta2
  configs += 'd_lr=%f\n' % FLAGS.d_lr
  configs += 'g_lr=%f\n' % FLAGS.g_lr
  configs += 'e_lr=%f\n' % FLAGS.e_lr
  configs += 'scale_lr_with_num_gpus=%s\n' % str(
      FLAGS.scale_lr_with_num_gpus).lower()
  configs += 'alternate_G_D_training=%s\n' % str(
      FLAGS.alternate_G_D_training).lower()
  configs += 'disc_steps_per_g=%d\n' % FLAGS.disc_steps_per_g
  configs += 'grad_clip_abs_val=%f\n' % FLAGS.grad_clip_abs_val
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Loss weights:\n'
              '## -------------\n')
  configs += 'lpips_net_type="%s"\n' % FLAGS.lpips_net_type
  configs += 'w_loss_vgg=%f\n' % FLAGS.w_loss_vgg
  configs += 'w_loss_lpips=%f\n' % FLAGS.w_loss_lpips
  configs += 'w_loss_segmentation=%f\n' % FLAGS.w_loss_segmentation
  configs += 'w_loss_z_l2=%f\n' % FLAGS.w_loss_z_l2
  configs += 'w_loss_z_layout_l2=%f\n' % FLAGS.w_loss_z_layout_l2
  configs += 'w_loss_z_cyclic=%f\n' % FLAGS.w_loss_z_cyclic
  configs += 'w_loss_identity=%f\n' % FLAGS.w_loss_identity
  configs += 'w_loss_vgg_face_recon=%f\n' % FLAGS.w_loss_vgg_face_recon
  configs += 'w_loss_feat=%f\n' % FLAGS.w_loss_feat
  configs += 'w_loss_l1=%f\n' % FLAGS.w_loss_l1
  configs += 'w_loss_gan=%f\n' % FLAGS.w_loss_gan
  configs += 'w_loss_z_gan=%f\n' % FLAGS.w_loss_z_gan
  configs += 'w_loss_kl=%f\n' % FLAGS.w_loss_kl
  configs += 'r1_gamma=%f\n' % FLAGS.r1_gamma
  configs += 'r2_gamma=%f\n' % FLAGS.r2_gamma
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Checkpoints and logging configuration:\n'
              '## --------------------------------------\n')
  configs += 'log_info_every_n_steps=%d\n' % FLAGS.log_info_every_n_steps
  configs += 'save_summaries_every_n_steps=%d\n' % (
      FLAGS.save_summaries_every_n_steps)
  configs += 'save_checkpoint_every_n_steps=%d\n' % (
      FLAGS.save_checkpoint_every_n_steps)
  configs += 'keep_checkpoint_every_n_hours=%d\n' % (
      FLAGS.keep_checkpoint_every_n_hours)
  configs += 'max_checkpoints_to_keep=%d\n' % FLAGS.max_checkpoints_to_keep
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Generator:\n'
              '## ----------\n')
  configs += 'output_nc=%d\n' % FLAGS.output_nc
  configs += 'output_nonlinearity="%s"\n' % str(FLAGS.output_nonlinearity)
  configs += 'num_segmentation_classes=%d\n' % FLAGS.num_segmentation_classes
  configs += 'spade_proj_filters=%d\n' % FLAGS.spade_proj_filters
  configs += 'synthesize_background=%s\n' % str(
      FLAGS.synthesize_background).lower()
  configs += 'g_nf=%d\n' % FLAGS.g_nf
  configs += 'segmap_g_nf=%d\n' % FLAGS.segmap_g_nf
  configs += 'g_nf_max=%d\n' % FLAGS.g_nf_max
  configs += 'G_num_downsamples=%d\n' % FLAGS.G_num_downsamples
  configs += 'G_num_upsamples=%d\n' % FLAGS.G_num_upsamples
  configs += 'G_num_bottleneck_blocks=%d\n' % FLAGS.G_num_bottleneck_blocks
  configs += 'G_arch="%s"\n' % FLAGS.G_arch
  configs += 'segmap_G_arch="%s"\n' % FLAGS.segmap_G_arch
  configs += 'concat_contours_to_layout=%s\n' % str(
      FLAGS.concat_contours_to_layout).lower()
  configs += 'generator_norm_type="%s"\n' % FLAGS.generator_norm_type
  configs += 'upsampling_mode="%s"\n' % FLAGS.upsampling_mode
  configs += 'use_parametric_norm_in_unet_encoder=%s\n' % str(
      FLAGS.use_parametric_norm_in_unet_encoder).lower()
  configs += 'conv_type="%s"\n' % FLAGS.conv_type
  configs += 'use_self_attn=%s\n' % str(FLAGS.use_self_attn).lower()
  configs += 'use_skip_connections=%s\n' % str(
      FLAGS.use_skip_connections).lower()
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Encoder:\n'
              '## --------\n')
  configs += 'separate_layout_encoder=%s\n' % str(
      FLAGS.separate_layout_encoder).lower()
  configs += 'style_nc=%d\n' % FLAGS.style_nc
  configs += 'num_latent_embeddings=%d\n' % FLAGS.num_latent_embeddings
  configs += 'e_nf=%d\n' % FLAGS.e_nf
  configs += 'segmap_e_nf=%d\n' % FLAGS.segmap_e_nf
  configs += 'E_num_downsamples=%d\n' % FLAGS.E_num_downsamples
  configs += 'use_vae=%s\n' % str(FLAGS.use_vae).lower()
  configs += 'encoder_norm_type="%s"\n' % FLAGS.encoder_norm_type
  configs += 'encoder_global_avg_pool=%s\n' % str(
      FLAGS.encoder_global_avg_pool).lower()
  configs += 'add_fc_to_encoder=%s\n' % str(FLAGS.add_fc_to_encoder).lower()
  configs += 'normalize_latent=%s\n' % str(FLAGS.normalize_latent).lower()
  configs += 'E_arch="%s"\n' % FLAGS.E_arch
  configs += 'segmap_E_arch="%s"\n' % FLAGS.segmap_E_arch
  configs += 'concat_conditional_inputs_to_encoder=%s\n' % str(
      FLAGS.concat_conditional_inputs_to_encoder).lower()
  configs += 'e_conv_type="%s"\n' % FLAGS.e_conv_type
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Discriminator:\n'
              '## --------------\n')
  configs += 'd_nf=%d\n' % FLAGS.d_nf
  configs += 'num_disc_scales=%d\n' % FLAGS.num_disc_scales
  configs += 'D_num_downsamples=%d\n' % FLAGS.D_num_downsamples
  configs += 'discriminator_norm_type="%s"\n' % FLAGS.discriminator_norm_type
  configs += 'D_arch="%s"\n' % FLAGS.D_arch
  configs += 'd_conv_type="%s"\n' % FLAGS.d_conv_type
  configs += 'use_conditional_disc=%s\n' % str(
      FLAGS.use_conditional_disc).lower()
  configs += 'd_loss_type="%s"\n' % FLAGS.d_loss_type
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Inference and subject fine-tuning:\n'
              '## ----------------------------------\n')
  configs += 'max_finetune_steps=%d\n' % FLAGS.max_finetune_steps
  configs += 'eval_subset_parent_dir="%s"\n' % FLAGS.eval_subset_parent_dir
  configs += 'finetune_subset="%s"\n' % FLAGS.finetune_subset
  configs += 'test_subset="%s"\n' % FLAGS.test_subset
  configs += 'subject_pattern="%s"\n' % FLAGS.subject_pattern
  configs += 'video_pattern="%s"\n' % FLAGS.video_pattern
  configs += 'finetune_d=%s\n' % str(FLAGS.finetune_d).lower()
  configs += 'finetune_g=%s\n' % str(FLAGS.finetune_g).lower()
  configs += 'finetune_e=%s\n' % str(FLAGS.finetune_e).lower()
  configs += 'finetune_g_layout=%s\n' % str(FLAGS.finetune_g_layout).lower()
  configs += 'finetune_e_layout=%s\n' % str(FLAGS.finetune_e_layout).lower()
  configs += 'finetune_latent=%s\n' % str(FLAGS.finetune_latent).lower()
  configs += 'decay_lr=%s\n' % str(FLAGS.decay_lr).lower()
  configs += 'evaluate_every_n_steps=%d\n' % FLAGS.evaluate_every_n_steps
  configs += 'evaluate_dense=%s\n' % str(FLAGS.evaluate_dense).lower()
  configs += 'save_subject_checkpoint=%s\n' % str(
      FLAGS.save_subject_checkpoint).lower()
  configs += '# ------------------------------------------------------------\n'

  configs += ('## Other flags:\n'
              '## ------------\n')
  configs += 'faithful_spade_hacks=%s\n' % str(
      FLAGS.faithful_spade_hacks).lower()
  configs += '# ------------------------------------------------------------\n'

  return configs

