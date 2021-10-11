from PIL import Image
from absl import app
from absl import flags
from skimage.metrics import structural_similarity as ssim
import functools
import glob
import losses.losses as losses
from losses.vgg_face_loss import VGGFaceCSIM
import numpy as np
import os
import os.path as osp
import skimage.measure
import tensorflow as tf
import tensorflow_gan as tfgan

FLAGS = flags.FLAGS
flags.DEFINE_string('val_set_out_dir', None,
                    'Output directory with concatenated fake and real images.')
flags.DEFINE_string('output_name_pattern', '*.png',
                    'Name pattern for output images.')
flags.DEFINE_string('experiment_title', 'experiment',
                    'Name for the experiment to evaluate')


def _read_and_maybe_extract(img_path, extract_idx=None, width_to_height_ratio=1,
                            resize_shape=None):
  img = Image.open(img_path)
  if extract_idx is not None:
    img = np.array(img)
    conc_h, conc_w = img.shape[:2]
    w = conc_h * width_to_height_ratio
    assert conc_w % w == 0
    num_concatenations = conc_w // w
    if extract_idx < 0: extract_idx += num_concatenations
    img = img[:, w*extract_idx:w*(extract_idx+1), :]
    if resize_shape is not None:
      img = np.array(Image.fromarray(img).resize(resize_shape))
  elif resize_shape:
    img = img.resize(resize_shape)
    img = np.array(img)
  else:
    img = np.array(img)
  return img


def _extract_real_and_fake_from_concatenated_output(
    val_set_out_dir, output_name_pattern='*.png'):
  """Extracts fake and real images from a concatenated image tuple.
  
  This function assumes a list of images, where each image contains a tuple of
   images concatenated along the width, and that the fake and real images
   occupy the last two columns in the concatenated tuple. It extracts the fake
   and real images and stores them in <val_set_out_dir>/fake and
   <val_set_out_dir>/real respectively.
   
  Args:
    val_set_out_dir: str, path to a directory containing the images to process.
    output_name_pattern: str, name pattern for the target output images that
     contain the concatenated fake and real tuples.
  """
  fakes_dir = osp.join(val_set_out_dir, 'fake')
  reals_dir = osp.join(val_set_out_dir, 'real')
  if not osp.exists(fakes_dir):
    os.makedirs(fakes_dir)
  if not osp.exists(reals_dir):
    os.makedirs(reals_dir)
  imgs_pattern = osp.join(val_set_out_dir, output_name_pattern)
  imgs_paths = sorted(glob.glob(imgs_pattern))
  print('Separating %d images in %s.' % (len(imgs_paths), val_set_out_dir))
  for img_path in imgs_paths:
    basename = osp.basename(img_path)[:-4]  # remove the '.png' suffix
    img = np.array(Image.open(img_path))
    img_res = img.shape[0]
    fake = img[:, -2*img_res:-img_res, :]
    real = img[:, -img_res:, :]
    fake_path = osp.join(fakes_dir, '%s_fake.png' % basename)
    real_path = osp.join(reals_dir, '%s_real.png' % basename)
    Image.fromarray(fake).save(fake_path)
    Image.fromarray(real).save(real_path)


def compute_ssim_metric(image_set1_paths, image_set2_paths, data_range=255,
                        fake_resize_shape=None, set2_extract_idx=None):
  """Computes the mean SSIM metric between two sets of images.
  
  Args:
    image_set1_paths: list of image paths.
    image_set2_paths: list of image paths.
    data_range: max value of image pixels / pixel range (assumes min pixel
      value is 0).
  """
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating SSIM for %d pairs' % len(image_set1_paths))

  total_metric_val = 0.
  for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                  image_set2_paths)):
    img1 = np.array(Image.open(img1_path), dtype=np.float32)
    # img2 = np.array(Image.open(img2_path), dtype=np.float32)
    img2 = _read_and_maybe_extract(
        img2_path, set2_extract_idx, resize_shape=fake_resize_shape).astype(
            np.float32)

    # ssim_val = tf.image.ssim(
    #     tf.convert_to_tensor(img1),
    #     tf.convert_to_tensor(img2),
    #     max_val=data_range).numpy()
    ssim_val = ssim(img1, img2, data_range=data_range, multichannel=True)
    total_metric_val += ssim_val

  return total_metric_val / len(image_set1_paths)


def compute_mean_absolute_error_metric(image_set1_paths,
                                       image_set2_paths,
                                       fake_resize_shape=None,
                                       set2_extract_idx=None):
  """Computes the mean absolute error (i.e L1 loss) between two sets of images.
  
  Args:
    image_set1_paths: list of image paths.
    image_set2_paths: list of image paths.
  """
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating L1 loss for %d pairs' % len(image_set1_paths))

  total_loss = 0.
  for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                  image_set2_paths)):
    img1 = np.array(Image.open(img1_path), dtype=np.float32)
    # img2 = np.array(Image.open(img2_path), dtype=np.float32)
    img2 = _read_and_maybe_extract(
        img2_path, set2_extract_idx, resize_shape=fake_resize_shape).astype(
            np.float32)

    loss_l1 = np.mean(np.abs(img1 - img2))
    total_loss += loss_l1

  return total_loss / len(image_set1_paths)


def compute_psnr_metric(image_set1_paths, image_set2_paths,
                        fake_resize_shape=None, set2_extract_idx=None):
  """Computes the mean PSNR metric between two sets of images.
  
  Args:
    image_set1_paths: list of image paths.
    image_set2_paths: list of image paths.
  """
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating PSNR metric for %d pairs' % len(image_set1_paths))

  total_loss = 0.
  for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                  image_set2_paths)):
    img1 = np.array(Image.open(img1_path))
    # img2 = np.array(Image.open(img2_path))
    img2 = _read_and_maybe_extract(img2_path, set2_extract_idx,
                                   resize_shape=fake_resize_shape)

    loss_psnr = skimage.metrics.peak_signal_noise_ratio(img1, img2)
    total_loss += loss_psnr

  return total_loss / len(image_set1_paths)


def compute_identity_similarity(image_set1_paths, image_set2_paths,
                                fake_resize_shape=None, set2_extract_idx=None):
  """Computes the mean identity similairty between two sets of images.

  Args:
    image_set1_paths: list of image paths.
    image_set2_paths: list of image paths.
  """
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating identity CSIM for %d pairs' % len(image_set1_paths))

  total_csim = 0.
  csim_fn = VGGFaceCSIM()
  sample_img = np.array(Image.open(image_set1_paths[0]))
  input_shape = (None,) + sample_img.shape
  csim_fn.load_pretrained_weights(input_shape=input_shape)
  for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                  image_set2_paths)):
    img1 = np.array(Image.open(img1_path), dtype=np.float32)
    # img2 = np.array(Image.open(img2_path), dtype=np.float32)
    img2 = _read_and_maybe_extract(
        img2_path, set2_extract_idx, resize_shape=fake_resize_shape).astype(
            np.float32)

    img1 = np.expand_dims(img1 * 2. / 255. - 1, axis=0)
    img2 = np.expand_dims(img2 * 2. / 255. - 1, axis=0)
    csim = csim_fn(img1, img2, axis=-1)
    total_csim += csim

  return total_csim / len(image_set1_paths)


def compute_lpips_metric(image_set1_paths, image_set2_paths,
                         fake_resize_shape=None, set2_extract_idx=None):
  """Computes the mean absolute error (i.e L1 loss) between two sets of images.

  Args:
    image_set1_paths: list of image paths.
    image_set2_paths: list of image paths.
  """
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating LPIPS loss for %d pairs' % len(image_set1_paths))

  total_loss = 0.
  lpips_fn = losses.LPIPS()
  for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                  image_set2_paths)):
    img1 = np.array(Image.open(img1_path), dtype=np.float32)
    # img2 = np.array(Image.open(img2_path), dtype=np.float32)
    img2 = _read_and_maybe_extract(
        img2_path, set2_extract_idx, resize_shape=fake_resize_shape).astype(
            np.float32)

    img1 = np.expand_dims(img1 * 2. / 255. - 1, axis=0)
    img2 = np.expand_dims(img2 * 2. / 255. - 1, axis=0)
    lpips_loss = lpips_fn(img1, img2)
    total_loss += lpips_loss

  return total_loss / len(image_set1_paths)


def compute_fid_metric(image_set1_paths, image_set2_paths, batch_size=256,
                       fake_resize_shape=None, set2_extract_idx=None):
  """Computes the frechet inception distance metric between two sets of images.
  
  Args:
    image_set1_paths: list of image paths.
    image_set2_paths: list of image paths.
  """
  def _read_minibatch(batch_set1_paths, batch_set2_paths): 
    imgs1 = np.array(
        [np.array(Image.open(x).resize((299, 299))) for x in batch_set1_paths])
    # imgs2 = np.array(
    #     [np.array(Image.open(x).resize((299, 299))) for x in batch_set2_paths])
    imgs2 = np.array(
        [_read_and_maybe_extract(x, set2_extract_idx, resize_shape=(
            299, 299)) for x in batch_set2_paths])
    # imgs2 = np.array([np.array(Image.fromarray(x).resize(
    #     (299, 299))) for x in imgs2])
    imgs1 = (imgs1 / 255. * 2 - 1).astype(np.float32)  # rescale to [-1, 1].
    imgs2 = (imgs2 / 255. * 2 - 1).astype(np.float32)  # rescale to [-1, 1].
    if set2_extract_idx is not None:
      imgs2 = imgs2[:, :, :, :]
    return imgs1, imgs2

  # # Method #1: for small sets of images that fit in memory. Works for up to
  # #  ~1600 pairs of images.
  # reals, fakes = _read_minibatch(image_set1_paths, image_set2_paths)
  # t_reals = tf.convert_to_tensor(reals)
  # t_fakes = tf.convert_to_tensor(fakes)
  # fid = tfgan.eval.frechet_inception_distance(t_reals, t_fakes, num_batches=1)
  # return fid.numpy()
  
  # Method #2: splits the image lists into minibatches and uses a streaming
  #  version of FID.
  num_images = len(image_set1_paths)
  num_batches = num_images // batch_size + (num_images % batch_size != 0)
  with tf.compat.v1.Graph().as_default() as graph:
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.dtypes.float32, [None, 299, 299, 3])
    y = tf.compat.v1.placeholder(tf.dtypes.float32, [None, 299, 299, 3])
    fid, update_fid_op = tfgan.eval.frechet_inception_distance_streaming(x, y)

    with tf.compat.v1.Session() as sess:
      sess.run(tf.compat.v1.local_variables_initializer())
      for i_batch in range(num_batches):
        st_idx = i_batch * batch_size
        end_idx = min((i_batch + 1) * batch_size, num_images)
        set1_minibatch_paths = image_set1_paths[st_idx : end_idx]
        set2_minibatch_paths = image_set2_paths[st_idx : end_idx]
        minibatch1, minibatch2 = _read_minibatch(
            set1_minibatch_paths, set2_minibatch_paths)
        print(f'FID: processing minibatch #{i_batch + 1}/{num_batches}')
        sess.run(update_fid_op, feed_dict={x: minibatch1, y: minibatch2})
      fid_val = sess.run(fid)
      return fid_val


def _evaluate_experiment_aux(set1,
                             set2,
                             fake_resize_shape=None,
                             metrics=('l1', 'psnr', 'ssim', 'fid'),
                             print_metrics=True,
                             experiment_title='experiment',
                             set2_extract_idx=None):
  metrics_dict = dict()
  for metric in metrics:
    if metric == 'fid':
      metric_val = compute_fid_metric(
          set1, set2, set2_extract_idx=set2_extract_idx)
    elif metric == 'identity_sim':
      raise NotImplementedError('Metric `{metric}` not yet implemented!')
    elif metric == 'l1':
      metric_val = compute_mean_absolute_error_metric(
          set1, set2, fake_resize_shape=fake_resize_shape,
          set2_extract_idx=set2_extract_idx)
    elif metric == 'lpips':
      metric_val = compute_lpips_metric(
          set1, set2, fake_resize_shape=fake_resize_shape,
          set2_extract_idx=set2_extract_idx)
    elif metric == 'psnr':
      metric_val = compute_psnr_metric(
          set1, set2, fake_resize_shape=fake_resize_shape,
          set2_extract_idx=set2_extract_idx)
    elif metric == 'ssim':
      metric_val = compute_ssim_metric(
          set1, set2, fake_resize_shape=fake_resize_shape,
          set2_extract_idx=set2_extract_idx)
    elif metric == 'csim':
      metric_val = compute_identity_similarity(
          set1, set2, fake_resize_shape=fake_resize_shape,
          set2_extract_idx=set2_extract_idx)
    elif metric == 'vgg':
      raise NotImplementedError('Metric `{metric}` not yet implemented!')
    else:
      raise ValueError(f'Unsupported evaluation metric: `{metric}`!')
    metrics_dict[metric] = metric_val
    print('*** %s metric for %s = %f' % (metric, experiment_title, metric_val))
  return metrics_dict


# Evaluates output structured as in voxceleb hierarchy (e.g. output for subject
#  fine-tuning and baselines such as FOMM).
def evaluate_structured_output(output_parent_dir,
                               gt_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/eval_subset_reorg',
                               subset=None,
                               concatenated_output=True,
                               fake_resize_shape=None,
                               metrics=('l1', 'psnr', 'ssim', 'fid'),
                               log_file_name='quantitative_metrics.txt',
                               experiment_title='experiment',
                               description='',
                               fake_img_ext='png',
                               real_img_ext='png',
                               real_img_suffix='rgb'):
  fake_person_dirs = sorted(os.listdir(output_parent_dir))
  fake_person_dirs = [d for d in fake_person_dirs if osp.isdir(
      osp.join(output_parent_dir, d))]
  real_person_dirs = sorted(os.listdir(gt_parent_dir))
  real_person_dirs = [d for d in real_person_dirs if osp.isdir(
      osp.join(gt_parent_dir, d))]
  print('Found %d fake person dirs and %d real person dirs.' % (
      len(fake_person_dirs), len(real_person_dirs)))
  fake_set = []
  real_set = []
  real_p_idx = 0
  for p_idx, person_id in enumerate(fake_person_dirs):
    while real_person_dirs[real_p_idx] != person_id:
      real_p_idx += 1
      print(f'Warning: real_person_dirs[#{real_p_idx}] != {person_id}.')
      if real_p_idx == len(real_person_dirs):
        raise ValueError(
            f"Can't find a ground truth match for person #{p_idx}: {person_id}")
    real_vids = sorted(glob.glob(osp.join(gt_parent_dir, person_id, '*')))
    fake_vids = sorted(glob.glob(osp.join(output_parent_dir, person_id, '*')))
    assert len(real_vids) == len(fake_vids), (
        f'{len(real_vids)} vs {len(fake_vids)}')
    for real_vid, fake_vid in zip(real_vids, fake_vids):
      if real_img_suffix:
        real_image_paths = sorted(
            glob.glob(osp.join(
                real_vid, '*', f'*_{real_img_suffix}.{real_img_ext}')))
      else:
        real_image_paths = sorted(
            glob.glob(osp.join(real_vid, '*', f'*.{real_img_ext}')))
      subset_dir = '*' if subset is None else f'{subset}_output'
      fake_image_paths = sorted(glob.glob(osp.join(
          fake_vid, subset_dir, f'*.{fake_img_ext}')))
      assert len(real_image_paths) == len(fake_image_paths), (
          f'{len(real_image_paths)} vs {len(fake_image_paths)}')
      real_set += real_image_paths
      fake_set += fake_image_paths
    real_p_idx += 1

  set2_extract_idx = -2 if concatenated_output else None
  metrics_dict =  _evaluate_experiment_aux(
      real_set, fake_set, metrics=metrics, experiment_title=experiment_title,
      fake_resize_shape=fake_resize_shape, set2_extract_idx=set2_extract_idx)
  # Print results to file.
  if log_file_name is not None:
    log_str = f'Title: {experiment_title}\n'
    if len(description):
      log_str += '\nDescription: %s\n\n' % description
    for metric, metric_val in metrics_dict.items():
      log_str += '%s: %.4f\n' % (metric, metric_val)
    with open(osp.join(output_parent_dir, log_file_name), 'a') as f:
      f.write(log_str)
  return log_str


def evaluate_experiment(val_set_out_dir,
                        output_name_pattern='*.png',
                        metrics=('l1', 'psnr', 'ssim', 'fid'),
                        log_file_name='quantitative_metrics.txt',
                        experiment_title='experiment',
                        description=''):
  """Evaluates quantitative metrics for the output of a trained model.
  
  Args:
    val_set_out_dir: str, path to a directory containing the images to process.
     Each image is assumed to contain a tuple of images  concatenated along the
     width, and that the fake and real images occupy the last two columns in the
     concatenated tuple.
    output_name_pattern: str, name pattern for the target output images that
     contain the concatenated fake and real tuples.
    metrics: list/tuple of evaluation metrics to compute. Supported metrics
     include {psnr, l1, ssim, fid, lpips, vgg, identity_sim}.
    log_file_name: optional str, if not None, then the evaluated metrics are
     logged to the file path: <val_set_out_dir>/../<log_file_name>.
    experiment_title: optional str, name of the experiment to evaluate.
    description: optional str, descrition of the experiment to evaluate.
  """
  fakes_dir = osp.join(val_set_out_dir, 'fake')
  reals_dir = osp.join(val_set_out_dir, 'real')
  _extract_real_and_fake_from_concatenated_output(
      val_set_out_dir, output_name_pattern=output_name_pattern)
  input_pattern1 = osp.join(reals_dir, '*.png')
  input_pattern2 = osp.join(fakes_dir, '*.png')
  set1 = sorted(glob.glob(input_pattern1))
  set2 = sorted(glob.glob(input_pattern2))
  metrics_dict =  _evaluate_experiment_aux(
      set1, set2, metrics=metrics, experiment_title=experiment_title)
  # Print results to file.
  if log_file_name is not None:
    log_str = f'Title: {experiment_title}\n'
    if len(description):
      log_str += '\nDescription: %s\n\n' % description
    for metric, metric_val in metrics_dict.items():
      log_str += '%s: %.4f\n' % (metric, metric_val)
    with open(osp.join(val_set_out_dir, '..', log_file_name), 'a') as f:
      f.write(log_str)
  return log_str


def compare_csim():
  val_set_dirs = [
      # '/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp15-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2',
      # '/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp16-finetune_with_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2',
      '/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp16_rerun-finetune_with_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2',
      # '/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/talking_heads-exp17-spade-full_losses-d_stylegan2',
      # '/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/talking_heads-exp18-unet-full_losses-d_stylegan2',
  ]
  k_vals = [1,4,8,32]
  all_logs = ''
  for i, val_set_dir in  enumerate(val_set_dirs):
    for k in k_vals:
      out_dir = osp.join(val_set_dir, f'val_set-K={k}')
      if not osp.exists(out_dir):
        continue
      exp_title = val_set_dir.split('/')[-1]
      log_str = evaluate_experiment(
          osp.join(val_set_dir, f'val_set-K={k}'),
          output_name_pattern=FLAGS.output_name_pattern,
          experiment_title=exp_title + ' - K=%d' % k,
          metrics=['csim'])
      all_logs += log_str + '\n'
      print(all_logs)


def compare_csim2():
  # for baseline in ['FOMM', 'x2face']:
  #   evaluate_structured_output(output_parent_dir=f'/fs/vulcan-projects/two_step_synthesis_meshry/results/{baseline}/test',
  #                              gt_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/eval_subset_reorg/test',
  #                              subset=None,
  #                              concatenated_output=False,
  #                              fake_resize_shape=(224, 224),
  #                              metrics=('csim',),
  #                              log_file_name='quantitative_metrics.txt',
  #                              experiment_title=f'{baseline}',
  #                              description=f'{baseline} Baseline.')

  k_vals = [8, 32]  # [1, 8, 32]
  for method in ['ff', 'ft']:
    for K in k_vals:
      evaluate_structured_output(output_parent_dir=f'/fs/vulcan-projects/two_step_synthesis_meshry/two_step_synthesis-old/drive_data_from_the_neural_talking_heads_paper/voxceleb2_results_{method}/{K}',
                                 gt_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/eval_subset_reorg/test',
                                 subset=None,
                                 concatenated_output=False,
                                 fake_resize_shape=(224, 224),
                                 metrics=('csim',),
                                 log_file_name='quantitative_metrics.txt',
                                 fake_img_ext='jpg',
                                 experiment_title=f'FSTH-{method} -- K={K}',
                                 description=f'FSTH-{method} released output for K={K}.')


def compare_csim3():
  val_set_dirs = [
      '/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp15-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2',
      # '/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp16-finetune_with_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2',
      # '/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/talking_heads-exp17-spade-full_losses-d_stylegan2',
      # '/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/talking_heads-exp18-unet-full_losses-d_stylegan2',
  ]
  k_vals = [1,]  # [1,8,32]
  all_logs = ''
  for i, val_set_dir in  enumerate(val_set_dirs):
    for k in k_vals:
      out_dir = osp.join(val_set_dir, 'subject_finetuning', f'{k}')
      # out_dir = osp.join(val_set_dir, 'subject_finetuning', f'{k}-024steps-high_recon')
      if not osp.exists(out_dir):
        continue
      exp_title = val_set_dir.split('/')[-1]
      log_str = evaluate_structured_output(
          output_parent_dir=out_dir,
          gt_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/eval_subset_reorg/test',
          subset='test',
          concatenated_output=True,
          fake_resize_shape=None,
          metrics=('csim',),
          log_file_name='quantitative_metrics.txt',
          experiment_title=f'{exp_title} - subject_ft - K={k}',
          description='')
      all_logs += log_str + '\n'
      print(all_logs)


def main(argv):
  # compare_csim()
  # compare_csim2()
  # compare_csim3()

  # evaluate_experiment(
  #     FLAGS.val_set_out_dir,
  #     output_name_pattern=FLAGS.output_name_pattern,
  #     experiment_title=FLAGS.experiment_title,
  #     metrics=['l1', 'psnr', 'ssim', 'lpips', 'fid', 'csim'])

  # # Evaluate FOMM and X2Face.
  # baseline = 'FOMM_online'  # FOMM, x2face, FOMM_newgens, FOMM_online, FOMM_offline
  # evaluate_structured_output(output_parent_dir=f'/fs/vulcan-projects/two_step_synthesis_meshry/results/{baseline}/test',
  #                            gt_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/eval_subset_reorg/test',
  #                            subset=None,
  #                            concatenated_output=False,
  #                            fake_resize_shape=(224, 224),
  #                            metrics=('l1', 'psnr', 'ssim', 'lpips', 'fid', 'csim'),
  #                            log_file_name='quantitative_metrics.txt',
  #                            experiment_title=f'{baseline}',
  #                            description=f'{baseline} Baseline.')

  # # FSTH released output
  # method = 'ft'  # ff, ft
  # K = 32  # 1, 8, 32
  # evaluate_structured_output(output_parent_dir=f'/fs/vulcan-projects/two_step_synthesis_meshry/two_step_synthesis-old/drive_data_from_the_neural_talking_heads_paper/voxceleb2_results_{method}/{K}',
  #                            gt_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/eval_subset_reorg/test',
  #                            subset=None,
  #                            concatenated_output=False,
  #                            fake_resize_shape=(224, 224),
  #                            metrics=('l1', 'psnr', 'ssim', 'lpips', 'fid', 'csim'),
  #                            log_file_name='quantitative_metrics.txt',
  #                            fake_img_ext='jpg',
  #                            experiment_title=f'FSTH-{method} -- K={K}',
  #                            description=f'FSTH-{method} released output for K={K}.')

  # Evaluate subject fine-tuning.
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp10-finetune_no_x_ent-segmap_nf16-layer_norm-sep_layout_enc-multi_task/subject_finetuning/1'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp10-finetune_no_x_ent-segmap_nf16-layer_norm-sep_layout_enc-multi_task/subject_finetuning/8'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp10-finetune_no_x_ent-segmap_nf16-layer_norm-sep_layout_enc-multi_task/subject_finetuning/32'
  # ----------
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp15-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2/subject_finetuning/1'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp15-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2/subject_finetuning/4'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp15-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2/subject_finetuning/8'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp15-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2/subject_finetuning/32'
  # ----------
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp16-finetune_with_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2/subject_finetuning/1'
  # ----------
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp16_rerun-finetune_with_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2/subject_finetuning/1'
  # ----------
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/talking_heads-exp17-spade-full_losses-d_stylegan2/subject_finetuning/1'
  # ----------
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/talking_heads-exp18-unet-full_losses-d_stylegan2/subject_finetuning/1'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/talking_heads-exp18-unet-full_losses-d_stylegan2/subject_finetuning/8'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/talking_heads-exp18-unet-full_losses-d_stylegan2/subject_finetuning/32'
  # ----------
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp20-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2-resume_exp15/subject_finetuning/1'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp20-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2-resume_exp15/subject_finetuning/4'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp20-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2-resume_exp15/subject_finetuning/8'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/train_v2/two_step_synthesis_exp20-finetune_no_x_ent-segmap_nf32-inst_norm-sep_layout_enc-d_stylegan2-resume_exp15/subject_finetuning/32'
  # ----------
  # evaluate_structured_output(output_parent_dir=output_parent_dir,
  #                            gt_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/eval_subset_reorg/test',
  #                            subset='test',
  #                            concatenated_output=True,
  #                            fake_resize_shape=None,
  #                            metrics=('l1', 'psnr', 'ssim', 'lpips', 'fid', 'csim'),
  #                            log_file_name='quantitative_metrics.txt',
  #                            experiment_title='Exp?? - (subject_ft - K=?)',
  #                            description='')
  # ----------
  # Evaluate LPD & bilayer
  output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/bilayer_with_black_bg/test'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/LPD_finetuned/LPD_processed/test/1'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/LPD_finetuned/LPD_processed/test/4'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/LPD_finetuned/LPD_processed/test/8'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/LPD_finetuned/LPD_processed/test/32'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/LPD_non_finetuned/nf_synthesized_processed/test/1'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/LPD_non_finetuned/nf_synthesized_processed/test/4'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/LPD_non_finetuned/nf_synthesized_processed/test/8'
  # output_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/LPD_non_finetuned/nf_synthesized_processed/test/32'
  evaluate_structured_output(output_parent_dir=output_parent_dir,
                             gt_parent_dir='/fs/vulcan-projects/two_step_synthesis_meshry/results/GT_LPD/preprocess_sanity/test-images-masked',
                             subset=None,
                             concatenated_output=False,
                             fake_resize_shape=None,
                             metrics=('l1', 'psnr', 'ssim', 'lpips', 'fid', 'csim'),
                             log_file_name='quantitative_metrics.txt',
                             fake_img_ext='png',
                             real_img_ext='jpg',
                             real_img_suffix=None,
                             experiment_title='Exp?? - (subject_ft - K=?)',
                             description='')


if __name__ == '__main__':
  app.run(main)
