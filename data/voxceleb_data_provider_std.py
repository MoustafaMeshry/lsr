"""
VoxCeleb2 Data provider from raw image files for few-shot image synthesis.
 This data provider assumes the data is in the format of the standard test
 subset provided by Zakharov et. al ("Few-Shot Adversarial Learning of Realistic
 Neural Talking Head Models") Each subject has a train and test directories.
 The input K-shots will be created from the subject `train` dir and driving
 sequence will be created from the `test` dir.
"""

from PIL import Image
from typing import Any, Dict, List, Optional, Set, Text, Tuple, Union
import functools
import glob
import numpy as np
import os
import os.path as osp
import shutil
import six
import tensorflow as tf


_SHUFFLE_BUFFER_SIZE = 256
_DATASET_KEYS = ('person_id', 'video_id', 'video_part_id',
                 'rgbs', 'contours', 'segmaps', 'frame_ids')
_KEYS_TO_PROCESS = _DATASET_KEYS[:-1]
_KEYS_TO_PROCESS_NO_SEGMAPS = _DATASET_KEYS[:-2]

PERSON_ID_KEY = 'person_id'
VIDEO_ID_KEY = 'video_id'
VIDEO_PART_ID_KEY = 'video_part_id'
FRAMES_KEY = 'frames_in'
TARGET_FRAME_KEY = 'frame_gt'
CONTOURS_KEY = 'contours_in'
TARGET_CONTOUR_KEY = 'contour_gt'
SEGMAPS_KEY = 'segmaps_in'
TARGET_SEGMAP_KEY = 'segmap_gt'


def _read_image(img_path, normalize=False, dtype=np.float32):
  img = np.array(Image.open(img_path))
  if normalize:
    img = img.astype(np.float32) * 2. / 255. - 1
  if dtype is not None:
    img = img.astype(dtype)
  return img


def _get_video_part_data(video_part_path,
                         use_segmaps=True,
                         max_frames=-1,
                         randomize_frames=False):
  # List image paths for all frames.
  video_part_path = video_part_path.numpy().decode('utf-8')
  rgbs = np.array(sorted(glob.glob(osp.join(video_part_path, '*_rgb*'))))
  contours = np.array(sorted(glob.glob(
    osp.join(video_part_path, '*_contour*'))))
  if use_segmaps:
    segmaps = np.array(sorted(glob.glob(
        osp.join(video_part_path, '*_segmap_merged*'))))
  else:
    segmaps = np.array([None] * len(rgbs))
  assert len(rgbs) > 0, 'No matching files found!'
  assert len(rgbs) == len(contours) and len(rgbs) == len(segmaps)
  assert len(rgbs) >= max_frames
  if max_frames > -1:
    if randomize_frames:
      idxs = np.random.permutation(len(rgbs))[:max_frames]
    else:
      idxs = np.linspace(0, len(rgbs), num=max_frames, endpoint=False,
                         dtype=int)
    rgbs = rgbs[idxs]
    contours = contours[idxs]
    segmaps = segmaps[idxs]

  # Read frames
  rgbs = [_read_image(img_path, normalize=True) for img_path in rgbs]
  contours = [_read_image(img_path, normalize=True) for img_path in contours]
  if use_segmaps:
    segmaps = [_read_image(img_path, dtype=np.uint8) for img_path in segmaps]
  else:
    segmaps = [np.zeros(rgbs[0].shape[:2], dtype=np.uint8)] * len(rgbs)

  toks = video_part_path.split('/')
  person_id = str(toks[-3])
  video_id = str(toks[-2])
  video_part_id = str(toks[-1])
  return person_id, video_id, video_part_id, rgbs, contours, segmaps


def _build_examples_mapper(train_video_part_path,
                           test_video_part_path,
                           k_frames=4,
                           max_frames=-1,
                           same_video_eval=True,
                           use_segmaps=True,
                           randomize_frames=False,
                           frame_shape=[224, 224]):

  
  build_examples_py_func = functools.partial(
      _build_examples,
      k_frames=k_frames,
      use_segmaps=use_segmaps,
      same_video_eval=same_video_eval,
      randomize_frames=randomize_frames,
      max_frames=max_frames)
  (person_ids_all, video_ids_all, video_part_ids_all,
   rgbs_in_all, target_rgb_all,
   contours_in_all, target_contour_all,
   segmaps_in_all, target_segmap_all) = tf.py_function(
       build_examples_py_func, [train_video_part_path, test_video_part_path],
       [tf.dtypes.string] * 3 + [tf.dtypes.float32] * 4 + [tf.dtypes.uint8] * 2)

  # Restore shapes after applying py_func.
  num_frames = max_frames if max_frames > -1 else None
  rgbs_in_all.set_shape([num_frames] + [k_frames] + frame_shape + [3])
  target_rgb_all.set_shape([num_frames] + frame_shape + [3])
  contours_in_all.set_shape([num_frames] + [k_frames] + frame_shape + [3])
  target_contour_all.set_shape([num_frames] + frame_shape + [3])
  segmaps_in_all.set_shape([num_frames] + [k_frames] + frame_shape)
  target_segmap_all.set_shape([num_frames] + frame_shape)

  dataset = tf.data.Dataset.from_tensor_slices(
      (person_ids_all,
       video_ids_all,
       video_part_ids_all,
       rgbs_in_all,
       target_rgb_all,
       contours_in_all,
       target_contour_all,
       segmaps_in_all,
       target_segmap_all))
  ds_options = tf.data.Options()
  ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  dataset = dataset.with_options(ds_options)
  return dataset



def _build_examples(train_video_part_path,
                    test_video_part_path,
                    k_frames=4,
                    max_frames=-1,
                    use_segmaps=True,
                    same_video_eval=True,
                    randomize_frames=False,
                    augmentation_flag=False):
  """TBD."""
  dev_data = _get_video_part_data(
      train_video_part_path,
      use_segmaps=use_segmaps,
      max_frames=k_frames,
      randomize_frames=randomize_frames)
  eval_data = _get_video_part_data(
      test_video_part_path,
      use_segmaps=use_segmaps,
      max_frames=max_frames)
  (dev_person_id, dev_video_id, dev_video_part_id, dev_rgbs, dev_contours,
   dev_segmaps) = dev_data
  (eval_person_id, eval_video_id, eval_video_part_id, eval_rgbs, eval_contours,
   eval_segmaps) = eval_data
  if same_video_eval:
    assert dev_person_id == eval_person_id
    assert dev_video_id == eval_video_id

  person_ids_all, video_ids_all, video_part_ids_all = [], [], []
  rgbs_in_all, target_rgb_all = [], []
  contours_in_all, target_contour_all = [], []
  segmaps_in_all, target_segmap_all = [], []
  for idx, (rgb, contour, segmap) in enumerate(zip(eval_rgbs, eval_contours,
                                                   eval_segmaps)):
    person_ids_all += [eval_person_id]
    video_ids_all += [eval_video_id]
    video_part_ids_all += [eval_video_part_id]
    rgbs_in_all += [dev_rgbs]
    target_rgb_all += [rgb] 
    contours_in_all += [dev_contours]
    target_contour_all += [contour]
    segmaps_in_all += [dev_segmaps]
    target_segmap_all += [segmap]

  return (person_ids_all, video_ids_all, video_part_ids_all,
          rgbs_in_all, target_rgb_all,
          contours_in_all, target_contour_all,
          segmaps_in_all, target_segmap_all)


def validate_and_align_data(train_paths, test_paths):
  # TODO: validate that train and test paths align correctly, and remove
  #  unaligned paths if possible.
  assert len(train_paths) == len(test_paths), (
      'Unmatching train and test paths lenghts: %d vs %d.' % (
          len(train_paths), len(test_paths)))
  for train_path, test_path in zip(train_paths, test_paths):
    train_toks = train_path.split('/')
    train_person_id, train_video_id = train_toks[-3], train_toks[-2]
    test_toks = test_path.split('/')
    test_person_id, test_video_id = test_toks[-3], test_toks[-2]
    # The following assertions are turned off to allow cross-subject reenactment.
    # assert train_person_id == test_person_id, 'Unaligned person ids %s vs %s' % (
    #     train_person_id, test_person_id)
    # assert train_video_id == test_video_id, 'Unaligned video ids %s vs %s' % (
    #     train_video_id, test_video_id)

  return train_paths, test_paths 


def provide_data(train_data_source_pattern: Text,
                 test_data_source_pattern: Text,
                 use_segmaps: bool,
                 batch_size: int,
                 k_frames: int = 4,
                 max_frames: int = -1,
                 num_concatenations: int = 10,
                 is_training: bool = False,
                 repeat: int = 1,
                 max_examples: int = -1,
                 same_video_eval: bool = True,
                 randomize_frames: bool = False,
                 shuffle: bool = False):
  """TBD.

  Args:
    data_sources: file pattern of tfrecords.
    batch_size: The number of images in each batch.
    k_frames: k number for k-shot learning.
    num_concatenations: Number of width-concatenated frames for SSTable parsing.
    is_training: Indicates whether the provided data is used for training. If
      True, random data augmentation and shuffling are performed.
    max_examples: Number of examples to subsample, -1 to return the full
      dataset.
    shuffle: Only considered if is_training=False. Controls whether to shuffle
      data or not.

  Returns:
    A dictionary dataset containing
     `person_id`: voxceleb2 subject id.
     `video_id`: voxceleb2 video id.
     `video_part_id`: voxceleb2 video part id.
     `frames_in`: the K-shot RGB inputs.
     `frame_gt`: the ground truth RGB output for the target pose.
     `contours_in`: the corresponding landmarks for the K-shot inputs. See
       --concat_contours_to_layout to optionally feed the landmarks to the
       identity encoder.
     `contour_gt`: the facial landmarks for the target pose.
     `segmaps_in`: the oracle segmentation maps for the K-shot inputs.
     `segmaps_gt`: the oracle segmentation map for the ground truth target pose.
  """
  assert not (repeat != 1 and max_examples != -1), ('Only one of `repeat` and '
                                                   '`max_examples` can be set!')
  if use_segmaps:
    keys_to_process = _KEYS_TO_PROCESS
  else:
    keys_to_process = _KEYS_TO_PROCESS_NO_SEGMAPS
  train_video_parts_filepaths = sorted(glob.glob(train_data_source_pattern))
  test_video_parts_filepaths = sorted(glob.glob(test_data_source_pattern))
  ds_options = tf.data.Options()
  ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  (train_video_parts_filepaths,
   test_video_parts_filepaths) = validate_and_align_data(
       train_video_parts_filepaths, test_video_parts_filepaths)
  dataset = tf.data.Dataset.from_tensor_slices(
      (train_video_parts_filepaths, test_video_parts_filepaths))
  dataset = dataset.with_options(ds_options)

  # Create few-shot input examples.
  dataset = dataset.flat_map(
      functools.partial(
          _build_examples_mapper,
          k_frames=k_frames,
          max_frames=max_frames,
          use_segmaps=use_segmaps,
          randomize_frames=randomize_frames,
          same_video_eval=same_video_eval))
  # Map each dataset example to dictionary.
  dataset = dataset.map(
      (lambda person_id, vid_id, vid_part_id, k_rgbs, rgb, k_contours, \
             contour, k_segmaps, segmap: {
                 PERSON_ID_KEY: person_id,
                 VIDEO_ID_KEY: vid_id,
                 VIDEO_PART_ID_KEY: vid_part_id,
                 FRAMES_KEY: k_rgbs,
                 TARGET_FRAME_KEY: rgb,
                 CONTOURS_KEY: k_contours,
                 TARGET_CONTOUR_KEY: contour,
                 SEGMAPS_KEY: k_segmaps,
                 TARGET_SEGMAP_KEY: segmap}),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training or shuffle:
    dataset = dataset.shuffle(_SHUFFLE_BUFFER_SIZE)
  if repeat != 1:
    dataset = dataset.repeat(repeat)
  elif max_examples != -1:
    dataset = dataset.take(max_examples)

  # Batch and prefetch.
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

