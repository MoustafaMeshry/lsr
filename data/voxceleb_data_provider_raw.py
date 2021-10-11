"""VoxCeleb2 Data provider from raw files for few-shot image synthesis."""

from PIL import Image
# from options import FLAGS as opts
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


def _denormalize_image(image, min_value=-1, dtype='float32'):
  if min_value == -1:
    image = (image + 1) / 2.
  image = image * 255.
  if tf.is_tensor(image):
    return tf.dtypes.cast(image, dtype)
  else:
    return image.astype(dtype)


def _read_image(img_path, normalize=False, dtype=np.float32):
  img = np.array(Image.open(img_path))
  if normalize:
    img = img.astype(np.float32) * 2. / 255. - 1
  if dtype is not None:
    img = img.astype(dtype)
  return img


def _build_example(video_part_path,
                   K=4,
                   use_segmaps=True,
                   randomize_frames=False,
                   max_sequence_frames=-1,
                   augmentation_flag=False):
  assert not augmentation_flag, 'Augmentations not yet supported for voxceleb!'
  # List image paths for all frames.
  video_part_path = video_part_path.decode('utf-8')
  rgbs = np.array(sorted(glob.glob(osp.join(video_part_path, '*_rgb*'))))
  contours = np.array(sorted(glob.glob(
    osp.join(video_part_path, '*_contour*'))))
  if use_segmaps:
    segmaps = np.array(sorted(glob.glob(
        osp.join(video_part_path, '*_segmap_merged*'))))
  else:
    segmaps = [None] * len(rgbs)
  assert len(rgbs) > 0, 'No matching files found!'
  assert len(rgbs) == len(contours) and len(rgbs) == len(segmaps)

  # Clamp number of frames per sequence deterministically.
  if max_sequence_frames > 0:
    assert max_sequence_frames <= len(rgbs), (
        'Required number of frames (%d) < number of available frames (%d)' % (
            max_sequence_frames, len(rgbs)))
    assert K <= max_sequence_frames
    sample_idxs = np.linspace(0, len(rgbs), num=max_sequence_frames,
                              endpoint=False,dtype=int)
    rgbs = rgbs[sample_idxs]
    contours = contours[sample_idxs]
    segmaps = segmaps[sample_idxs]
  # Sample K+1 frames.
  if randomize_frames:
    # Note that if K == len(rgbs), this will lead to idxs of length K (not K+1)
    #  and the target frame will be the kth element.
    idxs = np.random.permutation(len(rgbs))[:K+1]
  else:
    idxs = np.arange(K+1)
    if K == len(rgbs):
      idxs[-1] = -1
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
  return (person_id, video_id, video_part_id, rgbs[:K], rgbs[-1],
          contours[:K], contours[-1], segmaps[:K], segmaps[-1])


def _read_raw_train_example(video_part_path,
                            K=4,
                            use_segmaps=True,
                            randomize_frames=False,
                            frame_shape=[224, 224],
                            max_sequence_frames=-1,
                            augmentation_flag=False):

  build_example_py_func = functools.partial(
      _build_example, K=K, randomize_frames=randomize_frames,
      max_sequence_frames=max_sequence_frames)
  (person_id, video_id, video_part_id, frames_in, frame_gt, contours_in,
   contour_gt, segmaps_in, segmap_gt) = tf.compat.v1.py_func(
       build_example_py_func, [video_part_path],
       [tf.dtypes.string] * 3 + [tf.dtypes.float32] * 4 + [tf.dtypes.uint8] * 2)

  # Restore shapes after applying py_func.
  frames_in.set_shape([K] + frame_shape + [3])
  frame_gt.set_shape(frame_shape + [3])
  contours_in.set_shape([K] + frame_shape + [3])
  contour_gt.set_shape(frame_shape + [3])
  segmaps_in.set_shape([K] + frame_shape)
  segmap_gt.set_shape(frame_shape)

  # Add frames, contours and segmaps to the output dictionary.
  preprocessed_element = dict()
  preprocessed_element[PERSON_ID_KEY] = person_id
  preprocessed_element[VIDEO_ID_KEY] = video_id
  preprocessed_element[VIDEO_PART_ID_KEY] = video_part_id
  preprocessed_element[FRAMES_KEY] = frames_in
  preprocessed_element[TARGET_FRAME_KEY] = frame_gt
  preprocessed_element[CONTOURS_KEY] = contours_in
  preprocessed_element[TARGET_CONTOUR_KEY] = contour_gt
  preprocessed_element[SEGMAPS_KEY] = segmaps_in
  preprocessed_element[TARGET_SEGMAP_KEY] = segmap_gt

  return preprocessed_element


def provide_data(data_source_pattern: Text,
                 use_segmaps: bool,
                 batch_size: int,
                 k_frames: int = 8,
                 num_concatenations: int = 10,
                 is_training: bool = False,
                 repeat: int = 1,
                 max_examples: int = -1,
                 max_sequence_frames: int = -1,
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
  video_parts_filepaths = sorted(glob.glob(data_source_pattern))
  dataset = tf.data.Dataset.from_tensor_slices(video_parts_filepaths)
  # Disable autosharding
  ds_options = tf.data.Options()
  ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  dataset = dataset.with_options(ds_options)

  if is_training or shuffle:
    dataset = dataset.shuffle(_SHUFFLE_BUFFER_SIZE)
  if repeat != 1:  # -1 or None means repeat indefinitely.
    dataset = dataset.repeat(repeat)
  elif max_examples != -1:
    dataset = dataset.take(max_examples)

  # Create few-shot input examples.
  dataset = dataset.map(
      functools.partial(
          _read_raw_train_example, K=k_frames, randomize_frames=is_training,
          max_sequence_frames=max_sequence_frames),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Batch and prefetch.
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def print_minibatch(minibatch, show_images=False):
    for k, v in minibatch.items():
        v = v.numpy()
        batch_size = v.shape[0]
        # iterate over examples in the minibatch
        for i in range(batch_size):
          v_i = v[i,]
          if k in ['frames_in', 'contours_in', 'segmaps_in']:
            print(f'{k}: {v_i.shape}')
            if show_images:
              img_list = [x for x in v_i]
              print(len(img_list), img_list[0].shape)
              conc_img = np.concatenate(img_list, axis=1)
              print(conc_img.shape, conc_img.dtype)
              if 'segmap' not in k:
                conc_img = _denormalize_image(conc_img, dtype=np.uint8)
              Image.fromarray(conc_img).show(title=k)
          elif k in ['frame_gt', 'contour_gt', 'segmap_gt']:
            print(f'{k}: {v_i.shape}')
            if show_images:
              img = v_i if 'segmap' in k else _denormalize_image(v_i, dtype=np.uint8)
              Image.fromarray(img).show(title=k)
          else:
            print(f'{k}: {v_i}')
        print('-------------------------------')

