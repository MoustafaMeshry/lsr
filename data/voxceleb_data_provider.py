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

"""VoxCeleb2 Data provider for few-shot image synthesis models."""

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


_SHUFFLE_BUFFER_SIZE = 1<<11  # 1<<13  # 256
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


def _normalize_image(image, min_value=-1, dtype='float32'):
  """Normalize image from [0,255] to [`min_value`,1]."""
  if tf.is_tensor(image):
    image = tf.dtypes.cast(image, dtype)
  else:
    assert isinstance(image, np.ndarray)
    image = image.astype(dtype)
  image = image / 255.
  if min_value == -1:
    image = image * 2. - 1
  return image

def _denormalize_image(image, min_value=-1, dtype='float32'):
  if min_value == -1:
    image = (image + 1) / 2.
  image = image * 255.
  if tf.is_tensor(image):
    return tf.dtypes.cast(image, dtype)
  else:
    return image.astype(dtype)


def read_tfrecord(serialized_example, num_frames_per_example):
  features_description = {
      'rgbs': tf.io.FixedLenFeature((), tf.dtypes.string),
      'contours': tf.io.FixedLenFeature((), tf.dtypes.string),
      'segmaps': tf.io.FixedLenFeature((), tf.dtypes.string),
      'person_id': tf.io.FixedLenFeature((), tf.dtypes.string),
      'video_id': tf.io.FixedLenFeature((), tf.dtypes.string),
      'video_part_id': tf.io.FixedLenFeature((), tf.dtypes.string),
      'frame_ids': tf.io.FixedLenFeature((), tf.dtypes.string)
  }
  parsed_example = tf.io.parse_single_example(
      serialized_example, features=features_description)

  parsed_features = dict()
  parsed_features['rgbs'] = tf.io.parse_tensor(
      parsed_example['rgbs'], out_type=tf.dtypes.uint8)
  # parsed_features['rgbs'].set_shape(
  #         (224, 224 * num_frames_per_example, 3))
  parsed_features['contours'] = tf.io.parse_tensor(
      parsed_example['contours'], out_type=tf.dtypes.uint8)
  # parsed_features['contours'].set_shape(
  #         (224, 224 * num_frames_per_example, 3))
  parsed_features['segmaps'] = tf.io.parse_tensor(
      parsed_example['segmaps'], out_type=tf.dtypes.uint8)
  # Add newaxis in the last dimension to convert segmaps to a 4D tensor.
  parsed_features['segmaps'] = tf.expand_dims(
      parsed_features['segmaps'], axis=-1)
  # parsed_features['segmaps'].set_shape(
  #         (224, 224 * num_frames_per_example, 1))
  parsed_features['person_id'] = parsed_example['person_id']
  parsed_features['video_id'] = parsed_example['video_id']
  parsed_features['video_part_id'] = parsed_example['video_part_id']
  parsed_features['frame_ids'] = tf.io.parse_tensor(
      parsed_example['frame_ids'], out_type=tf.dtypes.string)

  return parsed_features


def _preprocess(dataset_element,
                keys_to_process,
                use_segmaps,
                num_concatenations=10,
                k=4,
                randomize_input_frames=False):
  """Preprocesses a raw dataset example into a few-shot input example."""

  def _get_encoder_and_generator_inputs_py_func(frames, contours, segmaps):
    """Randomly shuffles and picks k+1 frames."""
    if randomize_input_frames:
      idxs = np.random.permutation(num_concatenations)
    else:
      idxs = np.arange(num_concatenations)
    return (frames[idxs[:k]], frames[idxs[-1]], contours[idxs[:k]],
            contours[idxs[-1]], segmaps[idxs[:k]], segmaps[idxs[-1]])

  def _normalize_and_split(conc_image):
    conc_image = tf.cast(conc_image, tf.float32) * 2. / 255. - 1
    # Split along width (axis=1)
    split = tf.split(
        conc_image, num_or_size_splits=num_concatenations, axis=1)
    return split

  unprocessed_element = dict(dataset_element)
  preprocessed_element = dict()
  for key in keys_to_process:
    if key not in ['rgbs', 'contours', 'segmaps']:
      preprocessed_element[key] = unprocessed_element[key]
    elif not randomize_input_frames:
      if key == 'rgbs':
        split = _normalize_and_split(unprocessed_element[key])
        preprocessed_element[FRAMES_KEY] = split[:k]
        preprocessed_element[TARGET_FRAME_KEY] = split[-1]
      elif key == 'contours':
        split = _normalize_and_split(unprocessed_element[key])
        preprocessed_element[CONTOURS_KEY] = split[:k]
        preprocessed_element[TARGET_CONTOUR_KEY] = split[-1]
      elif key == 'segmaps':
        split = tf.split(
            unprocessed_element[key], num_concatenations, axis=1)
        preprocessed_element[SEGMAPS_KEY] = tf.squeeze(split[:k], axis=-1)
        preprocessed_element[TARGET_SEGMAP_KEY] = tf.squeeze(split[-1], axis=-1)

  if randomize_input_frames:
    # TODO(meshry) You should ignore segmaps if segmaps is not in keys_to_process.
    split_frames = _normalize_and_split(unprocessed_element['rgbs'])
    split_contours = _normalize_and_split(unprocessed_element['contours'])
    if use_segmaps:
      split_segmaps = tf.split(
          unprocessed_element['segmaps'], num_concatenations, axis=1)
    else:
      # TODO # FIXME find a better workaround!
      split_segmaps = [
          tf.slice(x, [0, 0, 0], [-1, -1, 1]) for x in split_contours]
      split_segmaps = tf.dtypes.cast(split_segmaps, tf.dtypes.uint8)
      # print('***DBG: split_contours[0].shape = ', split_contours[0].shape)
      # split_segmaps = [tf.zeros(
      #     x.shape[:-1] + [1], dtype=tf.dtypes.uint8) for x in split_contours]

    frame_shape = split_frames[0].shape
    (frames_in, frame_gt, contours_in, contour_gt, segmaps_in,
     segmap_gt) = tf.compat.v1.py_func(_get_encoder_and_generator_inputs_py_func,
                                       [split_frames, split_contours, split_segmaps],
                                       [tf.dtypes.float32] * 4 + [tf.dtypes.uint8] * 2)

    # Restore shapes after applying py_func.
    frames_in.set_shape([k] + frame_shape)
    frame_gt.set_shape(frame_shape)
    contours_in.set_shape([k] + frame_shape)
    contour_gt.set_shape(frame_shape)
    segmaps_in.set_shape([k] + frame_shape[:-1] + [1])
    segmap_gt.set_shape(frame_shape[:-1] + [1])

    # Add frames, contours and segmaps to the output dictionary.
    preprocessed_element[FRAMES_KEY] = frames_in
    preprocessed_element[TARGET_FRAME_KEY] = frame_gt
    preprocessed_element[CONTOURS_KEY] = contours_in
    preprocessed_element[TARGET_CONTOUR_KEY] = contour_gt
    preprocessed_element[SEGMAPS_KEY] = tf.squeeze(segmaps_in, axis=-1)
    preprocessed_element[TARGET_SEGMAP_KEY] = tf.squeeze(segmap_gt, axis=-1)

  return preprocessed_element


def provide_data(data_source_pattern: Text,
                 use_segmaps: bool,
                 batch_size: int,
                 k_frames: int = 8,
                 num_concatenations: int = 10,
                 is_training: bool = False,
                 repeat: int = 1,
                 max_examples: int = -1,
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
  filepaths = sorted(glob.glob(data_source_pattern))
  dataset = tf.data.TFRecordDataset(filepaths)
  ds_options = tf.data.Options()
  ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  dataset = dataset.with_options(ds_options)
  if is_training or shuffle:
    dataset = dataset.shuffle(_SHUFFLE_BUFFER_SIZE)
  if repeat != 1:
    dataset = dataset.repeat(repeat)
  elif max_examples != -1:
    dataset = dataset.take(max_examples)
  # Decode tfrecords.
  dataset = dataset.map(
      functools.partial(
          read_tfrecord, num_frames_per_example=num_concatenations),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Create few-shot input examples.
  dataset = dataset.map(
      functools.partial(_preprocess,
                        keys_to_process=keys_to_process,
                        use_segmaps=use_segmaps,
                        num_concatenations=num_concatenations,
                        k=k_frames,
                        randomize_input_frames=is_training),
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

