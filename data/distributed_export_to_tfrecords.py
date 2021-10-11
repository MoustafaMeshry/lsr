"""Exports the VoxCeleb2 dataset to TFRecords format."""

from absl import app
from absl import flags
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


args = flags.FLAGS
flags.DEFINE_integer('person_id_st', -1, '')
flags.DEFINE_integer('person_id_end', -1, '')
flags.DEFINE_integer('shard_idx', 0, '')
flags.DEFINE_integer('num_shards', 1000, '')
flags.DEFINE_string('output_parent_dir', '/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/tfrecords', '')


def _to_tf_example(dictionary, serialize_example=True):
  """Builds a tf.Example as a (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    # If v is an eager tensor, convert to it numpy value.
    if isinstance(v, type(tf.constant(0))):
      v = v.numpy()

    if isinstance(v, six.integer_types):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))
    elif isinstance(v, float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=[v]))
    elif isinstance(v, six.string_types):
      # if not six.PY2:  # Convert in python 3.
      #   v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    elif isinstance(v, bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))

  example_proto = tf.train.Example(features=tf.train.Features(feature=features))
  if serialize_example:
    return example_proto.SerializeToString()
  else:
    return example_proto


class VoxCelebSubset(object):
  def __init__(self,
               person_dirs_list,
               num_frames_per_example=10,
               serialize_features=False,
               image_features=('rgb', 'contour', 'segmap')):
    """
    Args:
      video_parts_pattern: string, file pattern to video parts directories that
       contains the extracted frames and their associated facial landmarks and
       semantic segmentations.
      num_frames_per_example: int, the number of frames to concatenate in each
       TFExample.
      serialize_features: boolean, whether to convert non-scalar features to
       binary strings.
    """
    self.num_frames_per_example = num_frames_per_example
    self.serialize_features = serialize_features
    self.image_features = image_features
    self.iter_idx = 0
    dir_paths = []
    for person_dir in person_dirs_list:
      dir_paths += sorted(glob.glob(osp.join(person_dir, '*', '*')))
    self.dir_paths = sorted(dir_paths)
    assert len(self.dir_paths) > 0, ('input %s didn\'t match any files!' %
                                     str(person_dirs_list))
    print('DBG: len(examples) = %d.' % len(self.dir_paths))

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def __call__(self):
    return self.__next__()

  def next(self):
    if self.iter_idx >= len(self.dir_paths):
      raise StopIteration()

    raw_example = dict()
    vpid_path = self.dir_paths[self.iter_idx]
    num_frames = -1
    for feature in self.image_features:
      # Hack to load the merged sgmaps.
      if feature == 'segmap':
        file_paths = sorted(glob.glob(osp.join(vpid_path, f'*_{feature}_merged.png')))
      else:
        file_paths = sorted(glob.glob(osp.join(vpid_path, f'*_{feature}.png')))
      if num_frames == -1:
        num_frames = len(file_paths)
        if num_frames < self.num_frames_per_example:  # Skip this video part.
          print('Skipping %s: insufficient number of frames (%d vs %d).' % (
              vpid_path, num_frames, self.num_frames_per_example))
          self.iter_idx += 1
          return self.next()
      else:
        assert len(file_paths) == num_frames, (
            f'{len(file_paths)} vs {num_frames}')

      file_paths = file_paths[:self.num_frames_per_example]
      imgs = [np.array(Image.open(fpath)) for fpath in file_paths]
      # Concatenate frames along width.
      conc_imgs = np.concatenate(imgs, axis=1)
      #  if len(conc_imgs.shape) == 2:
      #    conc_imgs = conc_imgs[:, :, np.newaxis]
      if self.serialize_features:
        conc_imgs = tf.io.serialize_tensor(conc_imgs).numpy()
      raw_example[feature + 's'] = conc_imgs

    toks = vpid_path.split('/')
    raw_example['person_id'] = str.encode(toks[-3])
    raw_example['video_id'] = str.encode(toks[-2])
    raw_example['video_part_id'] = str.encode(toks[-1])
    # basenames = [int(osp.basename(x).split('_')[-2]) for x in file_paths]
    basenames = [osp.basename(x).split('_')[-2] for x in file_paths]
    if self.serialize_features:
      basenames = tf.io.serialize_tensor(basenames).numpy()
    raw_example['frame_ids'] = basenames
    self.iter_idx += 1

    return raw_example


# Export dataset to tfrecords.
def distributed_export_voxceleb_to_tfrecords(dataset_parent_dir,
                                             subset,
                                             output_parent_dir,
                                             person_id_st,
                                             person_id_end,
                                             shard_idx,
                                             num_shards,
                                             num_frames_per_video=10):
  """
  Exports the VoxCeleb2 dataset to TFRecords for few-shot synthesis.

  Args:
    - raw_data_source: string, pattern for video_parts directory which contains
       the extracted frames and their corresponding facial landmarks and
       semantic segmentations.
    - output_parent_dir: parent path for the output tfrecords.
    - subset: string, subset being processed to include in the tfrecords
       filename; one of {dev, test}.
    - num_shards: int, number of output shards for the tfrecords files.
    - num_frames_per_video: int, number of extracted frames from each video in
       the dataset.
  """
  os.makedirs(output_parent_dir, exist_ok=True)
  person_dirs = sorted(glob.glob(osp.join(dataset_parent_dir, subset, '*')))
  person_dirs = [person_dirs[i] for i in range(person_id_st, person_id_end)]
  assert len(person_dirs) > 0, 'shard=%d, st=%d, end=%d' % (shard_idx, person_id_st, person_id_end)
  # Generator function that iterates and yields all dataset examples.
  def generator():
    gen = VoxCelebSubset(person_dirs,
                         num_frames_per_example=num_frames_per_video,
                         serialize_features=True)
    for ex in gen:
      yield _to_tf_example(ex)
  
  output_path = osp.join(
      output_parent_dir,
      'vox2_%s-part-%04d-of-%04d.tfrecords' % (subset, shard_idx, num_shards))
  scratch_output_dir = '/scratch0/mmeshry/tmp'
  os.makedirs(scratch_output_dir, exist_ok=True)
  scratch_output_path = osp.join(
      scratch_output_dir,
      'vox2_%s-part-%04d-of-%04d.tfrecords' % (subset, shard_idx, num_shards))
  print('DBG: Creating serialized examples for %s...' % output_path)
  serialized_features_dataset = tf.data.Dataset.from_generator(
      generator, output_types=tf.string, output_shapes=())
  print('DBG: Writing tfrecord...')
  # writer = tf.data.experimental.TFRecordWriter(output_path)
  writer = tf.data.experimental.TFRecordWriter(scratch_output_path)
  writer.write(serialized_features_dataset)
  print('DBG: Finished writing tfrecord...')
  print('DBG: Moving tfrecord from /scratch0/* to target path...')
  shutil.move(scratch_output_path, output_path)
  print('DBG: Done moving file!')


def main(argv):
  dataset_parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2'
  subset = 'dev'
  # output_parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/tfrecords'
  # output_parent_dir = '/vulcanscratch/mmeshry/voxceleb_dev_backup/tfrecords'
  # output_parent_dir = '/fs/vulcan-projects/few_shot_neural_texture_meshry/voxceleb2_backup/preprocessed/tfrecords'
  output_parent_dir = args.output_parent_dir
  person_id_st = args.person_id_st 
  person_id_end = args.person_id_end 
  shard_idx = args.shard_idx 
  num_shards = args.num_shards 
  num_frames_per_video = 10

  # Export the dev (train) subset.
  distributed_export_voxceleb_to_tfrecords(
      dataset_parent_dir,
      subset,
      output_parent_dir,
      person_id_st,
      person_id_end,
      shard_idx,
      num_shards,
      num_frames_per_video=num_frames_per_video)
  print('Success! :)')


if __name__ == '__main__':
  app.run(main)
