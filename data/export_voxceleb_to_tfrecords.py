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

FLAGS = flags.FLAGS


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


class VoxCelebDataset(object):
  def __init__(self,
               video_parts_filepattern,
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
    self.dir_paths = sorted(glob.glob(video_parts_filepattern))
    assert len(self.dir_paths) > 0, ('input %s didn\'t match any files!' %
                                     video_parts_filepattern)

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
        assert len(file_paths) == num_frames

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


def _export_to_tfrecods(dataset_generator, output_path, num_shards=1):
  """Exports a serialized dataset to TFRecords.
  
  Args:
    dataset_generator: a generator function that iterates over the
     dataset examples and yields each serialized exampele.
    output_path: string, basepath for the tfrecords files.
    num_shards: int, number of shards for the exported dataset.
  """
  if output_path.endswith('.tfrecords'):
    output_path = output_path[:-10]  # removes the `.tfrecords` extension
  serialized_features_dataset = tf.data.Dataset.from_generator(
      dataset_generator, output_types=tf.string, output_shapes=())
  for i in range(num_shards):
      writer = tf.data.experimental.TFRecordWriter(
          f'{output_path}-part-%03d-of-%03d.tfrecords' % (i, num_shards))
      writer.write(serialized_features_dataset.shard(num_shards, i))


# Export dataset to tfrecords.
def export_voxceleb_to_tfrecords(raw_data_source,
                                 output_parent_dir,
                                 subset,
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
  # Generator function that iterates and yields all dataset examples.
  def generator():
    gen = VoxCelebDataset(raw_data_source,
                          num_frames_per_example=num_frames_per_video,
                          serialize_features=True)
    for ex in gen:
      yield _to_tf_example(ex)
  
  output_path = osp.join(output_parent_dir, f'vox2_{subset}.tfrecords')
  _export_to_tfrecods(generator, output_path=output_path, num_shards=num_shards)


def main(argv):
  subsets_and_shards = [('dev', 1000), ('test', 10)]
  print(f'Exporing the `{FLAGS.subset}` subset to tfrecords with '
        f'{FLAGS.num_shards} shards.')
  raw_train_data_source = osp.join(
      FLAGS.dataset_parent_dir, FLAGS.subset, FLAGS.person_pattern,
      FLAGS.video_pattern, FLAGS.video_part_pattern)
  export_voxceleb_to_tfrecords(
      raw_data_source=raw_train_data_source,
      output_parent_dir=FLAGS.output_parent_dir,
      subset=FLAGS.subset,
      num_shards=FLAGS.num_shards,
      num_frames_per_video=FLAGS.num_frames_per_video)


if __name__ == '__main__':
  flags.DEFINE_string(
      'dataset_parent_dir',
      '_datasets/preprocessed_voxceleb2',
      'Parent directory for the VoxCeleb2 dataset containing the `dev` and '
      '`test` directories.')
  flags.DEFINE_string(
      'output_parent_dir',
      '_datasets/preprocessed_voxceleb2/tfrecords',
      'Output path for the tfrecords.')
  flags.DEFINE_string(
      'person_pattern', '*', 'Name pattern for subjects to process.')
  flags.DEFINE_string(
      'video_pattern', '*', 'Name pattern for videos to process.')
  flags.DEFINE_string(
      'video_part_pattern', '*', 'Name pattern for video parts to process.')
  flags.DEFINE_integer(
      'num_frames_per_video', 10, 'Number of frames to sample from each video.')
  flags.DEFINE_string(
      'subset', 'dev', 'One of {dev, test}. Dataset subset to process.')
  flags.DEFINE_integer(
      'num_shards', 1000, 'Number of shards for the generated tfrecords. We '
      'used 1000 and 10 shards for the `dev` and `test` splits respectively.')

  app.run(main)
