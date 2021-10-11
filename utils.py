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


"""Some basic utility functions."""

import cv2
import numpy as np
import tensorflow.compat.v1 as tf


def denormalize_image(image: tf.Tensor) -> tf.Tensor:
  """Rescales image from [-1, 1] to [0, 255]"""
  return (tf.clip_by_value(image, -1, 1) + 1) * 255. / 2.


def get_foreground_mask(one_hot_seg: tf.Tensor,
                        synthesize_background: bool) -> tf.Tensor:
  """Returns foreground mask for a CelebA-Mask-HQ semantic segmenter."""
  if synthesize_background:
    return 1.
  else:
    return 1 - tf.slice(one_hot_seg, [0, 0, 0, BACKGROUND_IDX], [-1, -1, -1, 1])


def crop_to_multiple(img, size_multiple=64):
  """Crops the image so that its dimensions are multiples of size_multiple."""
  new_width = (img.shape[1] // size_multiple) * size_multiple
  new_height = (img.shape[0] // size_multiple) * size_multiple
  offset_x = (img.shape[1] - new_width) // 2
  offset_y = (img.shape[0] - new_height) // 2
  return img[offset_y:offset_y + new_height, offset_x:offset_x + new_width, :]


def load_variable_from_checkpoint(checkpoint_dir, var_prefix):
  """
  Returns the tensor value if found, returns None if not found, and fails
   if multiple checkpoint variables match the given `var_prefix`.
  """
  if tf.train.latest_checkpoint(checkpoint_dir):
    var_list = tf.train.list_variables(
        tf.train.latest_checkpoint(checkpoint_dir))
    matches = [x for x in var_list if x[0].startswith(var_prefix)]
    assert len(matches) <= 1, (
        'Retreiving var_prefix="%s" found %d matches in %s.' % (
            var_prefix, len(matches), checkpoint_dir))
    if len(matches) == 1:
      var_name = matches[0][0]
      ckpt_reader = tf.compat.v1.train.NewCheckpointReader(
          tf.train.latest_checkpoint(checkpoint_dir))
      return ckpt_reader.get_tensor(var_name)
  return None


def to_png_numpy(x, channel_format='RGB'):
  """Convert a 3D numpy array to png.

  Args:
    x: ndarray, 01C formatted input image.

  Returns:
    ndarray, 1D string representing the image in png format.
  """
  y = np.clip(np.round(127.5 + 127.5 * x), 0, 255).astype(np.uint8)
  if channel_format == 'RGB':
    y = y[:, :, ::-1]  # Convert to BGR.
  raw_image = cv2.imencode('.png', y)[1]
  return raw_image.tobytes()


def to_png(x):
  """Convert a 3D tensor to png.

  Args:
    x: Tensor, 01C formatted input image.

  Returns:
    Tensor, 1D string representing the image in png format.
  """
  with tf.Graph().as_default():
    with tf.Session() as sess_temp:
      x = tf.constant(x)
      y = tf.image.encode_png(
          tf.cast(
              tf.clip_by_value(tf.round(127.5 + 127.5 * x), 0, 255), tf.uint8),
          compression=9)
      return sess_temp.run(y)

