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


from enum import Enum
from matplotlib import cm
import cv2
import numpy as np
import tensorflow as tf
import layers
from options import FLAGS as opts


class Mode(Enum):
  TRAIN = 0
  EVAL = 1
  PREDICT = 2


def visualize_one_hot_segmentation(segmap_one_hot, num_seg_classes=30,
                                   dtype='float32'):
  """TBD."""
  assert dtype in ['uint8', 'float32']
  seg_label_map = tf.argmax(segmap_one_hot, axis=-1)
  return visualize_label_map(seg_label_map, num_seg_classes=num_seg_classes,
                             dtype=dtype)

def visualize_label_map(seg_label_map, num_seg_classes=19, dtype='float32'):
  """TBD."""
  seg_label_map = tf.dtypes.cast(seg_label_map, tf.dtypes.int32)
  color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
                [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
                [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
                [0, 51, 0], [255, 153, 51], [0, 204, 0]]
  if num_seg_classes <= len(color_list):
    palette = np.array(color_list) / 255.
  else:
    palette = (cm.get_cmap('gist_rainbow')(np.linspace(
        0.0, 1.0, num_seg_classes))[:, :3])
    palette = palette[:, ::-1]  # BGR to RGB.
  if dtype == 'float32':
    palette = palette.astype(np.float32) * 2 - 1
  else:
    palette = (palette * 255).astype(np.uint8)
  _, H, W = seg_label_map.get_shape().as_list()
  class_indexes = tf.reshape(seg_label_map, [-1, H*W])
  segmap_vis = tf.gather(palette, class_indexes)
  segmap_vis = tf.reshape(segmap_vis, [-1, H, W, 3])
  return segmap_vis


@tf.function
def get_learning_rate_schedule(
    lr_init,
    global_step,
    lr_warmstart_steps=0,
    decay_start_step=np.inf,
    decay_end_step=np.inf,
    decay_num_intervals=100,
    starting_step=0,
    name=None,
    eps=1e-8):
  """TBD."""
  # Experimental:
  global_step = tf.dtypes.cast(global_step, tf.dtypes.float32)
  if global_step < starting_step + lr_warmstart_steps:
    alpha = (tf.dtypes.cast(
        global_step, tf.dtypes.float32) + eps - starting_step) / tf.dtypes.cast(
            lr_warmstart_steps, tf.dtypes.float32)
    lr = tf.dtypes.cast(lr_init * alpha, tf.dtypes.float32)
  elif global_step <= decay_start_step:  # 50000:
    lr = tf.dtypes.cast(lr_init, tf.dtypes.float32)
  else:
    alpha = (global_step - decay_start_step) / (
        decay_end_step - decay_start_step)
    scale = tf.math.maximum(np.float32(1), alpha * tf.dtypes.cast(
        decay_num_intervals, tf.dtypes.float32))
    lr = tf.dtypes.cast(lr_init / scale, tf.dtypes.float32)
  return lr


def _get_number_of_variables(tensor_shape):
  """TBD."""
  if not len(tensor_shape):
    return 0
  total_vars = 1
  for d in tensor_shape:
    total_vars *= d
  return total_vars


def print_model_summary(model, name='network', list_vars=False):
  """TBD."""
  total_params = 0
  net_vars = model.trainable_variables
  print('\n----------------------------------------------------------------')
  print('len(%s_vars) = %d' % (name, len(net_vars)))
  for i, v in enumerate(net_vars):
    if list_vars:
      print('%03d) name=%s, shape=%s, dtype=%s' % (i, v.name, v.shape, v.dtype))
    total_params += _get_number_of_variables(v.shape.as_list())
  total_bytes = total_params * 32 / 8
  print('len(%s_vars) = %d' % (name, len(net_vars)))
  print('Total float32 parameters for %s = %d' % (name, total_params))
  print('%s network size = %.f MB' % (name, (total_bytes / 1024 / 1024)))
  print('----------------------------------------------------------------\n')
  return total_params, total_bytes


def get_train_op(gradient_tape, optimizer, loss, var_list, grad_clip_abs_val=-1):
  """TBD."""
  grads = gradient_tape.gradient(loss, var_list)
  grads_and_vars = zip(grads, var_list)
  if grad_clip_abs_val > 0:
    grads_and_vars = [(tf.clip_by_value(grad, -1*grad_clip_abs_val,
                                        grad_clip_abs_val), var)
                      for grad, var in grads_and_vars]
  train_op = optimizer.apply_gradients(grads_and_vars)
  return train_op

