# Tensorflow version == 2.0.0
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input


class VGGFaceLoss(object):
  """VGGFace-based perceptual and identity losses."""

  def __init__(self,
               pretrained_weights_path='/fs/vulcan-projects/two_step_synthesis_meshry/third_party/pretrained_weights/vgg_face_weights.h5',
               order='NHWC'):
    self.model = VGGFaceModel(pretrained_weights_path=pretrained_weights_path)

  def load_pretrained_weights(self, input_shape):
    self.model.load_pretrained_weights(input_shape=input_shape)

  # Takes two 4D numpy arrays of shapes [N,H,W,3] in range of [-1, 1].
  def __call__(self,
               y_true,
               y_pred,
               axis=None,
               layer_idxs=(-2,),
               layer_weights=(1,)):
    """TBD."""
    # Layer index -2 is the logits layer in VGGFace, representing identity info.
    #  Reconstruction layers use in (https://arxiv.org/abs/1905.08233) are the
    #  following layer indices: {1, 6, 11, 18, 25}.
    y_true = preprocess_input(((y_true + 1) / 2.) * 255.)
    y_pred = preprocess_input(((y_pred + 1) / 2.) * 255.)
    _, act1 = self.model(y_true)
    _, act2 = self.model(y_pred)
    total_loss = 0
    for w, idx in zip(layer_weights, layer_idxs):
      true_features, pred_features = act1[idx], act2[idx]
      total_loss += w * tf.reduce_mean(
          tf.math.abs(true_features - pred_features), axis=axis)
    return total_loss


class VGGFaceCSIM(object):
  """VGGFace-based perceptual and identity losses."""

  def __init__(self,
               pretrained_weights_path='/fs/vulcan-projects/two_step_synthesis_meshry/third_party/pretrained_weights/vgg_face_weights.h5',
               order='NHWC'):
    self.model = VGGFaceModel(pretrained_weights_path=pretrained_weights_path)
    self.cosine_loss = tf.keras.losses.CosineSimilarity(
        axis=-1, reduction=tf.keras.losses.Reduction.NONE)

  def load_pretrained_weights(self, input_shape):
    self.model.load_pretrained_weights(input_shape=input_shape)

  # Takes two 4D numpy arrays of shapes [N,H,W,3] in range of [-1, 1].
  def __call__(self,
               y_true,
               y_pred,
               axis=None):
    """TBD."""
    # Layer index -2 is the logits layer in VGGFace, representing identity info.
    #  Reconstruction layers use in (https://arxiv.org/abs/1905.08233) are the
    #  following layer indices: {1, 6, 11, 18, 25}.
    y_true = preprocess_input(((y_true + 1) / 2.) * 255.)
    y_pred = preprocess_input(((y_pred + 1) / 2.) * 255.)
    _, act1 = self.model(y_true)
    _, act2 = self.model(y_pred)
    face_embedding_idx = -2
    emb1 = act1[face_embedding_idx]
    emb2 = act2[face_embedding_idx]
    csim = -1 * self.cosine_loss(emb1, emb2)
    csim = tf.expand_dims(csim, axis=-1)
    csim = tf.reduce_mean(csim, axis=axis)

    return csim

class VGGFaceModel(tf.keras.Model):
  """Defines the vgg_face model architecture."""

  def __init__(self,
               pretrained_weights_path='/fs/vulcan-projects/two_step_synthesis_meshry/third_party/pretrained_weights/vgg_face_weights.h5',
               trainable=False):
    """TBD."""
    super(VGGFaceModel, self).__init__(name='vgg_face')
    self.pretrained_weights_path = pretrained_weights_path
    layers = []
    layers.append(tf.keras.layers.ZeroPadding2D((1,1),input_shape=(224,224, 3)))  # 00
    layers.append(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu'))   # 01*
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 02
    layers.append(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu'))   # 03
    layers.append(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))             # 04
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 05 
    layers.append(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu'))  # 06*
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 07
    layers.append(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu'))  # 08
    layers.append(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))             # 09
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 10
    layers.append(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu'))  # 11*
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 12
    layers.append(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu'))  # 13
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 14
    layers.append(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu'))  # 15
    layers.append(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))             # 16
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 17
    layers.append(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu'))  # 18*
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 19
    layers.append(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu'))  # 20
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 21
    layers.append(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu'))  # 22
    layers.append(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))             # 23
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 24
    layers.append(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu'))  # 25*
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 26
    layers.append(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu'))  # 27
    layers.append(tf.keras.layers.ZeroPadding2D((1,1)))                           # 28
    layers.append(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu'))  # 29
    layers.append(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))             # 30
    layers.append(
        tf.keras.layers.Convolution2D(4096, (7, 7), activation='relu'))           # 31
    # layers.append(tf.keras.layers.Dropout(0.5))                                   # 32  removed!
    layers.append(
        tf.keras.layers.Convolution2D(4096, (1, 1), activation='relu'))           # 33 -> 32
    # layers.append(tf.keras.layers.Dropout(0.5))                                   # 34  removed!
    layers.append(tf.keras.layers.Convolution2D(2622, (1, 1)))                    # 35* -> 33*
    layers.append(tf.keras.layers.Flatten())                                      # 36 -> 34
    layers.append(tf.keras.layers.Activation('softmax'))                          # 37 -> 35
    self.net_layers = layers
    self.trainable = trainable

  def build(self, input_shape):
    super(VGGFaceModel, self).build(input_shape)
    # self.load_weights(self.pretrained_weights_path)
    # print('vgg_face weights loaded!')

  def load_pretrained_weights(self, input_shape):
    if not self.built:
      self.build(input_shape=input_shape)
    self.load_weights(self.pretrained_weights_path)
    print('vgg_face weights loaded!')
  
  def call(self, x, training=False):
    """TBD."""
    y = x
    fmaps = []
    for layer in self.net_layers:
      y = layer(y)
      fmaps.append(y)
    return y, fmaps

