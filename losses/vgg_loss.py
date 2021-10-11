import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input

class VGGLoss(object):
  """A VGG losss class."""

  def __init__(self,
               weights='imagenet',
               layers=('block1_conv1', 'block2_conv1', 'block3_conv1',
                       'block4_conv1', 'block5_conv1'),
               layer_weights=(1. / 32, 1. / 16, 1. / 8, 1. / 4, 1.0)):
    """Initializes the VGG network."""
    self._vgg = tf.keras.applications.VGG19(include_top=False, weights=weights)
    self._vgg.trainable = False
    self._layers = layers
    self._weights = layer_weights
    output_names = layers
    outputs = [self._vgg.get_layer(name).output for name in output_names]
    self._model = tf.keras.Model([self._vgg.input], outputs)

  def __call__(self,
               y_true,
               y_pred,
               axis=None,
               name='vgg_loss',
               summaries=False,
               loss_family='vgg_loss'):
    """Computes the VGG loss."""
    # Preprocess expects RGB inputs in range [0, 255].
    all_true_features = self._model(
        tf.keras.applications.vgg19.preprocess_input((1 + y_true) / 2. * 255.))
    all_pred_features = self._model(
        tf.keras.applications.vgg19.preprocess_input((1 + y_pred) / 2. * 255.))

    # loss = tf.constant(0., dtype=tf.float32)
    loss = 0.
    for true_features, pred_features, layer_name, weight in zip(
        all_true_features, all_pred_features, self._layers, self._weights):
      # layer_loss = weight * tf.reduce_mean(
      #     tf.reduce_sum(tf.math.abs(true_features - pred_features), axis=0))
      layer_loss = weight * tf.reduce_mean(
          tf.math.abs(true_features - pred_features), axis=axis)
      if summaries:
        tf.summary.scalar(
            '%s/%s' % (loss_family, layer_name),
            tf.reduce_mean(layer_loss))
      loss += layer_loss
      # tf.summary.scalar('%s/%s' % (loss_family, name), tf.reduce_mean(loss))

    return loss



