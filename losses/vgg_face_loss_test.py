from PIL import Image
import os
import numpy as np
# import losses.vgg_face_loss
import vgg_face_loss


parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/two_step_synthesis-old/drive_data_from_the_neural_talking_heads_paper/processed_voxceleb_data-2D/test'
img1_path = os.path.join(parent_dir, 'id00017/OLguY5ofUrY/00044/00005_rgb.png')
img2_path = os.path.join(parent_dir, 'id00017/OLguY5ofUrY/00044/00130_rgb.png')
img3_path = os.path.join(parent_dir, 'id00812/pLjziqnar8U/00341/00059_rgb.png')

def load_and_maybe_normalize_image(path, normalize=True):
  img = np.array(Image.open(path)).astype(np.float32)
  img = img / 255. * 2 - 1
  img = img[np.newaxis, ]
  print(img.shape, img.dtype)
  return img

pretrained_weights_path='/fs/vulcan-projects/two_step_synthesis_meshry/third_party/pretrained_weights/vgg_face_weights.h5'
img1 = load_and_maybe_normalize_image(img1_path)
img2 = load_and_maybe_normalize_image(img2_path)
img3 = load_and_maybe_normalize_image(img3_path)
# Image.fromarray(np.squeeze(((img1 + 2)/2.*255)).astype(np.uint8)).show(title='img1')
# Image.fromarray(np.squeeze(((img2 + 2)/2.*255)).astype(np.uint8)).show(title='img2')
# Image.fromarray(np.squeeze(((img3 + 2)/2.*255)).astype(np.uint8)).show(title='img3')

layer_idxs = (1,6,11,18,25)
layer_weights = tuple(0.005 * np.array((1,1,1,1,1)))
vgg_face_loss = vgg_face_loss.VGGFaceLoss()
input_shape = (None,) + img1.shape[-3:]
vgg_face_loss.load_pretrained_weights(input_shape=input_shape)
model = vgg_face_loss.model
# # Load VGG Face model weights
# model.build(img1.shape)
# model.load_weights(pretrained_weights_path)
# print('vgg_face weights loaded!')
loss1 = vgg_face_loss(img1, img1, layer_idxs=layer_idxs, layer_weights=layer_weights)
loss2 = vgg_face_loss(img1, img2, layer_idxs=layer_idxs, layer_weights=layer_weights)
loss3 = vgg_face_loss(img1, img3, layer_idxs=layer_idxs, layer_weights=layer_weights)
print(f'loss_face_recon_1 = {loss1}')
print(f'loss_face_recon_2 = {loss2}')
print(f'loss_face_recon_3 = {loss3}')

layer_idxs = (-2,)
layer_weights = tuple(1*np.array((1,)))
loss_id1 = vgg_face_loss(img1, img1, layer_idxs=layer_idxs, layer_weights=layer_weights)
loss_id2 = vgg_face_loss(img1, img2, layer_idxs=layer_idxs, layer_weights=layer_weights)
loss_id3 = vgg_face_loss(img1, img3, layer_idxs=layer_idxs, layer_weights=layer_weights)
print(f'loss_id_1 = {loss_id1}')
print(f'loss_id_2 = {loss_id2}')
print(f'loss_id_3 = {loss_id3}')
