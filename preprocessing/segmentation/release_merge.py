# python release_merge.py --path /fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_test_videos_processed/
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import cm
import cv2
import os.path
import time
from tqdm import tqdm
import numpy as np

from torchvision.utils import save_image
from PIL import Image
import argparse

def get_params():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default='/fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_fsth_eval_subset_processed')
	return parser.parse_args()


def merge_segmentations(dir_path):
	label_list = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
	take_big = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hat', 'ear_r', 'neck_l']

	for person_id in tqdm(os.listdir(dir_path)):
		for video_id in os.listdir(os.path.join(dir_path, person_id)):
			for video in os.listdir(os.path.join(dir_path, person_id, video_id)):
				for img in os.listdir(os.path.join(dir_path, person_id, video_id, video)):
					if('rgb' in img):
						image_1 = os.path.join(dir_path, person_id, video_id,video,img[:5]+ '_plain_segmap_' + str(256) + '.png')
						image_2 = os.path.join(dir_path, person_id, video_id,video,img[:5]+ '_plain_segmap_' + str(512) + '.png')
						a = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE) #256
						b = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE) #512
						orig_a = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE) #256
						orig_b = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE) #512
						b = cv2.resize(b, (256, 256), interpolation = cv2.INTER_NEAREST)

						base_img = np.zeros(b.shape)
						base_img[a==13] = a[a==13] #hair
						
						for label in np.unique(b):
							if(label_list[label] in take_big):
								base_img[b==label] = b[b==label]
						base_img[a==17] = a[a==17] #neck
						base_img[a==18] = a[a==18] #cloth

						labels_predict_plain = cv2.resize(base_img, (224, 224), interpolation = cv2.INTER_NEAREST)
						# labels_predict_color = generate_label(labels_predict_plain, 224)
						cv2.imwrite(os.path.join(dir_path, person_id, video_id,video,img[:5]+ '_segmap_merged.png'), labels_predict_plain)
						cv2.imwrite(os.path.join(dir_path, person_id, video_id,video,img[:5]+ '_segmap_256.png'), cv2.resize(orig_a, (224, 224), interpolation = cv2.INTER_NEAREST))
						cv2.imwrite(os.path.join(dir_path, person_id, video_id,video,img[:5]+ '_segmap_512.png'), cv2.resize(orig_b, (224, 224), interpolation = cv2.INTER_NEAREST))
						if(os.path.exists(image_1)):
							os.remove(image_1)
						if(os.path.exists(image_2)):
							os.remove(image_2)
	print('Processed')

if __name__ == '__main__':
  config = get_params()
  merge_segmentations(config.path)
