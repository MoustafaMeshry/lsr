# python release_landmark.py --data_dir /fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_test_videos --output_dir /fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_test_videos_processed --mode videos
import torch
import os
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm
import face_alignment
from matplotlib import pyplot as plt
import time
import argparse
import os
import torch

class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		parser.add_argument('--data_dir', type=str, \
			default="/fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_test_videos", \
			help='data directory path containing person id based folders')
		parser.add_argument('--output_dir', type=str, \
			default="/fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_test_videos_processed", help='output directory path')
		parser.add_argument('--k', type=int, \
			default=10, help='sampling rate for frames from video')
		parser.add_argument('--mode', type=str, \
			default="videos", help='images | videos')
		self.initialized = True
		return parser

	def gather_options(self):
		# initialize parser with basic options
		if not self.initialized:
			parser = argparse.ArgumentParser(
				formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		# get the basic options
		opt, _ = parser.parse_known_args()
		self.parser = parser

		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------'
		print(message)


	def parse(self, print_options=True):

		opt = self.gather_options()
		# process opt.suffix
		if print_options:
			self.print_options(opt)
		
		self.opt = opt
		return self.opt



def generate_landmarks(frames_list, face_aligner, path):
	frame_landmark_list = []
	fa = face_aligner
	fail=0
	for i in range(len(frames_list)):
		try:
			input = frames_list[i]
			preds = fa.get_landmarks(input)[0]

			dpi = 100
			fig = plt.figure(figsize=(input.shape[1]/dpi, input.shape[0]/dpi), dpi = dpi)
			ax = fig.add_subplot(1,1,1)
			ax.imshow(np.zeros(input.shape))
			plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

			#chin
			ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
			#left and right eyebrow
			ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
			ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
			#nose
			ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
			ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
			#left and right eye
			ax.plot(preds[36:42,0],preds[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
			ax.plot(preds[42:48,0],preds[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
			#outer and inner lip
			ax.plot(preds[48:60,0],preds[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
			ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
			ax.axis('off')

			fig.canvas.draw()

			data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
			data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

			frame_landmark_list.append((input, data))
			plt.close(fig)
		except:
			fail=1
			print('Error: Video corrupted or no landmarks visible')
	
	for i in range(len(frames_list) - len(frame_landmark_list)):
		#filling frame_landmark_list in case of error
		frame_landmark_list.append(frame_landmark_list[i])
	if(fail==1):
		open('./sanity_fails/' + path, 'a').close()
	
	return frame_landmark_list

def pick_images_from_videos(video_path, num_images):
	cap = cv2.VideoCapture(video_path)
	
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	idxes = [1 if i%(n_frames//num_images+1)==0 else 0 for i in range(n_frames)]
	
	frames_list = []
	
	# Read until video is completed or no frames needed
	ret = True
	frame_idx = 0
	frame_counter = 0
	frame_ind = []
	while(ret and frame_idx < n_frames):
		ret, frame = cap.read()
		
		if ret and idxes[frame_idx] == 1:
			RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frames_list.append(RGB)
			frame_ind.append(frame_idx)
		frame_idx += 1

	cap.release()
	return frames_list, frame_ind

def pick_images_from_images(path):
	frame_ind = []
	frames_list = []
	for frame_idx, image in enumerate(sorted(os.listdir(path))):
		RGB = cv2.cvtColor(cv2.imread(os.path.join(path, image)), cv2.COLOR_BGR2RGB)
		frames_list.append(RGB)
		frame_ind.append(frame_idx)
	return frames_list, frame_ind


if __name__ == '__main__':
	args = BaseOptions().parse()
	K = args.k
	device = torch.device('cuda:0')
	face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')

	path_to_mp4 = args.data_dir
	path_to_preprocess = args.output_dir
	#verify structure of data I removed the train and test disr for iamges need to give path
	# to dir containing folder for person

	for person_id in tqdm(os.listdir(path_to_mp4)):
		for video_id in os.listdir(os.path.join(path_to_mp4, person_id)):
			for video in os.listdir(os.path.join(path_to_mp4, person_id, video_id)):
				save_path = os.path.join(path_to_preprocess, person_id, video_id, video.split(".")[0])
				try:
					video_path = os.path.join(path_to_mp4, person_id, video_id, video)
					if args.mode=='videos':
						frame_mark, frame_ind = pick_images_from_videos(video_path, K)
					else:
						frame_mark, frame_ind = pick_images_from_images(video_path)
					frame_mark = generate_landmarks(frame_mark, face_aligner, person_id+'_'+video_id+'_'+video.split(".")[0])
					if ((len(frame_mark) == K) or args.mode=='images'): #verify this
						os.makedirs(save_path, exist_ok=True)
						for i in range(len(frame_ind)):
							final_list = np.array(frame_mark[i][1])
							final_list = cv2.cvtColor(final_list, cv2.COLOR_BGR2RGB)
							cv2.imwrite(save_path +"/" +format(frame_ind[i], '05d') + "_contour.png", final_list)
							final_list = np.array(frame_mark[i][0])
							final_list = cv2.cvtColor(final_list, cv2.COLOR_BGR2RGB)
							cv2.imwrite(save_path +"/" +format(frame_ind[i], '05d') + "_rgb.png", final_list)
				except:
					print('ERROR: ', video_path)
	print('Processed.')
