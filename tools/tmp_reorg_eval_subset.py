from datetime import datetime
from PIL import Image
import argparse
import glob
import numpy as np
import os
import os.path as osp
import socket
import shutil
import time


# Rename *.jpg to *_rgb.png
def convert_jpg_to_png_and_rename(subset='train'):
  parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/eval_subset'
  parent_dir = osp.join(parent_dir, subset)
  
  person_dirs = sorted(glob.glob(osp.join(parent_dir, '*')))
  print('Processing: %s.' % parent_dir)
  print('Number of subject dirs = %d.' % len(person_dirs))
  for person_idx, person_dir in enumerate(person_dirs):
    print(f'Processing person #{person_idx:02}/{len(person_dirs):02}: {person_dir}.')
    video_dirs = sorted(glob.glob(osp.join(person_dir, '*')))
    for video_idx, video_dir in enumerate(video_dirs):
      video_part_dirs = sorted(glob.glob(osp.join(video_dir, '*')))
      for video_part_idx, video_part_dir in enumerate(video_part_dirs):
        img_paths = sorted(glob.glob(osp.join(video_part_dir, '*.jpg')))
        for img_idx, img_path in enumerate(img_paths):
          basename = osp.splitext(osp.basename(img_path))[0]
          save_name = f'{basename}_rgb.png'
          save_path = osp.join(video_part_dir, save_name)
          Image.open(img_path).save(save_path, 'png')


def rename_dense_frames():
  parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/eval_subset/test_dense'
  
  person_dirs = sorted(glob.glob(osp.join(parent_dir, '*')))
  print('Processing: %s.' % parent_dir)
  print('Number of subject dirs = %d.' % len(person_dirs))
  for person_idx, person_dir in enumerate(person_dirs):
    print(f'Processing person #{person_idx:02}/{len(person_dirs):02}: {person_dir}.')
    video_dirs = sorted(glob.glob(osp.join(person_dir, '*')))
    for video_idx, video_dir in enumerate(video_dirs):
      video_part_dirs = sorted(glob.glob(osp.join(video_dir, '*')))
      for video_part_idx, video_part_dir in enumerate(video_part_dirs):
        img_paths = sorted(glob.glob(osp.join(video_part_dir, '*.png')))
        for img_idx, img_path in enumerate(img_paths):
          basename = osp.splitext(osp.basename(img_path))[0]
          toks = basename.split('_')
          if 'tommy' in video_part_dir:
            frame_idx = img_idx
          else:
            frame_idx = int(toks[-2])
          save_name = f'{frame_idx:05}_rgb.png'
          save_path = osp.join(video_part_dir, save_name)
          # print(f'Renaming:\n{img_path}\n{save_path}')
          # return
          shutil.move(img_path, save_path)


# convert_jpg_to_png_and_rename(subset='test')
rename_dense_frames()
