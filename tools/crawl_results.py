from absl import app
from absl import flags
import glob
import numpy as np
import os
import os.path as osp
import re


def crawl_results(parent_dir, results_filename='quantitative_metrics.txt'):
  train_dirs = sorted(glob.glob(osp.join(parent_dir, '*')))
  print(f'Processing {len(train_dirs)} experiments!\n')
  for dir_idx, train_dir in enumerate(train_dirs):
    results_file_path = osp.join(train_dir, results_filename)
    if not osp.exists(results_file_path):
      print(f'`{osp.basename(train_dir)}` has no `{results_filename}`!')
      print('-------------------------------------------\n')
      continue
    with open(results_file_path, 'r') as f:
      lines = [line.strip() for line in f.readlines()]
    metrics, vals = '', ''
    for line in lines:
      if not len(line):
        continue
      # print(line)
      if line.startswith('Evaluation') or line.startswith('Title'):
        title = line.split(' ')[-1]
        if title[-1] == ':':
          title = title[:-1]
        # print(dir_idx)
        # print(title)
      elif line.startswith('Description') or line.startswith('Descrption'):
        description = line[len('Description: '):]
        # print(description)
      else:
        toks = re.split('[: ]', line)
        if not metrics.endswith('fid,'):
          metrics += toks[0] + ','
        vals += toks[-1] + ','
        if toks[0] == 'fid':
          # metrics += '\n'
          vals += '\n'
    print(dir_idx)
    print(title)
    print(description)
    print(metrics)
    print(vals)
    print('-------------------------------------------\n')


def main(argv):
  # parent_dir = '/vulcan/scratch/mmeshry/SPADE/train'
  # parent_dir = '/vulcanscratch/mmeshry/SPADE/train'
  # parent_dir = '/fs/vulcan-projects/gan_residuals_meshry/train'
  parent_dir = '/fs/vulcan-projects/gan_residuals_meshry/train_v3'
  crawl_results(parent_dir)


if __name__ == '__main__':
  app.run(main)
