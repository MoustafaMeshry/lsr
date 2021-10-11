from absl import app
from absl import flags
import glob
import numpy as np
import os
import os.path as osp
import re
import shutil


def get_latest_checkpoint(train_dir):
  checkpoint_filepath = osp.join(train_dir, 'checkpoint')
  assert osp.exists(
      checkpoint_filepath), 'No such file `%s`!' % checkpoint_filepath
  with open(checkpoint_filepath, 'r') as f:
    lines = [line.strip() for line in f.readlines()]
  st_idx = lines[0].find('"')
  end_idx = lines[0].find('"', st_idx+1)
  assert end_idx == len(lines[0]) - 1
  latest_ckpt = lines[0][st_idx+1 : end_idx]
  assert st_idx > 0 and end_idx > 0 and len(latest_ckpt)
  return osp.basename(latest_ckpt)

def remove_training_checkpoints(parent_dir, simulate_only=False):
  train_dirs = sorted(glob.glob(osp.join(parent_dir, '*')))
  print(f'Processing {len(train_dirs)} experiments!\n')
  for dir_idx, train_dir in enumerate(train_dirs):
    print(f'Processing file {dir_idx}/{len(train_dirs)}: {train_dir}')
    latest_ckpt = get_latest_checkpoint(train_dir)
    toks = latest_ckpt.split('-')
    ckpt_number = toks[-1]
    ckpt_prefix = latest_ckpt[:-len(ckpt_number) - 1]
    print('DBG: ', latest_ckpt, ckpt_prefix, ckpt_number)
    assert (f'{ckpt_prefix}-{ckpt_number}' == latest_ckpt and
            ckpt_number.isnumeric)
    filepaths = sorted(glob.glob(osp.join(train_dir, ckpt_prefix + '*')))
    for path in filepaths:
      basename = osp.basename(path)
      if simulate_only:
        if not basename.startswith(latest_ckpt):
          print('Removing %s!' % path)
        else:
          print('Keeping %s!' % path)
      else:
        os.remove(path)


def main(argv):
  # parent_dir = '/vulcan/scratch/mmeshry/SPADE/train'
  # parent_dir = '/fs/vulcan-scratch/mmeshry/appearance_pretraining/train/edges2handbags/custom'
  # parent_dir = '/fs/vulcan-projects/gan_residuals_meshry/train'
  # parent_dir = '/fs/vulcan-projects/gan_residuals_meshry/spade_baseline/train_old_versions/SPADE/train_old'
  parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/SPADE/train'
  remove_training_checkpoints(parent_dir, simulate_only=False)


if __name__ == '__main__':
  app.run(main)
