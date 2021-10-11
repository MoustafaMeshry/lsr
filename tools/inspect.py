import numpy as np
import os
import os.path as osp
import glob

# status = np.zeros(999, dtype=bool)
status = np.load('status.npy')
print(np.sum(status))
assert False
success_count = 0
parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/code/two_step_synthesis'
all_files = sorted(glob.glob(osp.join(parent_dir, 'logs', 'output/shard*.log')))
print(len(all_files))
for i, fpath in enumerate(all_files):
  toks = osp.basename(fpath).split('-')
  shard_idx = int(toks[0][6:])
  print('Processing file %03d: %03d' % (i, shard_idx))
  with open(fpath) as f:
    lines = f.readlines()
  line = lines[-1].strip()
  if line.startswith('Success'):
    success_count += 1
    print('shard %03d is a success!' % shard_idx)
    status[shard_idx] = True
print('Total success = %d' % success_count)
np.save('status.npy', status)
print('Done!')
