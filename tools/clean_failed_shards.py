import numpy as np
import os
import os.path as osp
import glob

# status = np.zeros(999, dtype=bool)
status = np.load('status.npy')
print('Number of successfully written shards = %d' % np.sum(status))
parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2'
all_files = sorted(glob.glob(osp.join(parent_dir, 'tfrecords', 'vox2_dev*.tfrecords')))
print(len(all_files))
rm_cnt = 0
kept_cnt = 0
for i, fpath in enumerate(all_files):
  toks = osp.basename(fpath).split('-')  # vox2_dev-part-0998-of-0999
  shard_idx = int(toks[2])
  if status[shard_idx]:
    print('Keeping shard %d.' % shard_idx)
    kept_cnt += 1
  else:
    print('Removing shard %d.' % shard_idx)
    rm_cnt += 1
    os.remove(fpath)
print('Total kept = %d' % kept_cnt)
print('Total removed = %d' % rm_cnt)
np.save('status.npy', status)
print('Done!')
