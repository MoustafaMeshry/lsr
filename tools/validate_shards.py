from absl import app
from absl import flags
import glob
import numpy as np
import os
import os.path as osp


def main(argv):
  logs_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/code/two_step_synthesis/validating_shards'
  # logs_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/code/two_step_synthesis/data'
  logs = sorted(glob.glob(osp.join(logs_dir, '*.log')))

  num_shards = 999
  status = np.zeros(num_shards, dtype=bool)
  corrupted = 0
  duplicates = 0
  for i_log, log_path in enumerate(logs):
    print('Processing log file #%02d/%02d: %s.' % (i_log, len(logs), log_path))
    with open(log_path) as f:
      lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = lines[3:-2]
    for i_line, line in enumerate(lines):
      toks = line.split('part-')
      shard_idx = int(toks[1][:4])
      if 'ERROR' in line:
        print(f'ERROR: shard #{shard_idx} needs to be recomputed!')
        corrupted += 1
      elif status[shard_idx]:
        print(f'ERROR: shard #{shard_idx} is duplicated!')
        duplicates += 1
      else:
        status[shard_idx] = True

  print('Done!')
  print('Number of corrupted shards = %d' % corrupted)
  print('Number of duplicated shards = %d' % duplicates)
  print('Number of validated shards = %d/%d' % (np.sum(status), num_shards))
  missing_shards = np.where(np.any(status[..., np.newaxis] == False, axis=1))[0]
  print('There is a total of %d missing shards listed below:' % len(missing_shards))
  for idx in missing_shards:
    print(idx)


if __name__ == '__main__':
  app.run(main)
