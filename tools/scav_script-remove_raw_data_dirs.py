# Run command: python tools/scav_script-remove_raw_data_dirs.py --scav --cores 4 --mem 10 --gpu=0

from datetime import datetime
import argparse
import glob
import numpy as np
import os
import os.path as osp
import socket
import time


# Function to check for validity of QOS
# TODO: add time/duration check.
def check_qos(args):
  qos_dict = {'high' : {'gpu':4, 'cores': 16, 'mem':128},
              'medium' : {'gpu':2, 'cores': 8, 'mem':64},
              'default' : {'gpu':1, 'cores': 4, 'mem':32}}
  for key, max_value in qos_dict[args.qos].items():
    val_from_args = getattr(args, key)
    if val_from_args != None:
      if val_from_args > max_value:
        raise ValueError('Invalid paramter for {} for {}'.format(key, args.qos))
    else:
      setattr(args, key, max_value)
      print('Setting {} to max value of {} in QOS {} as not specified in arguements.'.format(key, max_value, args.qos))
  return args


#TODO: Add day funtionality too 
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=72)
parser.add_argument('--base-dir', default=f'{os.getcwd()}/logs')
parser.add_argument('--output_dirname', default='rm_output')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--scav', action='store_true')
parser.add_argument('--qos', default=None, type=str, help='Qos to run')
parser.add_argument('--env', type=str, help = 'Set the name of the dir you want to dump')
parser.add_argument('--gpu', default=None, type=int, help='Number of gpus')
parser.add_argument('--cores', default=None, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=None, type=int, help='RAM in G')
args = parser.parse_args()


output_dir = os.path.join(args.base_dir, args.output_dirname)
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
print('Output Directory: %s' % output_dir)


# Setting the paramters for the scripts to run, modify as per your need
num_subjects = 5994
parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/preprocessed_voxceleb2/dev'
dirs = sorted(glob.glob(osp.join(parent_dir, '*')))
print('Removing %d directories at %s ...' % (len(dirs), parent_dir))

params = []
for i_dir, dir_path in enumerate(dirs):
  params.append((i_dir, dir_path))

chunk_size = 500
num_chunks = len(dirs) // chunk_size + (1 if len(dirs) % chunk_size else 0)
print('Number of directories = %d\nChunk size = %d\nNumber of chunks = %d' % (
    len(dirs), chunk_size, num_chunks))
chunk_idx = 3  # 2, 1, 0
# params = params[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
# params = params[:1000]
print(params[:5])
# assert len(dirs) == num_subjects

num_commands = len(params)
#######################################################################


# Making text files which will store the python command to run, stdout, and error if any  
with open(f'{args.base_dir}/{args.output_dirname}/now.txt', 'w') as nowfile,\
	 open(f'{args.base_dir}/{args.output_dirname}/log.txt', 'w') as output_namefile,\
	 open(f'{args.base_dir}/{args.output_dirname}/err.txt', 'w') as error_namefile,\
	 open(f'{args.base_dir}/{args.output_dirname}/name.txt', 'w') as namefile:

	# Iterate over all hyper parameters
  for i_dir, dir_path in params:
    now = datetime.now()
    datetimestr = now.strftime('%m%d_%H%M:%S.%f')
    basename = osp.basename(dir_path)
    name = '%05d-%s' % (i_dir, basename)
    cmd  = f'rm -rf {dir_path}'
    nowfile.write(f'{cmd}\n')
    namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
    output_namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
    error_namefile.write(f'{(os.path.join(output_dir, name))}.error\n')
###########################################################################


# Make a {name}.slurm file in the {output_dir} which defines this job.
#slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
start=1
slurm_script_path = os.path.join(output_dir, f'rm_vox_{chunk_idx}.slurm')
slurm_command = 'sbatch %s' % slurm_script_path

# Make the .slurm file
with open(slurm_script_path, 'w') as slurmfile:
  slurmfile.write('#!/bin/bash\n')
  slurmfile.write(f'#SBATCH --array=1-{num_commands}\n') #parallelize across commands.
  slurmfile.write('#SBATCH --output=/dev/null\n')
  slurmfile.write('#SBATCH --error=/dev/null\n')
  slurmfile.write('#SBATCH --requeue\n') #fuck. Restart the job 

  #slurmfile.write('#SBATCH --cpus-per-task=16\n')
  if args.scav:
    slurmfile.write('#SBATCH --account=scavenger\n')
    slurmfile.write('#SBATCH --partition scavenger\n')
    slurmfile.write('#SBATCH --qos=scavenger\n')
    slurmfile.write('#SBATCH --time=%d:00:00\n' % args.nhrs)
    slurmfile.write('#SBATCH --gres=gpu:%d\n' % args.gpu)
    slurmfile.write('#SBATCH --cpus-per-task=%d\n' % args.cores)
    slurmfile.write('#SBATCH --mem=%dG\n' % args.mem)
    # slurmfile.write('#SBATCH --exclude=vulcan[22]\n')
    if args.gpu is None or args.gpu == 0:
      slurmfile.write('#SBATCH --exclude=vulcan[00-23]\n')
  else:
    args = check_qos(args)
    slurmfile.write('#SBATCH --qos=%s\n' % args.qos)
    slurmfile.write('#SBATCH --time=%d:00:00\n' % args.nhrs)
    slurmfile.write('#SBATCH --gres=gpu:%d\n' % args.gpu)
    slurmfile.write('#SBATCH --cpus-per-task=%d\n' % args.cores)
    slurmfile.write('#SBATCH --mem=%dG\n' % args.mem)

  slurmfile.write('\n')
  slurmfile.write('cd ' + args.base_dir + '\n')
  # slurmfile.write('eval \'$(conda shell.bash hook)\'' '\n')

  # slurmfile.write('source setup_env.sh \n')
  slurmfile.write(f'srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/{args.output_dirname}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/{args.output_dirname}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/{args.output_dirname}/now.txt | tail -n 1)\n')

  slurmfile.write('\n')
print(slurm_command)
print('Running on {}, with {} gpus, {} cores, {} mem for {} hour'.format(args.qos, args.gpu, args.cores, args.mem , args.nhrs))
if not args.dryrun:
   os.system('%s &' % slurm_command)
