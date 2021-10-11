from PIL import Image
from skimage.metrics import structural_similarity as ssim
import argparse
import functools
import glob
import losses.losses as losses
import numpy as np
import os
import os.path as osp
import skimage.measure
import tensorflow as tf
import tensorflow_gan as tfgan


parser = argparse.ArgumentParser()
# Evaluation flags
parser.add_argument('--run_mode', type=str, default='eval', help='One of '
                    '{`eval`, `subject_finetuning`} (defaults to `eval`).')
parser.add_argument('--exp_train_dir', type=str, required=True, help='Path to '
                    'the train dir of the target experiment to evaluate.')
parser.add_argument('--K', type=int, default=1,
                    help='Number of input few shots (defaults to 1).')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size per gpu (defaults to 4).')
parser.add_argument('--subject_pattern', type=str, default='*',
                    help='subject pattern for fine-tuning (defaults to `*`).')
parser.add_argument('--target_subject_id', type=str, default=None,
                    help='target subject id for cross-subject reenactment.')
parser.add_argument('--target_video_id', type=str, default=None,
                    help='target video id for cross-subject reenactment.')
parser.add_argument('--max_finetune_steps', type=int, default=None,
                    help='Number of fine-tune steps.')
parser.add_argument('--save_summaries_every_n_steps', type=int, default=-1,
                    help='Save tensorboard summaries during fine-tuning every'
                         'n steps. Set to np.inf to disable (defaults to 10)')
parser.add_argument('--evaluate_every_n_steps', type=int, default=-1,
                    help='periodic evaluation during subject fine-tuning'
                         'every n steps. Set to -1 to disable (default=-1).')
parser.add_argument('--decay_lr', type=bool, default=True, help='Whether or not'
                    'to decay learning rates during subject fine-tuning.')
parser.add_argument('--save_subject_checkpoint', action='store_true',
                    help='Wehther or not to save the fine-tuned checkpoint '
                    'for target subjects.')
parser.add_argument('--evaluate_dense', action='store_true', help='Evaluate '
                    'the dense subset after subjec fine-tuning.')
parser.add_argument('--d_lr', type=float, default=None,
                    help='Overwites the discriminator learning rate for '
                         'subject fine-tuning.')
parser.add_argument('--g_lr', type=float, default=None,
                    help='Overwites the generator learning rate for subject.'
                         'fine-tuning.')
parser.add_argument('--w_loss_l1', type=float, default=None,
                    help='Overwites the l1 loss weight for subject '
                         'fine-tuning.')
parser.add_argument('--w_loss_vgg', type=float, default=None,
                    help='Overwites the vgg loss weight for subject '
                         'fine-tuning.')
parser.add_argument('--w_loss_vgg_face_recon', type=float, default=None,
                    help='Overwites the vgg_face_recon loss weight for '
                         'subject fine-tuning.')
parser.add_argument('--w_loss_identity', type=float, default=None,
                    help='Overwites the identity loss weight for subject '
                         'fine-tuning.')
parser.add_argument('--w_loss_segmentation', type=float, default=None,
                    help='Overwites the l1 loss weight for subject '
                         'fine-tuning.')
# Slurm flags
parser.add_argument('--srun', action='store_true')
parser.add_argument('--scav', action='store_true')
parser.add_argument('--qos', default=None, type=str, help='Qos to run')
parser.add_argument('--ngpus', default=None, type=int, help='Number of gpus')
parser.add_argument('--gpu_type', default=None, type=str, help='GPU type (e.g. '
                    'one of {p6000, gtx1080ti, p100, rtx2080ti}')
parser.add_argument('--cores', default=None, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=None, type=int, help='RAM in G')
parser.add_argument('--nhrs', type=int, default=36)

args = parser.parse_args()


def parse_config(config_path):
  print(f'Parsing config file: {config_path}.')
  assert osp.exists(config_path)
  with open(config_path) as f:
    lines = f.readlines()
  lines = [l.strip() for l in lines]
  config = dict()
  for line in lines:
    # print(f'DBG: Parsing: {line}')
    if line[0] == '#':
      continue
    split_idx = line.find('=')
    key = line[:split_idx]
    val = line[split_idx+1:]
    config[key] = val
  return config


def prepare_srun_resources(args):
  assert args.srun 
  srun_cmd = 'srun'
  if args.ngpus:
    if args.gpu_type is not None:
      assert args.gpu_type in ['gtx1080ti', 'p6000', 'p100', 'rtx2080ti'], (
          f'Invalid --gpu_type value of {args.gpu_type}')
      srun_cmd += f' --gres=gpu:{args.gpu_type}:{args.ngpus}'
    else:
      srun_cmd += f' --gres=gpu:{args.ngpus}'
  else:
    srun_cmd += ' --exclude=vulcan[00-23]'

  if args.scav:
    srun_cmd += ' --account=scavenger'
    srun_cmd += ' --partition=scavenger'
    srun_cmd += ' --qos=scavenger'
  elif args.qos:
    srun_cmd += f' --qos={args.qos}'

  if args.nhrs:
    srun_cmd += f' --time={args.nhrs:02}:00:00'
  if args.mem:
    srun_cmd += f' --mem={args.mem}G'
  if args.cores:
    srun_cmd += f' --cpus-per-task={args.cores}'
  return srun_cmd


def main():
                      
  code_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/code/two_step_synthesis'

  # Parse experiment configuration (run_config.txt).
  config_path = osp.join(args.exp_train_dir, 'run_config.txt')
  config = parse_config(config_path)

  # Set/overwrite evaluation config.
  slurm_flags = [
      'srun',
      'scav',
      'qos',
      'ngpus',
      'gpu_type',
      'cores',
      'mem',
      'nhrs',
  ]
  for arg in vars(args):
    if arg == 'exp_train_dir' or arg in slurm_flags:
      continue
    val = getattr(args, arg)
    if val is not None:
      old_val = f'{config[arg]}' if arg in config else 'N/A (unset)'
      print(f'Overwriting --{arg} from {old_val} to {val}.')
      config[arg] = val

  # Prepare bash command.
  cmd = f'python {code_dir}/main.py'
  for k, v in config.items():
    cmd += f' \\\n  --{k}={v}'
  cmd += ' \\\n  --alsologtostderr'

  print('Command to run is:')
  print(cmd)

  # Wrap with slurm srun (if needed). Otherwise, excuite command using
  #  os.system(cmd)
  if args.srun:
    srun_cmd = prepare_srun_resources(args)
    srun_cmd += ' bash -c "$( cat << EOF\n'
    srun_cmd += f'{cmd}\n'
    srun_cmd += 'EOF\n'
    srun_cmd += ')"'
    os.system(srun_cmd)
  else:
    os.system(cmd)

  # Run evaluation.
  print('Done!')


if __name__ == '__main__':
  main()
