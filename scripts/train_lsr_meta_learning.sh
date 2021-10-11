parent_train_dir=./_train

experiment_title=lsr_meta_learning
experiment_description="Meta learning."

model_type='two_step_syn'  # One of {pretrain_layout, two_step_syn}
run_mode='train'  # One of {train, eval, infer}
warmup_checkpoint='_train/lsr_pretraining'
K=4  # Number of K-shot inputs during training. At test time K can take any value.
num_frame_concatenations=10  # Number of total frames in each training example.
trainset_size=1090194  # Total size of the processed training set.
total_k_examples=16350  # Total number of training examples (in kilo). Number of train epochs is total_k_examples * 1000 / trainset_size
num_lr_decay_k_examples=1090  # Number of training examples (in kilo) to apply a linear learning rate decay.
batch_size=4
dataset_name='voxceleb2'
dataset_parent_dir='_datasets/preprocessed_voxceleb2/tfrecords'  # Path to the direcotry containing the tfrecord dataset.
trainset_pattern='vox2_dev*.tfrecords'  # Pattern for the train subset of the tfrecords.
evalset_pattern='vox2_test*.tfrecords'  # Pattern for the test subset of the tfrecords.
train_dir="$parent_train_dir/$experiment_title"  # Checkpoints, tensorboard summaries and model output will be saved in this path.

# Loss weights for meta learning.
w_loss_gan=1
w_loss_vgg=0.1
w_loss_l1=1
w_loss_segmentation=0
w_loss_vgg_face_recon=0.002
w_loss_identity=1  # For subject fine-tuning try setting this to 20!
w_loss_z_l2=1
w_loss_z_layout_l2=1

python main.py \
  --model_type=$model_type \
  --run_mode=$run_mode \
  --warmup_checkpoint=$warmup_checkpoint \
  --K=$K \
  --num_frame_concatenations=$num_frame_concatenations \
  --train_dir=$train_dir \
  --trainset_pattern=$trainset_pattern \
  --evalset_pattern=$evalset_pattern \
  --experiment_description="$experiment_description" \
  --dataset_name=$dataset_name \
  --dataset_parent_dir=$dataset_parent_dir \
  --trainset_size=$trainset_size \
  --batch_size=$batch_size \
  --total_k_examples=$total_k_examples \
  --num_lr_decay_k_examples=$num_lr_decay_k_examples \
  --w_loss_gan=$w_loss_gan \
  --w_loss_vgg=$w_loss_vgg \
  --w_loss_segmentation=$w_loss_segmentation \
  --w_loss_vgg_face_recon=$w_loss_vgg_face_recon \
  --w_loss_identity=$w_loss_identity \
  --w_loss_l1=$w_loss_l1 \
  --w_loss_z_l2=$w_loss_z_l2 \
  --w_loss_z_layout_l2=$w_loss_z_layout_l2 \
  --alsologtostderr
