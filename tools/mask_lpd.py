from PIL import Image
import os
import os.path as osp
import numpy as np
import glob

def mask_lpd():
  images_parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/results/GT_LPD/preprocess_sanity/test-images-cropped'
  masks_parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/results/GT_LPD/preprocess_sanity/test-segmentation-cropped'
  out_parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/results/GT_LPD/preprocess_sanity/test-images-masked'
  
  
  person_ids = sorted(os.listdir(images_parent_dir))
  person_ids = [p for p in person_ids if p.startswith('id')]
  for pidx, pid in enumerate(person_ids):
    v_ids = os.listdir(osp.join(images_parent_dir, pid))
    print(f'Processing person #{pidx:2d}: {pid}: found {len(v_ids)} videos.')
    for vid in v_ids:
      vpart_ids = os.listdir(osp.join(images_parent_dir, pid, vid))
      print(f'  > vid={vid}: found {len(vpart_ids)} video parts.')
      for vpart_id in vpart_ids:
        img_paths = sorted(glob.glob(
            osp.join(images_parent_dir, pid, vid, vpart_id, '*.jpg')))
        mask_paths = sorted(glob.glob(
            osp.join(masks_parent_dir, pid, vid, vpart_id, '*.png')))
        print(f'    \t> vpart_id={vpart_id}: found {len(img_paths)} images.')
        out_dir = osp.join(out_parent_dir, pid, vid, vpart_id)
        os.makedirs(out_dir, exist_ok=True)
        for img_idx, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
          img = np.array(Image.open(img_path)).astype(np.float32)
          mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.
          mask = mask[..., np.newaxis]
          out = (img * mask).astype(np.uint8)
          out_name = osp.basename(img_path)
          out_path = osp.join(out_dir, out_name)
          print(f'      Saving image #{img_idx:02}: {out_path}')
          Image.fromarray(out).save(out_path)
    
def mask_bilayer_with_black_background():
  images_parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/results/bilayer/test'
  out_parent_dir = '/fs/vulcan-projects/two_step_synthesis_meshry/results/bilayer_with_black_bg/test'
  
  person_ids = sorted(os.listdir(images_parent_dir))
  person_ids = [p for p in person_ids if p.startswith('id')]
  for pidx, pid in enumerate(person_ids):
    v_ids = os.listdir(osp.join(images_parent_dir, pid))
    print(f'Processing person #{pidx:2d}: {pid}: found {len(v_ids)} videos.')
    for vid in v_ids:
      vpart_ids = os.listdir(osp.join(images_parent_dir, pid, vid))
      print(f'  > vid={vid}: found {len(vpart_ids)} video parts.')
      for vpart_id in vpart_ids:
        img_paths = sorted(glob.glob(
            osp.join(images_parent_dir, pid, vid, vpart_id, '*pretrained.png')))
        mask_paths = sorted(glob.glob(
            osp.join(images_parent_dir, pid, vid, vpart_id, '*mask.png')))
        print(f'    \t> vpart_id={vpart_id}: found {len(img_paths)} images.')
        out_dir = osp.join(out_parent_dir, pid, vid, vpart_id)
        os.makedirs(out_dir, exist_ok=True)
        for img_idx, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
          img = np.array(Image.open(img_path)).astype(np.float32)
          mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.
          mask = mask[..., np.newaxis]
          out = (img * mask).astype(np.uint8)
          out_name = osp.basename(img_path)
          out_path = osp.join(out_dir, out_name)
          print(f'      Saving image #{img_idx:02}: {out_path}')
          Image.fromarray(out).save(out_path)


# mask_lpd()
mask_bilayer_with_black_background()
