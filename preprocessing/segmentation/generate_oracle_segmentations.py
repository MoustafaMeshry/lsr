# python -u release_main.py --batch_size 1 --version parsenet --train False --test_image_path /fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_test_videos_processed --imsize 256
# python -u release_main.py --batch_size 1 --version parsenet --train False --test_image_path /fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_test_videos_processed --imsize 512
from release_parameter import *
from release_tester import Tester
from release_merge import merge_segmentations
import os

def main():
    config = get_parameters()
    print('Processing images at resolution 256 ...')
    config.imsize = 256
    tester = Tester(config)
    tester.test()
    print('Processing images at resolution 512 ...')
    config.imsize = 512
    tester = Tester(config)
    tester.test()
    print('Merging segmentations ...')
    merge_segmentations(config.test_image_path)
    print('Removing intermediate files ...')
    os.system(f'rm {config.test_image_path}/*/*/*/*_segmap_256.png')
    os.system(f'rm {config.test_image_path}/*/*/*/*_segmap_512.png')

if __name__ == '__main__':
    main()
