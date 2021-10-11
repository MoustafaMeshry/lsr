# python -u release_main.py --batch_size 1 --version parsenet --train False --test_image_path /fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_test_videos_processed --imsize 256
# python -u release_main.py --batch_size 1 --version parsenet --train False --test_image_path /fs/vulcan-projects/network_analysis_sakshams/test_two_Step/lsr/_datasets/sample_test_videos_processed --imsize 512
from release_parameter import *
from trainer import Trainer
from release_tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder

def main(config):
    # For fast training
    cudnn.benchmark = True

    if config.train:

    # Create directories if not exist
        make_folder(config.model_save_path, config.version)
        make_folder(config.sample_path, config.version)
        make_folder(config.log_path, config.version)

        data_loader = Data_Loader(config.img_path, config.label_path, config.imsize,
                             config.batch_size, config.train)
        trainer = Trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    # print(config)
    main(config)
