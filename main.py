import os
import shutil
from configs.config import Config
from configs.models import architectures

from lib.utils import setup_seed
from torch import optim
from models.architectures import KPFCNN


def backup(option=False):
    if option:
        os.system(f'cp -r models {config.snapshot_dir}')
        os.system(f'cp -r datasets {config.snapshot_dir}')
        os.system(f'cp -r lib {config.snapshot_dir}')
        shutil.copy2('main.py', config.snapshot_dir)


setup_seed(0)

if __name__ == '__main__':
    # load configurations
    config = Config()

    # backup the current file options and codes True or False
    backup()

    # model initialization
    config.architecture = architectures[config.dataset]
    config.model = KPFCNN(config)
