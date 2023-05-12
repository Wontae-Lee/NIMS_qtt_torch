from lib.utils import load_obj
from datasets.indoor import IndoorDataset


def get_datasets(config):
    if config.dataset == 'indoor':
        info_train = load_obj(config.train_info)
        info_val = load_obj(config.val_info)
        info_benchmark = load_obj(f'configs/indoor/{config.benchmark}.pkl')

        train_set = IndoorDataset(info_train, config, data_augmentation=True)
        val_set = IndoorDataset(info_val, config, data_augmentation=False)
        benchmark_set = IndoorDataset(info_benchmark, config, data_augmentation=False)
    else:
        raise NotImplementedError

    # return train_set, val_set, benchmark_set


if __name__ == '__main__':
    pass
