import pandas as pd
from prettytable import PrettyTable
from dataset import Dataset
from ..shared import config

logger = config.get_logger("OC782KDataset")


class OC782KDataset(Dataset):
    name: str = 'OC-782K'

    @staticmethod
    def parse():
        logger.info("Parsing dataset OC-782K ...")
        file_path = 'data/OC-782K/'
        logger.debug(f"Loading data from {file_path}")
        data = {'train': None, 'test': None, 'valid': None}
        files = {'training.txt': 'train', 'testing.txt': 'test', 'validation.txt': 'valid'}
        for file in files.keys():
            data[files[file]] = pd.read_csv(file_path + file, sep='\t', header=None, names=['h', 'r', 't'])

        return data

    def load(self):
        super().load()

    def save(self):
        super().save()

    def print_stats(self):
        super().print_stats()