import json
import os
import pandas as pd
from src.datasets.dataset import Dataset
from src.shared import config

logger = config.get_logger("OC782KDataset")


class OC782KDataset(Dataset):
    name: str = 'OC-782K'

    @staticmethod
    def parse(format: str = 'triples'):
        if format == 'triples':
            return OC782KDataset._parse_triples()
        elif format == 'dict':
            return OC782KDataset._parse_dict()

    @staticmethod
    def _parse_triples():
        logger.info("Parsing dataset OC-782K ...")
        file_path = os.path.join(config.DATASET_DIR, 'OC-782K/')
        logger.debug(f"Loading data from {file_path}")
        data = {'train': None, 'test': None, 'valid': None}
        files = {'training.txt': 'train', 'testing.txt': 'test', 'validation.txt': 'valid'}
        for file in files.keys():
            data[files[file]] = pd.read_csv(file_path + file, sep='\t', header=None, names=['h', 'r', 't'])

        return data

    @staticmethod
    def parse_train():
        logger.info("Parsing train dataset OC-782K ...")
        file_path = os.path.join(config.DATASET_DIR, 'OC-782K/training.txt')
        return pd.read_csv(file_path + file_path, sep='\t', header=None, names=['h', 'r', 't'])

    @staticmethod
    def parse_valid():
        logger.info("Parsing train dataset OC-782K ...")
        file_path = os.path.join(config.DATASET_DIR, 'OC-782K/and_eval.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as file:
            data = json.load(file)

        return data

    def print_stats(self):
        super().print_stats()
