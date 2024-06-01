from ..shared import config
import pandas as pd
import os

logger = config.get_logger("Dataset")


class Dataset:
    name: str = 'name'

    def __init__(
            self,
            train: pd.DataFrame = None,
            test: pd.DataFrame = None,
            valid: pd.DataFrame = None,
    ):
        self.train = train
        self.test = test
        self.valid = valid

    @staticmethod
    def parse(format: str = 'triples'):
        """ Parse a dataset from raw data. Available formats: triples, dict."""
        if format == 'triples':
            return Dataset._parse_triples()
        elif format == 'dict':
            return Dataset._parse_dict()

    @staticmethod
    def _parse_triples():
        """ Parse a dataset and return as triples. """
        raise NotImplementedError("Method _parse_triples not implemented.")

    @staticmethod
    def _parse_dict():
        """ Parse a dataset and return as a dictionary. """
        raise NotImplementedError("Method _parse_dict not implemented.")

    def process(self):
        """ Process the dataset. """
        raise NotImplementedError("Method process not implemented.")

    def load(self):
        """ Load a previously saved dataset from disk. """
        logger.info(f"Loading {self.name} dataset from disk ... ", end='')
        if not os.path.exists(f'datasets_processed/{self.name}'):
            raise FileNotFoundError(f"Dataset {self.name} not found.")

        # Load the pandas DataFrame from disk
        if os.path.exists(f'datasets_processed/{self.name}/train.csv'):
            self.train = pd.read_csv(f'datasets_processed/{self.name}/train.csv')
        if os.path.exists(f'datasets_processed/{self.name}/test.csv'):
            self.test = pd.read_csv(f'datasets_processed/{self.name}/test.csv')
        if os.path.exists(f'datasets_processed/{self.name}/valid.csv'):
            self.valid = pd.read_csv(f'datasets_processed/{self.name}/valid.csv')
        logger.info("done.")

    def save(self):
        """ Save the dataset to disk. """
        logger.info(f"Saving {self.name} dataset to disk ... ", end='')
        # Create the directory to store the processed dataset
        os.makedirs(f'datasets_processed/{self.name}', exist_ok=True)

        # Store the pandas DataFrame to disk
        if self.train:
            self.train.to_csv(f'datasets_processed/{self.name}/train.csv', index=False)
        if self.test:
            self.test.to_csv(f'datasets_processed/{self.name}/test.csv', index=False)
        if self.valid:
            self.valid.to_csv(f'datasets_processed/{self.name}/valid.csv', index=False)
        logger.info("done.")

    def print_stats(self):
        logger.info(f"{self.name} - Dataset Statistics:")
        logger.info("_______________________________________________")
        if self.train:
            logger.info(f"Train ")
            logger.info(f"- # triples: {self.train.shape[0]}")
            logger.info(f"- # unique relations: {self.train['r'].nunique()}")
            logger.info(f"- # unique head entities: {self.train['h'].nunique()}")
        if self.test:
            logger.info(f"Test ")
            logger.info(f"- # triples: {self.test.shape[0]}")
            logger.info(f"- # unique relations: {self.test['r'].nunique()}")
            logger.info(f"- # unique head entities: {self.test['h'].nunique()}")
        if self.valid:
            logger.info(f"Valid ")
            logger.info(f"- # triples: {self.valid.shape[0]}")
            logger.info(f"- # unique relations: {self.valid['r'].nunique()}")
            logger.info(f"- # unique head entities: {self.valid['h'].nunique()}")
        logger.info("===============================================")
