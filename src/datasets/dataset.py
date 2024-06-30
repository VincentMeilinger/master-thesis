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
