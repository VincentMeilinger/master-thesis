import json
import os

from ..shared import config
from ..datasets.who_is_who import WhoIsWhoDataset
import src.shared.run_state

logger = config.get_logger("DatasetPreProcessing")


def dataset_pre_processing():
    """Preprocess the dataset."""
    logger.info("Preprocessing the dataset ...")
    if not src.shared.run_state.completed('dataset_pre_processing', 'who_is_who'):
        logger.info("Preprocessing the WhoIsWho dataset ...")
        #WhoIsWhoDataset.parse_graph()
        src.shared.run_state.set_state('dataset_pre_processing', 'who_is_who', 'completed')
