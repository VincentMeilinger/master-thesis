import json
import os

from .create_embeddings import create_embeddings_batch
from ..shared import config
from ..datasets.who_is_who import WhoIsWhoDataset

logger = config.get_logger("DatasetPreProcessing")


def save_processed_data(data: list, file_name: str):
    """ Save the processed data to a file. """
    # Save as json
    with open(file_name, 'w') as file:
        json.dump(data, file)


def process_who_is_who():
    """ Process the WhoIsWho dataset. """
    data = WhoIsWhoDataset.parse(format='dict')
    data = [value for key, value in data.items()]
    embeddings = create_embeddings_batch(data)
    save_processed_data(embeddings, os.path.join(config.dataset_processed_dir, 'who_is_who.json'))


if __name__ == '__main__':
    process_who_is_who()
