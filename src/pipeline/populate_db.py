import json
import os

from ..shared.database_wrapper import DatabaseWrapper
from ..shared import config

logger = config.get_logger('PopulateDB')


def iter_processed_publication_data():
    """ Save the processed data to a file. """
    # Save as json
    file_names = os.listdir(config.dataset_processed_dir)
    num_files = len(file_names)
    current_file = 1
    for file_name in file_names:
        file_path = os.path.join(config.dataset_processed_dir, file_name)
        with open(file_path, 'r') as file:
            logger.info(f"Parsing file {current_file}/{num_files} ...")
            current_file += 1
            data = json.load(file)
            yield data


def add_publication_nodes(db: DatabaseWrapper):
    logger.debug("Populating neo4j graph database with publication nodes ...")
    for pub_id, pub_embedding in iter_processed_publication_data():
        # Add a node containing paper id and embedding
        db.merge_paper({
            pub_id: pub_embedding
        })


def populate_db(data: list):
    """Expects a list of dictionaries where each dictionary comprises the publication id, the embedding, and any
    information that can be used to form links between publications."""

    params = config.get_params()['populate_db']
    state = config.get_pipeline_state()

    logger.info(f"Populating neo4j database {params['neo4j_database_name']}")
    db = DatabaseWrapper()

    # Add publication nodes
    add_publication_nodes(db)

    # Add connections to other papers based on author, venue, and keywords
    raise NotImplementedError
    logger.info("Done.")
