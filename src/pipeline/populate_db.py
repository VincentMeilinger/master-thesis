import json
import os

from ..shared.database_wrapper import DatabaseWrapper
from ..shared import config

logger = config.get_logger('PopulateDB')


def iter_processed_publications():
    """ Save the processed data to a file. """
    # Save as json
    file_names = os.listdir(config.PROCESSED_DATA_DIR)
    num_files = len(file_names)
    current_file = 1
    for file_name in file_names:
        file_path = os.path.join(config.PROCESSED_DATA_DIR, file_name)
        with open(file_path, 'r') as file:
            logger.info(f"Parsing file {current_file}/{num_files} ...")
            current_file += 1
            data = json.load(file)
            for pub in data:
                yield pub


def add_publication_nodes(db: DatabaseWrapper):
    logger.debug("Populating neo4j graph database with publication nodes ...")
    for pub in iter_processed_publications():
        # Add a node containing paper id and embedding
        # Pop id value from the dictionary
        pub_id = pub.pop('id')
        db.merge_node("Publication", pub_id, pub)


def populate_db():
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
