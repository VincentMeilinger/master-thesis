import json
import os

from ..shared.database_wrapper import DatabaseWrapper
from ..shared import config
from ..datasets.who_is_who import WhoIsWhoDataset
from ..shared.run_config import RunConfig
logger = config.get_logger('PopulateDB')


def add_publication_nodes(db: DatabaseWrapper, data, run_config: RunConfig):
    logger.debug("Populating neo4j graph database with publication nodes ...")
    for pub in data:
        pub_id = pub.pop('id')
        pub_data = {
            'title': pub['title'],
            'abstract': pub['abstract'][0:run_config.populate_db.max_seq_len],
            'venue': pub['venue'],
            'year': pub['year'],
            'keywords': pub['keywords'],
        }
        db.merge_node("Publication", pub_id, pub_data)


def populate_db():
    """Expects a list of dictionaries where each dictionary comprises the publication id, the embedding, and any
    information that can be used to form links between publications."""

    run_config = RunConfig(config.RUN_DIR)
    state = config.get_pipeline_state()

    logger.info(f"Populating neo4j database {run_config.populate_db.db_name} ...")
    db = DatabaseWrapper()

    data = WhoIsWhoDataset.parse(format='dict')
    data = [value for key, value in data.items()]
    if run_config.populate_db.max_nodes is not None:
        data = data[:run_config.populate_db.max_nodes]

    # Add publication nodes
    if not state['populate_db']['nodes'] == 'completed':
        add_publication_nodes(db, data, run_config)
        state['populate_db']['nodes'] = 'completed'

    # Add connections to other papers based on author, venue, and keywords
    # TODO

    logger.info("Done.")
