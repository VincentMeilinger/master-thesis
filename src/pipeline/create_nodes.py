import os
import json

from ..shared import config
from ..shared.run_config import RunConfig
from ..datasets.who_is_who import WhoIsWhoDataset
from ..shared.pipeline_state import PipelineState
from ..shared.database_wrapper import DatabaseWrapper

logger = config.get_logger('PopulateDB')


def add_publication_nodes(db: DatabaseWrapper, data, run_config: RunConfig):
    logger.info("Creating publication nodes ...")
    for pub in data:
        pub_id = pub.pop('id')
        pub_data = {
            'title': pub['title'],
            'abstract': pub['abstract'][0:run_config.create_nodes.max_seq_len],
            'venue': pub['venue'],
            'year': pub['year'],
            'keywords': pub['keywords'],
            'true_authors': []
        }
        db.merge_node("Publication", pub_id, pub_data)


def add_true_authors(db: DatabaseWrapper):
    logger.info("Adding true authors to publication nodes ...")
    data = WhoIsWhoDataset.parse_train()
    for author_id, values in data.items():
        for pub_id in values['normal_data']:
            query = """
            MATCH (n:Publication {id: $pub_id})
            SET n.true_authors = coalesce(n.true_authors, []) + $author_id
            RETURN n        
            """
            db.custom_query(query, {'pub_id': pub_id, 'author_id': author_id})


def create_nodes():
    """Expects a list of dictionaries where each dictionary comprises the publication id, the embedding, and any
    information that can be used to form links between publications."""

    run_config = RunConfig(config.RUN_DIR)
    state = PipelineState(config.RUN_ID, config.RUN_DIR)

    logger.info(f"Populating neo4j database {run_config.create_nodes.db_name} ...")
    db = DatabaseWrapper()

    data = WhoIsWhoDataset.parse(format='dict')
    data = [value for key, value in data.items()]
    if run_config.create_nodes.max_nodes is not None:
        data = data[:run_config.create_nodes.max_nodes]

    # Add publication nodes
    if not state.create_nodes.create_publication_nodes_state == 'completed':
        add_publication_nodes(db, data, run_config)
        state.create_nodes.create_publication_nodes_state = 'completed'
        state.save()

    # Add true authors from training data
    if not state.create_nodes.add_true_authors_state == 'completed':
        add_true_authors(db)
        state.create_nodes.add_true_authors_state = 'completed'
        state.save()

    logger.info("Done.")
