import os
import json

from ..shared import config
from ..shared.run_config import RunConfig
from ..datasets.who_is_who import WhoIsWhoDataset
from ..shared import run_state
from ..shared.database_wrapper import DatabaseWrapper

logger = config.get_logger('CreateKG')


def create_nodes():
    """Expects a list of dictionaries where each dictionary comprises the publication id, the embedding, and any
    information that can be used to form links between publications."""

    run_config = RunConfig(config.RUN_DIR)
    logger.info(f"Populating neo4j database {run_config.create_nodes.db_name} ...")

    if not run_state.completed('create_nodes', 'who_is_who'):
        logger.info("Creating nodes for WhoIsWho dataset ...")
        WhoIsWhoDataset.parse_graph(run_config.create_nodes.max_nodes)
        run_state.set_state('create_nodes', 'who_is_who', 'completed')

    logger.info("Done.")
