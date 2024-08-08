import os
import json

from ..shared import config
from ..datasets.who_is_who import WhoIsWhoDataset
from ..shared import run_state
from ..shared import run_config
from ..shared.database_wrapper import DatabaseWrapper

logger = config.get_logger('CreateKG')


def create_nodes():
    """Expects a list of dictionaries where each dictionary comprises the publication id, the embedding, and any
    information that can be used to form links between publications."""

    logger.info(f"Populating neo4j database {run_config.get_config('create_nodes', 'db_name')} ...")

    if not run_state.completed('create_nodes', 'who_is_who'):
        logger.info("Creating nodes for WhoIsWho dataset ...")
        max_nodes = run_config.get_config('create_nodes', 'max_nodes')
        max_nodes = max_nodes if isinstance(max_nodes, int) else None
        WhoIsWhoDataset.parse_graph(max_nodes)
        run_state.set_state('create_nodes', 'who_is_who', 'completed')
    else:
        logger.info("WhoIsWho dataset already processed. Skipping ...")

    logger.info("Done.")
