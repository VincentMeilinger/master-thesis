import os
import json

from ..shared import config
from ..shared.run_config import RunConfig
from ..datasets.who_is_who import WhoIsWhoDataset
from ..shared.run_state import RunState
from ..shared.database_wrapper import DatabaseWrapper

logger = config.get_logger('PopulateDB')


def create_nodes():
    """Expects a list of dictionaries where each dictionary comprises the publication id, the embedding, and any
    information that can be used to form links between publications."""

    run_config = RunConfig(config.RUN_DIR)
    run_state = RunState(config.RUN_ID, config.RUN_DIR)

    if run_state.create_nodes.state == 'completed':
        logger.info("Nodes already created.")
        return

    logger.info(f"Populating neo4j database {run_config.create_nodes.db_name} ...")

    WhoIsWhoDataset.parse_graph(run_config.create_nodes.max_nodes)

    logger.info("Done.")
