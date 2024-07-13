from src.shared import config
from src.shared.run_config import RunConfig
from src.shared.run_state import RunState
from src.shared.database_wrapper import DatabaseWrapper
from src.shared.graph_schema import NodeType, PublicationEdge, AuthorEdge
logger = config.get_logger("LinkNodes")


def link_nodes():
    """Create edges between nodes in the graph database based on author, venue, and keyword similarity.
    """
    run_config = RunConfig(config.RUN_DIR)
    db = DatabaseWrapper()
    state = RunState(config.RUN_ID, config.RUN_DIR)
    if not state.create_edges.state == 'completed':
        create_emb_sim_edges(db, run_config)
        state.create_edges.state = 'completed'
        state.save()
