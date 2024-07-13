from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.shared import config
from src.shared.run_config import RunConfig
from src.shared.run_state import RunState
from src.shared.database_wrapper import DatabaseWrapper
from src.shared.graph_schema import NodeType, EdgeType, PublicationEdge, AuthorEdge

logger = config.get_logger("LinkNodes")

def string_similarity(s1, s2):
    return len(set(s1) & set(s2)) / len(set(s1) | set(s2))

def link_string_nodes(node_type_1: NodeType, node_type_2: NodeType, edge_type: EdgeType, threshold=0.9):
    model = SentenceTransformer('jordyvl/scibert_scivocab_uncased_sentence_transformer')

def link_nodes():
    """Create edges between nodes in the graph database based on author, venue, and keyword similarity.
    """
    run_config = RunConfig(config.RUN_DIR)
    run_state = RunState(config.RUN_ID, config.RUN_DIR)

    db = DatabaseWrapper()

    if run_state.link_nodes.state == 'not_started':
        logger.info("Creating edges between nodes ...")
        db.create_edges(NodeType.AUTHOR, AuthorEdge)
        db.create_edges(NodeType.PUBLICATION, PublicationEdge)
        state.link_nodes.state = 'completed'
        state.save()
        logger.info("Done.")
