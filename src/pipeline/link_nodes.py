from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.shared import config
from src.shared.run_config import RunConfig
from src.shared.run_state import RunState
from src.shared.database_wrapper import DatabaseWrapper
from src.shared.graph_schema import NodeType, EdgeType, PublicationEdge, AuthorEdge, SimilarityEdge

logger = config.get_logger("LinkNodes")

def string_similarity(s1, s2):
    return len(set(s1) & set(s2)) / len(set(s1) | set(s2))

def transformer_string_similarity(model, strings: list):
    embeddings = model.encode(strings)
    similarity = cosine_similarity(embeddings)
    return similarity

def link_node_attr_cosine(run_config: RunConfig, db: DatabaseWrapper, node_type: NodeType, vec_attr: str, edge_type: EdgeType):
    for nodes in db.iter_nodes(node_type, ['id', vec_attr]):
        logger.debug(f"Finding similar nodes for {len(nodes)} {node_type} nodes ...")
        for node in nodes:
            similar_nodes = db.get_similar_nodes_vec(
                node_type,
                vec_attr,
                node[vec_attr],
                run_config.link_nodes.similarity_threshold,
                run_config.link_nodes.k_nearest_limit
            )
            for node in similar_nodes:
                db.create_edge(node_type, node_type, node[0], node[0], edge_type, node[1])

def link_nodes():
    """Create edges between nodes in the graph database based on author, venue, and keyword similarity.
    """
    run_config = RunConfig(config.RUN_DIR)
    run_state = RunState(config.RUN_ID, config.RUN_DIR)

    db = DatabaseWrapper()

    if run_state.link_nodes.state == 'not_started':
        logger.info("Creating edges between nodes ...")
        link_node_attr_cosine(run_config, db, NodeType.ORGANIZATION, 'id_emb', SimilarityEdge.SIM_ORG)
        link_node_attr_cosine(run_config, db, NodeType.VENUE, 'id_emb', SimilarityEdge.SIM_VENUE)
        run_state.link_nodes.state = 'completed'
        run_state.save()
        logger.info("Done.")
