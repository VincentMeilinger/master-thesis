import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.shared import config
from src.shared.run_config import RunConfig
from src.shared import run_state
from src.shared.database_wrapper import DatabaseWrapper
from src.shared.graph_schema import NodeType, EdgeType

logger = config.get_logger("LinkNodes")


def string_similarity(s1, s2):
    return len(set(s1) & set(s2)) / len(set(s1) | set(s2))


def transformer_string_similarity(model, strings: list):
    embeddings = model.encode(strings)
    similarity = cosine_similarity(embeddings)
    return similarity


def link_node_attr_cosine(run_config: RunConfig, db: DatabaseWrapper, node_type: NodeType, vec_attr: str,
                          edge_type: EdgeType):
    if run_state.completed('link_nodes', f'link_{node_type.value}_{edge_type.value}'):
        logger.info(f"Linking {node_type.value} nodes already completed. Skipping ...")
        return

    for nodes in db.iter_nodes(node_type, ['id', vec_attr]):
        logger.debug(f"Finding similar nodes for {len(nodes)} {node_type} nodes ...")
        for node in nodes:
            # Skip nodes with zero vectors
            if np.sum(node[vec_attr]) == 0:
                continue

            similar_nodes = db.get_similar_nodes_vec(
                node_type,
                vec_attr,
                node[vec_attr],
                0.9,
                run_config.link_nodes.k_nearest_limit
            )
            for ix, row in similar_nodes.iterrows():
                if row['id'] == node['id']:
                    continue
                #print(f"Similarity {row['sim']} between \n{node['id']}\n{row['id']}")
                db.merge_edge(node_type, node['id'], node_type, row['id'], edge_type, {"sim": row['sim']})

    run_state.set_state('link_nodes', f'link_{node_type.value}_{edge_type.value}', 'completed')


def link_nodes():
    """Create edges between nodes in the graph database based on author, venue, and keyword similarity.
    """
    run_config = RunConfig(config.RUN_DIR)

    db = DatabaseWrapper()

    if not run_state.completed('link_nodes', 'link_node_attributes'):
        logger.info("Creating edges between nodes ...")
        link_node_attr_cosine(run_config, db, NodeType.ORGANIZATION, 'vec', EdgeType.SIM_ORG)
        link_node_attr_cosine(run_config, db, NodeType.VENUE, 'vec', EdgeType.SIM_VENUE)
        link_node_attr_cosine(run_config, db, NodeType.PUBLICATION, 'keywords_emb', EdgeType.SIM_KEYWORDS)
        run_state.set_state('link_nodes', 'link_node_attributes', 'completed')
        logger.info("Done.")
