import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.shared import config
from src.shared import run_config
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


def link_node_attr_cosine(db: DatabaseWrapper, node_type: NodeType, vec_attr: str,
                          edge_type: EdgeType):
    if run_state.completed('link_nodes', f'link_{node_type.value}_{edge_type.value}'):
        logger.info(f"Linking {node_type.value} nodes already completed. Skipping ...")
        return

    num_nodes = db.count_nodes(node_type)
    logger.info(f"Linking {node_type.value} nodes based on {vec_attr} attribute ...")
    with tqdm(total=num_nodes, desc=f"Progress {node_type.value} {vec_attr}") as pbar:
        for nodes in db.iter_nodes(node_type, ['id', vec_attr]):
            pbar.update(len(nodes))
            for node in nodes:
                # Skip nodes with zero vectors
                if np.sum(node[vec_attr]) == 0:
                    continue

                k = int(run_config.get_config('link_nodes', 'k_nearest_limit'))
                similar_nodes = db.get_similar_nodes_vec(
                    node_type,
                    vec_attr,
                    node[vec_attr],
                    0.99,
                    k
                )
                for ix, row in similar_nodes.iterrows():
                    if row['id'] == node['id']:
                        continue
                    db.merge_edge(node_type, node['id'], node_type, row['id'], edge_type, {"sim": row['sim']})

    run_state.set_state('link_nodes', f'link_{node_type.value}_{edge_type.value}', 'completed')


def link_nodes():
    """Create edges between nodes in the graph database based on author, venue, and keyword similarity.
    """

    db = DatabaseWrapper()

    logger.info("Linking nodes based on attribute similarities ...")
    link_node_attr_cosine(db, NodeType.ORGANIZATION, 'vec', EdgeType.SIM_ORG)
    link_node_attr_cosine(db, NodeType.VENUE, 'vec', EdgeType.SIM_VENUE)
    link_node_attr_cosine(db, NodeType.PUBLICATION, 'keywords_emb', EdgeType.SIM_KEYWORDS)

    db.close()
    logger.info("Done.")
