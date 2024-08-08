import json
import numpy as np
from time import time
from sentence_transformers import SentenceTransformer

from src.shared.database_wrapper import DatabaseWrapper
from src.shared.graph_schema import NodeType, EdgeType
from src.shared import run_config
from src.shared import run_state
from src.shared import config

logger = config.get_logger("EmbedNodes")


def embed_string_attr(
        model: SentenceTransformer,
        db: DatabaseWrapper,
        node_type: NodeType,
        attr_key: str,
        feature_key: str = None
):
    if run_state.completed('embed_nodes', f'embed_{node_type.value}_{attr_key}_emb'):
        logger.info(f"Embedding {node_type.value} {attr_key} attributes already completed. Skipping ...")
        return
    logger.info(f"Embedding {node_type.value} {attr_key} attributes ...")

    if feature_key is None:
        feature_key = f"{attr_key}_emb"

    for nodes in db.iter_nodes(node_type, ['id', attr_key]):
        logger.debug(f"Embedding {len(nodes)} {node_type.value} nodes ...")
        node_ids = []
        node_attrs = []
        for node in nodes:
            if node[attr_key] is None:
                continue
            node_ids.append(node['id'])
            node_attrs.append(node[attr_key])

        embeddings = model.encode(node_attrs)
        merge_nodes = []
        for node_id, emb in zip(node_ids, embeddings):
            merge_nodes.append(
                {'id': node_id, 'properties': {f'{feature_key}': emb.tolist()}}
            )
        db.merge_nodes(node_type, merge_nodes)

    reduced_dim = run_config.get_config('transformer_dim_reduction', 'reduced_dim')
    db.create_vector_index(feature_key, node_type, feature_key, reduced_dim)
    run_state.set_state('embed_nodes', f'embed_{node_type.value}_{attr_key}_emb', 'completed')


def embed_pub_keywords(
        model: SentenceTransformer,
        db: DatabaseWrapper,
        node_type: NodeType,
        attr_key: str,
        feature_key: str = None
):
    if run_state.completed('embed_nodes', 'embed_keywords'):
        logger.info(f"Embedding {node_type.value} {attr_key} attributes already completed. Skipping ...")
        return
    logger.info(f"Embedding {node_type.value} {attr_key} attributes ...")

    if feature_key is None:
        feature_key = f"{attr_key}_emb"
    for nodes in db.iter_nodes(node_type, ['id', attr_key]):
        logger.debug(f"Embedding {len(nodes)} {node_type.value} nodes ...")
        ids = [node['id'] for node in nodes]
        strings = [' '.join(node[attr_key]) for node in nodes]
        embeddings = model.encode(strings)
        # If the string is empty or 'null', replace the embedding with all zeros
        embeddings = [emb if (len(strings[i]) > 4) else np.zeros_like(emb) for i, emb in enumerate(embeddings)]

        merge_nodes = []
        for node_id, emb in zip(ids, embeddings):
            merge_nodes.append(
                {'id': node_id, 'properties': {f'{feature_key}': emb.tolist()}}
            )
        db.merge_nodes(node_type, merge_nodes)

    reduced_dim = run_config.get_config('transformer_dim_reduction', 'reduced_dim')
    db.create_vector_index(feature_key, node_type, feature_key, reduced_dim)
    run_state.set_state('embed_nodes', f'embed_{node_type.value}_{attr_key}_emb', 'completed')


def embed_nodes():
    """ Process the WhoIsWho dataset. """
    # Check pipeline state
    logger.info("Embedding nodes in the neo4j graph ...")

    db = DatabaseWrapper()

    if not run_state.completed('embed_nodes', 'embed_node_attributes'):
        logger.info("Embedding node attributes ...")
        model_name = run_config.get_config('embed_nodes', 'transformer_model')
        model = SentenceTransformer(
            model_name,
            device=config.DEVICE
        )
        # Embed Organization nodes
        embed_string_attr(model, db, NodeType.ORGANIZATION, 'name', 'vec')
        # Embed Venue nodes
        embed_string_attr(model, db, NodeType.VENUE, 'name', 'vec')
        # Embed Author nodes
        embed_string_attr(model, db, NodeType.AUTHOR, 'name', 'vec')
        # Embed Paper nodes
        embed_string_attr(model, db, NodeType.PUBLICATION, 'abstract', 'vec')

        # Embed Publication keywords
        embed_pub_keywords(model, db, NodeType.PUBLICATION, 'keywords', 'keywords_emb')

        run_state.set_state('embed_nodes', 'embed_node_attributes', 'completed')


if __name__ == '__main__':
    embed_nodes()
