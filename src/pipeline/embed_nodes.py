import json
import numpy as np
from time import time
from sentence_transformers import SentenceTransformer

from ..shared import config
from ..shared.run_config import RunConfig
from ..shared import run_state
from ..shared.database_wrapper import DatabaseWrapper
from ..shared.graph_schema import NodeType, EdgeType
from src.shared import run_config

logger = config.get_logger("EmbedNodes")


def embed_string_attr(
        run_config: RunConfig,
        model: SentenceTransformer,
        db: DatabaseWrapper,
        node_type: NodeType,
        attr_key: str,
        feature_key: str = None
):
    if run_state.completed('embed_nodes', f'embed_{node_type.value}_{attr_key}_emb'):
        logger.info(f"Embedding {node_type.value} {attr_key} attributes already completed. Skipping ...")
        return

    if feature_key is None:
        feature_key = f"{attr_key}_emb"
    for nodes in db.iter_nodes(node_type, ['id', attr_key]):
        logger.debug(f"Embedding {len(nodes)} {node_type.value} nodes ...")
        ids = [node['id'] for node in nodes]
        strings = [node[attr_key] for node in nodes]
        embeddings = model.encode(strings)
        merge_nodes = []
        for node_id, emb in zip(ids, embeddings):
            merge_nodes.append(
                {'id': node_id, 'properties': {f'{feature_key}': emb.tolist()}}
            )
        db.merge_nodes(node_type, merge_nodes)

    db.create_vector_index(feature_key, node_type, feature_key, run_config.transformer_dim_reduction.reduced_dim)
    run_state.set_state('embed_nodes', f'embed_{node_type.value}_{attr_key}_emb', 'completed')


def embed_nodes():
    """ Process the WhoIsWho dataset. """
    # Check pipeline state
    logger.info("Embedding nodes in the neo4j graph ...")
    run_config = RunConfig(config.RUN_DIR)

    db = DatabaseWrapper()

    if not run_state.completed('embed_nodes', 'embed_node_attributes'):
        logger.info("Embedding node attributes ...")
        model = SentenceTransformer(
            'jordyvl/scibert_scivocab_uncased_sentence_transformer',
            device=config.DEVICE
        )
        # Embed Organization nodes
        embed_string_attr(run_config, model, db, NodeType.ORGANIZATION, 'name', 'vec')
        # Embed Venue nodes
        embed_string_attr(run_config, model, db, NodeType.VENUE, 'name', 'vec')
        # Embed Author nodes
        embed_string_attr(run_config, model, db, NodeType.AUTHOR, 'name', 'vec')
        # Embed Paper nodes
        embed_string_attr(run_config, model, db, NodeType.PUBLICATION, 'abstract', 'vec')

        run_state.set_state('embed_nodes', 'embed_node_attributes', 'completed')


if __name__ == '__main__':
    embed_nodes()
