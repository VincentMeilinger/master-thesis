import json
import numpy as np
from time import time
from sentence_transformers import SentenceTransformer

from ..shared import config
from ..shared.run_config import RunConfig
from ..shared.run_state import RunState
from ..shared.database_wrapper import DatabaseWrapper
from ..shared.graph_schema import NodeType, EdgeType
from src.shared import run_config

logger = config.get_logger("EmbedNodes")


def yield_node_embeddings(model, publication_data: list) -> list:
    """ Create embeddings for each publication in the publication data dict.
    """
    pub_ids = []
    titles = []
    abstracts = []
    year_embeddings = []

    for pub in publication_data:
        pub_ids.append(pub['id'])
        titles.append(pub['title'])
        abstracts.append(pub['abstract'])
        if 'year' in pub:
            year = int(pub['year'])
            normalized_year = (year - 1900) / (2100 - 1900)  # Normalize year between 1900 and 2100
            year_emb = [
                np.sin(2 * np.pi * normalized_year),
                np.cos(2 * np.pi * normalized_year)
            ]
            year_embeddings.append(year_emb)
        else:
            year_embeddings.append([0] * 2)

    title_embeddings = model.encode(titles, convert_to_numpy=True)
    abstract_embeddings = model.encode(abstracts, convert_to_numpy=True)

    data = []
    for pub, title_emb, abstract_emb, year_emb in zip(publication_data, title_embeddings, abstract_embeddings, year_embeddings):
        yield pub['id'], title_emb, abstract_emb, year_emb
    return data


def create_node_embeddings_batch(state: dict):
    """ Create embeddings for each publication in the publication data dict in batches. Save each batch to disk.
    """
    # Load configuration parameters
    run_config = RunConfig(config.RUN_DIR)
    db = DatabaseWrapper()

    # Load SentenceTransformer model
    logger.info(f"Loading SentenceTransformer model {run_config.embed_datasets.transformer_model} ...")
    model = SentenceTransformer(
        run_config.embed_datasets.transformer_model,
        device=config.DEVICE,
        local_files_only=True
    )

    logger.info(f"Creating publication data embeddings in batches of {run_config.embed_datasets.batch_size} ...")
    # Process publication data in batches
    for batch in db.iterate_all_papers(run_config.embed_datasets.batch_size):
        start_time = time()

        # Process batch
        for pub_id, title_emb, abstract_emb, year_emb in yield_node_embeddings(model, batch):
            db.merge_node("Publication", pub_id, {
                'title_emb': title_emb.tolist(),
                'abstract_emb': abstract_emb.tolist(),
                'year_emb': year_emb
            })

        logger.info(f"Batch processed in {time() - start_time:.2f} seconds.")

def embed_string_attr(run_config: RunConfig, model: SentenceTransformer, db: DatabaseWrapper, node_type: NodeType, attr_key: str):
    emb_attr_key = f"{attr_key}_emb"
    db.create_vector_index(emb_attr_key, node_type, emb_attr_key, run_config.transformer_dim_reduction.reduced_dim)
    for nodes in db.iter_nodes(node_type, [attr_key]):
        logger.debug(f"Embedding {len(nodes)} {node_type.value} nodes ...")
        strings = [node[attr_key] for node in nodes]
        embeddings = model.encode(strings)
        merge_nodes = []
        for node_id, emb in zip(strings, embeddings):
            merge_nodes.append(
                {'id': node_id, 'properties': {f'{emb_attr_key}': emb.tolist()}}
            )
        db.merge_nodes(node_type, merge_nodes)

def embed_nodes():
    """ Process the WhoIsWho dataset. """
    # Check pipeline state
    logger.info("Embedding nodes in the neo4j graph ...")
    run_state = RunState(config.RUN_ID, config.RUN_DIR)
    run_config = RunConfig(config.RUN_DIR)

    if run_state.embed_nodes.state == 'completed':
        logger.info("Datasets already embedded. Skipping ...")
        return

    db = DatabaseWrapper()

    model = SentenceTransformer(
        'jordyvl/scibert_scivocab_uncased_sentence_transformer',
        device=config.DEVICE
    )

    # Embed Organization nodes
    embed_string_attr(run_config, model, db, NodeType.ORGANIZATION, 'name')
    # Emebed Venue nodes
    embed_string_attr(run_config, model, db, NodeType.VENUE, 'name')

    #create_node_embeddings_batch(run_state)
    run_state.embed_nodes.state = 'completed'
    run_state.save()


if __name__ == '__main__':
    embed_nodes()
