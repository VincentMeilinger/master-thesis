import json
from time import time
from sentence_transformers import SentenceTransformer

from ..shared import config
from ..shared.run_config import RunConfig
from ..shared.database_wrapper import DatabaseWrapper

logger = config.get_logger("EmbedDatasets")


def yield_node_embeddings(model, publication_data: list) -> list:
    """ Create embeddings for each publication in the publication data dict.
    """
    titles = []
    abstracts = []

    for pub in publication_data:
        titles.append(pub['title'])
        abstracts.append(pub['abstract'])

    title_embeddings = model.encode(titles, convert_to_numpy=True)
    abstract_embeddings = model.encode(abstracts, convert_to_numpy=True)

    # Encode the ndarray as a base64 string
    data = []
    for pub, title_emb, abstract_emb in zip(publication_data, title_embeddings, abstract_embeddings):
        yield pub['id'], title_emb, abstract_emb
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
        for pub_id, title_emb, abstract_emb in yield_node_embeddings(model, batch):
            db.merge_node("Publication", pub_id, {
                'title_emb': title_emb.tolist(),
                'abstract_emb': abstract_emb.tolist()
            })

        logger.info(f"Batch processed in {time() - start_time:.2f} seconds.")


def create_edge_embeddings(publication_data: list, state: dict, batch_file_name: str) -> dict:
    """ Create embeddings for edges between publications.
    """

    raise NotImplementedError


def save_processed_data(data: list, file_name: str):
    """ Save the processed data to a file. """
    # Save as json
    with open(file_name, 'w') as file:
        json.dump(data, file)


def embed_nodes():
    """ Process the WhoIsWho dataset. """
    # Check pipeline state
    logger.info("Embedding nodes in the neo4j graph ...")
    state = config.get_pipeline_state()

    if state['embed_datasets']['embed_nodes']['state'] == 'completed':
        logger.info("Datasets already embedded. Skipping ...")
        return

    # Embed nodes
    create_node_embeddings_batch(state)
    state['embed_datasets']['embed_nodes']['state'] = 'completed'
    config.save_pipeline_state(state)


if __name__ == '__main__':
    embed_nodes()
