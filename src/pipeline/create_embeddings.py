from sentence_transformers import SentenceTransformer
import numpy as np
from ..shared import config

logger = config.get_logger("CreateEmbeddings")


def create_embeddings_batch(publication_data: list) -> list:
    """ Create embeddings for each publication in the publication data dict in batches.
    """
    params = config.get_params()

    logger.info(f"Loading SentenceTransformer model {params['transformer']['model']} ...")
    model = SentenceTransformer(
        params['transformer']['model'],
        device=config.device,
        local_files_only=True
    )

    logger.info(f"Creating publication data embeddings in batches ...")
    num_batches = len(publication_data) // params['transformer']['batch_size']
    embeddings_list = []
    for i in range(0, len(publication_data), params['transformer']['batch_size']):
        logger.info(f"Processing batch {i // params['transformer']['batch_size'] + 1 if i != 0 else 0}/{num_batches}")
        batch = publication_data[i:i + params['transformer']['batch_size']]
        embeddings = create_embeddings(model, batch)
        embeddings_list.extend(embeddings)


    return embeddings_list


def create_embeddings(model, publication_data: list) -> list:
    """ Create embeddings for each publication in the publication data dict.
    """
    logger.info(f"Creating publication data embeddings ...")
    titles = []
    abstracts = []

    for pub in publication_data:
        titles.append(pub['title'])
        abstracts.append(pub['abstract'])

    title_embeddings = model.encode(titles, convert_to_numpy=True)
    abstract_embeddings = model.encode(abstracts, convert_to_numpy=True)
    combined_embeddings = np.hstack((title_embeddings, abstract_embeddings))

    return [{"id": pub["id"], "emb": emb} for pub, emb in zip(publication_data, combined_embeddings)]

