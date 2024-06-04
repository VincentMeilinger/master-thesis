from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import base64
from time import time
from ..shared import config
from ..datasets.who_is_who import WhoIsWhoDataset

logger = config.get_logger("EmbedDatasets")


def create_embeddings(model, publication_data: list) -> dict:
    """ Create embeddings for each publication in the publication data dict.
    """
    titles = []
    abstracts = []

    for pub in publication_data:
        titles.append(pub['title'])
        abstracts.append(pub['abstract'])

    title_embeddings = model.encode(titles, convert_to_numpy=True)
    abstract_embeddings = model.encode(abstracts, convert_to_numpy=True)
    combined_embeddings = np.hstack((title_embeddings, abstract_embeddings))

    # Encode the ndarray as a base64 string
    return {pub["id"]: base64.b64encode(emb.tobytes()).decode('utf-8')
            for pub, emb in zip(publication_data, combined_embeddings)}


def create_embeddings_batch(publication_data: list, state: dict, batch_file_name: str):
    """ Create embeddings for each publication in the publication data dict in batches. Save each batch to disk.
    """
    # Load configuration parameters
    params = config.get_params()

    # Load SentenceTransformer model
    logger.info(f"Loading SentenceTransformer model {params['embed_datasets']['transformer_model']} ...")
    model = SentenceTransformer(
        params['embed_datasets']['transformer_model'],
        device=config.device,
        local_files_only=True
    )

    logger.info(f"Creating publication data embeddings in batches of {params['embed_datasets']['batch_size']} ...")
    num_batches = len(publication_data) // params['embed_datasets']['batch_size']

    # Process publication data in batches
    for i in range(0, len(publication_data), params['embed_datasets']['batch_size']):
        start_time = time()
        current_batch = i // params['embed_datasets']['batch_size'] + 1
        if int(state['embed_datasets']['progress']) >= current_batch:
            logger.info(f"Skipping batch {current_batch}/{num_batches} (already processed).")
            continue

        # Process batch
        logger.info(f"Processing batch {current_batch}/{num_batches} ...")
        batch = publication_data[i:i + params['embed_datasets']['batch_size']]
        embeddings = create_embeddings(model, batch)

        # Save processed data
        logger.info(f"Saving processed data to disk ...")
        save_processed_data(embeddings, os.path.join(config.dataset_processed_dir, f'{batch_file_name}_{current_batch}.json'))

        # Update progress and save pipeline state
        elapsed_time = time() - start_time
        est_time = (elapsed_time * (num_batches - current_batch)) / 60
        logger.info(
            f"Batch processed in {time() - start_time:.2f} seconds. Estimated time left: {est_time:.2f} minutes.")
        state['embed_datasets']['progress'] = current_batch
        config.save_pipeline_state(state)


def save_processed_data(data: dict, file_name: str):
    """ Save the processed data to a file. """
    # Save as json
    with open(file_name, 'w') as file:
        json.dump(data, file)


def clear_processed_data_dir():
    """ Clear the processed data directory. """
    for file in os.listdir(config.dataset_processed_dir):
        os.remove(os.path.join(config.dataset_processed_dir, file))


def embed_datasets():
    """ Process the WhoIsWho dataset. """
    # Check pipeline state
    logger.info("Embedding datasets ...")
    state = config.get_pipeline_state()
    if state['embed_datasets']['state'] == 'in_progress':
        logger.info("Embedding datasets in progress. Trying to recover ...")
        if config.check_params_changed():
            logger.warning("Parameters have changed. Restart embedding process and delete old data? (y/n)")
            if input().lower() == 'y':
                clear_processed_data_dir()
                config.init_pipeline_state()
                embed_datasets()
                return
            else:
                logger.info("Exiting ...")
                return
    elif state['embed_datasets']['state'] == 'completed':
        logger.info("Datasets already embedded. Skipping ...")
        return
    else:
        state['embed_datasets']['progress'] = '0'

    # Process WhoIsWho dataset
    state['embed_datasets']['state'] = 'in_progress'
    data = WhoIsWhoDataset.parse(format='dict')
    data = [value for key, value in data.items()]
    create_embeddings_batch(data, state, 'who_is_who')
    state['embed_datasets']['state'] = 'completed'
    config.save_pipeline_state(state)


if __name__ == '__main__':
    embed_datasets()
