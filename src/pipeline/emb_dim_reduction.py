import os
from time import time
import random
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from ..datasets.who_is_who import WhoIsWhoDataset
from ..shared import config

logger = config.get_logger("EmbDimReduction")


def _compare_performance(emb1, emb2):
    # Compare the similarity computation performance of the two models
    sim1 = cosine_similarity(emb1)
    sim2 = cosine_similarity(emb2)
    difference_matrix = np.abs(sim2 - sim1)
    return np.mean(difference_matrix), np.std(difference_matrix)


def edr_eval(train, full_emb, new_dimension: int, model_name: str = "all-mpnet-base-v2"):
    # Generate embeddings
    logger.info("Generating embeddings ...")
    model = SentenceTransformer(model_name)

    # PCA on train embeddings
    logger.info("Performing PCA on train embeddings ...")
    pca = PCA(n_components=new_dimension)
    pca.fit(full_emb)
    pca_comp = np.asarray(pca.components_)

    # We add a dense layer to the model, so that it will produce directly embeddings with the new size
    logger.info("Adding dense layer to the model ...")
    dense = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=new_dimension,
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module("dense", dense)

    # Evaluate the model with the reduce embedding size
    logger.info(f"Model with {new_dimension} dimensions:")
    red_emb = model.encode(train, convert_to_numpy=True)
    mean_diff, std_diff = _compare_performance(full_emb, red_emb)
    logger.info(f"Mean difference: {mean_diff}, Std difference: {std_diff}")

    # If you like, you can store the model on disc by uncommenting the following line
    model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    os.makedirs("data/models", exist_ok=True)
    model.save(f"data/models/{model_name}-{new_dimension}dim")


if __name__ == '__main__':
    # Parse datasets
    # titles = []
    abstracts = []
    # keywords = set()
    # venues = set()

    data = WhoIsWhoDataset.parse(format='dict')
    for paper_id, paper_info in data.items():
        # titles.append(paper_info['title'])
        abstracts.append(paper_info['abstract'])
        # keywords.update(set(paper_info['keywords']))
        # venues.add(paper_info['venue'])

    # Get train, test, valid splits
    logger.info("Splitting data into train, test, valid ...")
    random.shuffle(abstracts)
    train_size = int(0.8 * len(abstracts))
    test_size = int(0.1 * len(abstracts))
    train, test, valid = abstracts[:train_size], abstracts[train_size:train_size + test_size], abstracts[
                                                                                               train_size + test_size:]

    train = train[0:10000]

    model_name: str = "all-mpnet-base-v2"
    logger.info(f"Embedding train data using full model {model_name} ...")
    full_model = SentenceTransformer(model_name, cache_folder=config.model_dir)
    start = time()
    full_emb = full_model.encode(train, convert_to_numpy=True)
    logger.info(f"Full model embedding time: {time() - start}s")

    edr_eval(train, full_emb, 128, model_name)
    edr_eval(train, full_emb, 64, model_name)
    edr_eval(train, full_emb, 32, model_name)
