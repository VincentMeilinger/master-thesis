import os
import torch
import random
import numpy as np
from time import time
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, models

from ..shared import config
from ..shared.run_config import RunConfig
from ..shared.pipeline_state import PipelineState
from ..datasets.who_is_who import WhoIsWhoDataset

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
    model = SentenceTransformer(
        model_name,
        device=config.DEVICE
    )

    # PCA on train embeddings
    logger.info("Performing PCA on train embeddings ...")
    pca = PCA(n_components=new_dimension)
    pca.fit(full_emb)
    pca_comp = np.asarray(pca.components_)

    # Add a dense layer to the model
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

    # Store the model on disc
    model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    os.makedirs("data/models", exist_ok=True)
    model.save(f"data/models/{model_name}-{new_dimension}dim")


def prep_transformer():
    run_config = RunConfig(config.RUN_DIR)
    state = PipelineState(config.RUN_ID, config.RUN_DIR)

    if state.transformer_dim_reduction_state.state == 'completed':
        logger.info("Transformer dimensionality reduction already completed. Skipping ...")
        return
    logger.info("Adding a dense layer to the transformer model to reduce the embedding dimension ...")

    # Parse datasets
    abstracts = []
    data = WhoIsWhoDataset.parse(format='dict')

    for paper_id, paper_info in data.items():
        abstracts.append(paper_info['abstract'])

    # Get train, test, valid splits
    logger.info("Splitting data into train, test, valid ...")
    random.shuffle(abstracts)
    train_size = int(0.8 * len(abstracts))
    test_size = int(0.1 * len(abstracts))
    train, test, valid = abstracts[:train_size], abstracts[train_size:train_size + test_size], abstracts[
                                                                                               train_size + test_size:]
    max_samples = int(run_config.transformer_dim_reduction.num_pca_samples)
    train = train[0:max_samples]

    # Embed train data using full model for comparison
    logger.info(f"Embedding train data using full model {run_config.transformer_dim_reduction.base_model} ...")
    full_model = SentenceTransformer(
        run_config.transformer_dim_reduction.base_model,
        cache_folder=config.MODEL_DIR,
        device=config.DEVICE
    )
    start = time()
    full_emb = full_model.encode(train, convert_to_numpy=True)
    logger.info(f"Full model embedding time: {time() - start}s")

    # Reduce the dimensionality of the embeddings, evaluate the performance compared to the full model
    edr_eval(
        train,
        full_emb,
        new_dimension=run_config.transformer_dim_reduction.reduced_dim,
        model_name=run_config.transformer_dim_reduction.base_model
    )
    state.transformer_dim_reduction_state.state = 'completed'
    state.save()
