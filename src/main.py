import os
import argparse
from sentence_transformers import SentenceTransformer

from .database import populate_db
from .datasets import who_is_who, oc782k
from .model.GAT import train_gat
from .shared import config
from .pipeline.create_embeddings import create_embeddings

logger = config.get_logger("Main")


def populate_neo():
    raise NotImplementedError


def train_model():
    raise NotImplementedError


def evaluate_model():
    raise NotImplementedError


def embed_publications():
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the AND model, populate the Neo4j database.")

    # Arguments
    parser.add_argument(
        '--ds_stats', '-dss',
        action='store_true',
        help='Set to True to print dataset statistics.'
    )
    parser.add_argument(
        '--populate_neo', '-neo',
        action='store_true',
        help='Set to True to populate the Neo4j database.'
    )
    parser.add_argument(
        '--train', '-t',
        action='store_true',
        help='Set to True to train the AND model.'
    )
    parser.add_argument(
        '--eval', '-e',
        action='store_true',
        help='Set to True to evaluate the AND model.'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Create directories if they don't exist
    config.create_dirs()

    # Access the build argument
    if args.ds_stats:
        logger.info("Calculating dataset statistics.")
        raise NotImplementedError
    if args.populate_neo:
        logger.info("Populating the Neo4j database.")
        populate_neo()
    if args.train:
        logger.info("Training the AND model.")
        train_model()
    if args.eval:
        logger.info("Evaluating the AND model.")
        evaluate_model()
