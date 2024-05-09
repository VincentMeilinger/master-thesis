from .database import populate_db
from .data_processing import parse_datasets
from .model.GAT import train_gat
from .shared import config
import argparse
import os

logger = config.get_logger("Main")


def populate_neo():
    who_is_who_path = os.path.join(config.data_dir, 'IND-WhoIsWho/pid_to_info_all.json')
    data = parse_datasets.parse_who_is_who(who_is_who_path)
    populate_db.populate_who_is_who(data)
    pass


def train_model(data):
    train_gat.train()
    pass


def evaluate_model():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the AND model, populate the Neo4j database.")

    # Arguments
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

    # Access the build argument
    if args.populate_neo:
        logger.info("Populating the Neo4j database.")
        populate_neo()
    if args.train:
        logger.info("Training the AND model.")
        train_model()
    if args.eval:
        logger.info("Evaluating the AND model.")
        evaluate_model()