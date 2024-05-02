from src.database import populate_db
from src.data_processing import parse_datasets
from shared import config
import argparse

logger = config.get_logger("Main")


def populate_neo():
    data = parse_datasets.parse_who_is_who('/data/IND-WhoIsWho/pid_to_info_all.json')
    populate_db.populate_who_is_who(data)
    pass


def train_model():
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
    if args.neo:
        logger.info("Populating the Neo4j database.")
        populate_neo()
    if args.train:
        logger.info("Training the AND model.")
        train_model()
    if args.eval:
        logger.info("Evaluating the AND model.")
        evaluate_model()
