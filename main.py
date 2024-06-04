import os
import argparse
from src.shared import config
from src.pipeline.embed_datasets import embed_datasets
from src.pipeline.prep_transformer import prep_transformer
logger = config.get_logger("Main")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the AND model, populate the Neo4j database.")

    # Arguments
    parser.add_argument(
        '--clear_pipeline_state', '-clear',
        action='store_true',
        help='Set to True to print dataset statistics.'
    )
    parser.add_argument(
        '--prepare_pipeline', '-prep',
        action='store_true',
        help='Set to True to create the models needed for the pipeline.'
    )
    parser.add_argument(
        '--ds_stats', '-dss',
        action='store_true',
        help='Set to True to print dataset statistics.'
    )
    parser.add_argument(
        '--embed_publications', '-embed',
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

    # Access the build argument
    if args.clear_pipeline_state:
        logger.info("Clearing the pipeline state.")
        raise NotImplementedError
    if args.ds_stats:
        logger.info("Calculating dataset statistics.")
        raise NotImplementedError
    if args.prepare_pipeline:
        logger.info("Preparing the pipeline.")
        prep_transformer()
    if args.embed_publications:
        logger.info("Embedding publications.")
        embed_datasets()
    if args.populate_neo:
        logger.info("Populating the Neo4j database.")
        raise NotImplementedError
    if args.train:
        logger.info("Training the AND model.")
        raise NotImplementedError
    if args.eval:
        logger.info("Evaluating the AND model.")
        raise NotImplementedError
