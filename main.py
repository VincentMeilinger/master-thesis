import os
import argparse

from src.shared import (
    config,
    database_wrapper
)
from src.pipeline import (
    embed_nodes,
    transformer_dim_reduction,
    populate_db,
    dataset_pre_processing,
    train_gat,
    create_edges
)

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
        help='Set to True to create publication node embeddings.'
    )
    parser.add_argument(
        '--populate_neo', '-neo',
        action='store_true',
        help='Set to True to populate the Neo4j database.'
    )
    parser.add_argument(
        '--edges', '-edges',
        action='store_true',
        help='Set to True to create the edges of the neo4j KG.'
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
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Set to True to run whole pipeline.'
    )
    parser.add_argument(
        '--delete_db', '-del',
        action='store_true',
        help='Delete the Neo4J database.'
    )

    # Setup logging
    logger = config.get_logger("Main")

    # Parse the arguments
    args = parser.parse_args()

    config.init()

    # Access the build argument
    if args.clear_pipeline_state:
        logger.info("Clearing the pipeline state.")
        config.clear_pipeline_state()
    if args.delete_db:
        logger.info("Deleting the Neo4j database.")
        db = database_wrapper.DatabaseWrapper()
        db.delete_all_nodes()
    if args.ds_stats:
        logger.info("Calculating dataset statistics.")
        raise NotImplementedError
    if args.prepare_pipeline:
        logger.info("Preparing the pipeline.")
        transformer_dim_reduction.prep_transformer()

    # Run the whole pipeline
    if args.all:
        logger.info("No arguments provided. Running all steps.")
        dataset_pre_processing.dataset_pre_processing()
        populate_db.populate_db()
        embed_nodes.embed_nodes()
        create_edges.create_edges()
        train_gat.train_gat()
        exit(0)

    # Run individual steps
    if args.populate_neo:
        logger.info("Populating the Neo4j database.")
        populate_db.populate_db()
    if args.embed_publications:
        logger.info("Embedding publications.")
        embed_nodes.embed_nodes()
    if args.edges:
        logger.info("Creating the edges of the Neo4j KG.")
        create_edges.create_edges()
    if args.train:
        logger.info("Training the AND model.")
        train_gat.train_gat()
    if args.eval:
        logger.info("Evaluating the AND model.")
        raise NotImplementedError

