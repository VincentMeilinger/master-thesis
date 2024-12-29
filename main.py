import os
import json
import argparse
from sentence_transformers import SentenceTransformer

import pipeline_config
from src.datasets.who_is_who import WhoIsWhoDataset
from src.shared import (
    config,
    database_wrapper
)
from src.pipeline import (
    transformer_dim_reduction,
    create_nodes,
    link_nodes,
    prediction
)

def main():
    parser = argparse.ArgumentParser(description="Train the AND model, populate the Neo4j database.")

    # Arguments
    parser.add_argument(
        '--reset_state', '-reset',
        action='store_true',
        help='Set to True to print dataset statistics.'
    )
    parser.add_argument(
        '--delete_db', '-del',
        action='store_true',
        help='Delete the Neo4J database.'
    )
    parser.add_argument(
        '--prepare_pipeline', '-prep',
        action='store_true',
        help='Set to True to create the models needed for the pipeline.'
    )
    parser.add_argument(
        '--create_nodes', '-create',
        action='store_true',
        help='Set to True to populate the Neo4j database.'
    )
    parser.add_argument(
        '--prediction', '-pred',
        action='store_true',
        help='Set to True to disambiguate author names.'
    )


    # Setup logging
    logger = config.get_logger("Main")

    # Parse arguments
    args = parser.parse_args()

    # Initial setup
    config.create_dirs()
    config.print_config()

    configuration = pipeline_config.config

    if args.delete_db:
        logger.info("Deleting the Neo4j database.")
        db = database_wrapper.DatabaseWrapper(database=configuration["database"])
        db.delete_all_nodes()
        db.close()

    if args.prepare_pipeline:
        logger.info("Preparing the pipeline.")
        transformer_dim_reduction.prep_transformer(configuration=configuration)

    # Setup pipeline
    db = database_wrapper.DatabaseWrapper(database=configuration["database"])
    data = WhoIsWhoDataset.parse_data()
    training_data = WhoIsWhoDataset.parse_train()

    # Run pipelines
    if args.create_nodes:
        logger.info("Creating nodes.")

        model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device='cuda'
        )

        # Create nodes for the graph
        create_nodes.create_nodes(
            db=db,
            model=model,
            data=data,
            train_data=training_data,
            config=configuration,
        )

        # Link nodes by attribute similarity
        link_nodes.link_all_attributes(
            db=db,
            model=model,
            config=configuration
        )

        # Link nodes based on co-author overlap
        link_nodes.link_co_author_network(
            db=db,
            data=data,
            config=configuration,
        )

    if args.prediction:
        logger.info("Disambiguating authors.")
        prediction.predict(
            db=db,
            config=configuration,
        )


if __name__ == '__main__':
    main()