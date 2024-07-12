import os
import json
import pandas as pd
import uuid

from .dataset import Dataset
from ..shared import config
from ..shared.graph_schema import GraphNode, PublicationEdge, AuthorEdge
from ..shared.database_wrapper import DatabaseWrapper

logger = config.get_logger("Dataset")


class WhoIsWhoDataset(Dataset):
    name: str = 'IND-WhoIsWho'

    @staticmethod
    def parse_data() -> dict:
        file_path = os.path.join(config.DATASET_DIR, 'IND-WhoIsWho/pid_to_info_all.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as file:
            data = json.load(file)

        if not data:
            raise ValueError(
                f"Unable to parse {os.path.join(config.DATASET_DIR, 'IND-WhoIsWho/pid_to_info_all.json')}.")

        return data

    @staticmethod
    def parse_valid() -> dict:
        """ Parse the WhoIsWho validation dataset. """
        file_path = os.path.join(config.DATASET_DIR, 'IND-WhoIsWho/ind_valid_author.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as file:
            data = json.load(file)

        return data

    @staticmethod
    def parse_train() -> dict:
        """ Parse the WhoIsWho test dataset. """
        file_path = os.path.join(config.DATASET_DIR, 'IND-WhoIsWho/train_author.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as file:
            data = json.load(file)

        return data

    @staticmethod
    def parse_graph(max_iterations: int = None):
        logger.info("Parsing WhoIsWho dataset, merging data into database ...")
        db = DatabaseWrapper()
        node_data= WhoIsWhoDataset.parse_data()
        true_author_data = WhoIsWhoDataset.parse_train()

        logger.info("Merging file 'pid_to_info_all.json' ...")
        current_iteration = 0
        for author_id, values in node_data.items():
            if max_iterations is not None and current_iteration >= max_iterations:
                break
            current_iteration += 1

            # Prepare publication data
            pub_authors = values.pop('authors')
            venue = values.pop('venue')

            # Publication node
            db.merge_node(GraphNode.PUBLICATION.name, values['id'], values)

            if venue:
                # Venue node
                db.merge_node(GraphNode.VENUE.name, venue, {})
                # Publication -> Venue
                db.merge_edge(GraphNode.PUBLICATION.name, values['id'], GraphNode.VENUE.name, venue, PublicationEdge.VENUE.name)

            for pub_author in pub_authors:
                if not pub_author['name']:
                    continue

                author_node = {
                    'id': str(uuid.uuid4()),
                    'name': pub_author['name']
                }

                # Author node
                db.merge_node(GraphNode.AUTHOR.name, author_node['id'], author_node)
                # Author -> Publication
                db.merge_edge(GraphNode.AUTHOR.name, author_node['id'], GraphNode.PUBLICATION.name, values['id'], AuthorEdge.PUBLICATION.name)
                # Publication -> Author
                db.merge_edge(GraphNode.PUBLICATION.name, values['id'], GraphNode.AUTHOR.name, author_node['id'], PublicationEdge.AUTHOR.name)

                if pub_author['org']:
                    # Organization node
                    db.merge_node(GraphNode.ORGANIZATION.name, pub_author['org'])
                    # Author -> Organization
                    db.merge_edge(GraphNode.AUTHOR.name, author_node['id'], GraphNode.ORGANIZATION.name, pub_author['org'], AuthorEdge.ORGANIZATION.name)

        logger.info("Merging true author data into database ...")
        for author_id, values in true_author_data.items():
            true_author_node = {
                'id': author_id,
                'name': values['name']
            }

            # True Author node
            db.merge_node(GraphNode.TRUE_AUTHOR.name, true_author_node['id'], true_author_node)

            for pub_id in values['normal_data']:
                # Publication -> True Author
                db.merge_edge(GraphNode.PUBLICATION.name, pub_id, GraphNode.TRUE_AUTHOR.name, true_author_node['id'], PublicationEdge.TRUE_AUTHOR.name)

        logger.info("Finished merging WhoIsWho dataset.")
