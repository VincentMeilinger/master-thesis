import os
import json
import pandas as pd
import uuid

from .dataset import Dataset
from ..shared import config
from ..shared.graph_schema import NodeType, PublicationEdge, AuthorEdge
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
            db.merge_node(NodeType.PUBLICATION, values['id'], values)

            if venue:
                venue_node = {
                    'id': str(uuid.uuid4()),
                    'name': venue
                }

                # Venue node
                db.merge_node(NodeType.VENUE, venue_node['id'], venue_node)
                # Publication -> Venue
                db.merge_edge(NodeType.PUBLICATION, values['id'], NodeType.VENUE, venue, PublicationEdge.VENUE)

            for ix, pub_author in enumerate(pub_authors):
                if not pub_author['name']:
                    continue

                author_node = {
                    'id': str(uuid.uuid4()),
                    'name': pub_author['name']
                }

                # Author node
                if ix==0:
                    db.merge_node(NodeType.AUTHOR, author_node['id'], author_node)
                else:
                    db.merge_node(NodeType.CONTRIBUTOR, author_node['id'], author_node)
                # Author -> Publication
                db.merge_edge(NodeType.AUTHOR, author_node['id'], NodeType.PUBLICATION, values['id'], AuthorEdge.PUBLICATION)
                # Publication -> Author
                db.merge_edge(NodeType.PUBLICATION, values['id'], NodeType.AUTHOR, author_node['id'], PublicationEdge.AUTHOR)

                if pub_author['org']:
                    org_node = {
                        'id': str(uuid.uuid4()),
                        'name': pub_author['org']
                    }

                    # Organization node
                    db.merge_node(NodeType.ORGANIZATION, org_node['id'], org_node)
                    # Author -> Organization
                    db.merge_edge(NodeType.AUTHOR, author_node['id'], NodeType.ORGANIZATION, pub_author['org'], AuthorEdge.ORGANIZATION)

        logger.info("Merging true author data into database ...")
        for author_id, values in true_author_data.items():
            true_author_node = {
                'id': author_id,
                'name': values['name']
            }

            # True Author node
            db.merge_node(NodeType.TRUE_AUTHOR, true_author_node['id'], true_author_node)

            for pub_id in values['normal_data']:
                # Publication -> True Author
                db.merge_edge(NodeType.PUBLICATION, pub_id, NodeType.TRUE_AUTHOR, true_author_node['id'], PublicationEdge.TRUE_AUTHOR)

        logger.info("Finished merging WhoIsWho dataset.")
