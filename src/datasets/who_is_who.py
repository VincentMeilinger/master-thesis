import os
import json
import time

import mmh3
import uuid
from tqdm import tqdm
from collections import defaultdict

from .dataset import Dataset
from ..shared import config
from ..shared.graph_schema import NodeType, EdgeType
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

    """
    
    @staticmethod
    def parse_graph(max_iterations: int = None):
        logger.info("Parsing WhoIsWho dataset, merging data into database ...")
        db = DatabaseWrapper()
        node_data = WhoIsWhoDataset.parse_data()
        true_author_data = WhoIsWhoDataset.parse_train()

        logger.info("Merging file 'pid_to_info_all.json' ...")
        total_entries = len(node_data.keys()) if max_iterations is None else max_iterations
        current_iteration = 0
        start_time = time.time()
        for author_id, values in node_data.items():
            if max_iterations is not None and current_iteration >= max_iterations:
                break

            current_iteration += 1
            if current_iteration % 1000 == 0:
                logger.info(f"Progress: {current_iteration / total_entries * 100:.2f}%, estimated time remaining: {((time.time() - start_time) / current_iteration) * (total_entries - current_iteration):.2f}s")

            # Prepare publication data
            pub_authors = values.pop('authors')
            venue = values.pop('venue')

            # Publication node
            db.merge_node(NodeType.PUBLICATION, values['id'], values)

            if venue:
                venue_node = {
                    'id': str(mmh3.hash(venue)),
                    'name': venue.lower()
                }

                # Venue node
                db.merge_node(NodeType.VENUE, venue_node['id'], venue_node)
                # Publication -> Venue
                db.merge_edge(NodeType.PUBLICATION, values['id'], NodeType.VENUE, venue_node['id'], EdgeType.PUB_VENUE)
                db.merge_edge(NodeType.VENUE, venue_node['id'], NodeType.PUBLICATION, values['id'], EdgeType.VENUE_PUB)

            for ix, pub_author in enumerate(pub_authors):
                if not pub_author['name']:
                    continue

                author_node = {
                    'id': str(uuid.uuid4()),
                    'name': pub_author['name'].lower()
                }

                # Author node
                if ix == 0:
                    node_type = NodeType.AUTHOR
                else:
                    node_type = NodeType.CO_AUTHOR

                db.merge_node(node_type, author_node['id'], author_node)

                # Author -> Publication
                db.merge_edge(node_type, author_node['id'], NodeType.PUBLICATION, values['id'],
                              EdgeType.AUTHOR_PUB)
                # Publication -> Author
                db.merge_edge(NodeType.PUBLICATION, values['id'], node_type, author_node['id'],
                              EdgeType.PUB_AUTHOR)

                if pub_author['org']:
                    org_node = {
                        'id': str(mmh3.hash(venue)),
                        'name': pub_author['org'].lower()
                    }

                    # Organization node
                    db.merge_node(NodeType.ORGANIZATION, org_node['id'], org_node)
                    # Author -> Organization
                    db.merge_edge(node_type, author_node['id'], NodeType.ORGANIZATION, org_node['id'],
                                  EdgeType.AUTHOR_ORG)
                    # Organization -> Author
                    db.merge_edge(NodeType.ORGANIZATION, org_node['id'], node_type, author_node['id'],
                                  EdgeType.ORG_AUTHOR)
                    # Publication -> Organization
                    db.merge_edge(NodeType.PUBLICATION, values['id'], NodeType.ORGANIZATION, org_node['id'],
                                  EdgeType.PUB_ORG)
                    # Organization -> Publication
                    db.merge_edge(NodeType.ORGANIZATION, org_node['id'], NodeType.PUBLICATION, values['id'],
                                  EdgeType.ORG_PUB)

        # Merge true author data
        logger.info("Merging true author data into database ...")
        for author_id, values in true_author_data.items():
            for pub_id in values['normal_data']:
                db.merge_properties(NodeType.PUBLICATION, pub_id, {'true_author': author_id})
            
        logger.info("Finished merging WhoIsWho dataset.")
    """

    @staticmethod
    def parse_graph(max_iterations: int = None, batch_size: int = 1000):
        logger.debug("Parsing WhoIsWho dataset, merging data into database ...")
        db = DatabaseWrapper()
        node_data = WhoIsWhoDataset.parse_data()
        true_author_data = WhoIsWhoDataset.parse_train()

        logger.debug("Merging file 'pid_to_info_all.json' ...")
        total_entries = len(node_data.keys()) if max_iterations is None else max_iterations
        current_iteration = 0

        batch_nodes = defaultdict(list)
        batch_edges = defaultdict(list)

        def process_batch():
            logger.debug(f"Merging batch of {batch_size} entries into the database ...")
            for node_type, nodes in batch_nodes.items():
                logger.debug(f"Merging {len(nodes)} nodes of type {node_type.value} ...")
                if nodes:
                    db.merge_nodes(node_type, nodes)
            for edge_type, edges in batch_edges.items():
                logger.debug(f"Merging {len(edges)} edges of type {edge_type.value} ...")
                if edges:
                    start_type, end_type = edge_type.start_end()
                    db.merge_edges(start_type, end_type, edge_type, edges)
            batch_nodes.clear()
            batch_edges.clear()

        with tqdm(total=total_entries, desc="Merging WhoIsWho pid_to_info_all.json") as pbar:
            for author_id, values in node_data.items():
                if max_iterations is not None and current_iteration >= max_iterations:
                    break

                if current_iteration % batch_size == 0 and current_iteration > 0:
                    pbar.update(batch_size)
                    process_batch()

                current_iteration += 1

                pub_authors = values.pop('authors')
                venue = values.pop('venue')

                # Publication node
                batch_nodes[NodeType.PUBLICATION].append(values)

                if venue:
                    venue_node = {
                        'id': str(mmh3.hash(venue)),
                        'name': venue.lower()
                    }
                    # Venue node
                    batch_nodes[NodeType.VENUE].append(venue_node)
                    # Publication -> Venue
                    batch_edges[EdgeType.PUB_VENUE].append((values['id'], venue_node['id']))
                    batch_edges[EdgeType.VENUE_PUB].append((venue_node['id'], values['id']))

                for ix, pub_author in enumerate(pub_authors):
                    if not pub_author['name']:
                        continue

                    author_node = {
                        'id': str(uuid.uuid4()),
                        'name': pub_author['name'].lower()
                    }

                    batch_nodes[NodeType.AUTHOR].append(author_node)

                    # Co-/Author -> Publication
                    batch_edges[EdgeType.AUTHOR_PUB].append((author_node['id'], values['id']))
                    # Publication -> Co-/Author
                    batch_edges[EdgeType.PUB_AUTHOR].append((values['id'], author_node['id']))

                    if pub_author['org']:
                        org_node = {
                            'id': str(mmh3.hash(pub_author['org'])),
                            'name': pub_author['org'].lower()
                        }

                        batch_nodes[NodeType.ORGANIZATION].append(org_node)
                        # Author -> Organization
                        batch_edges[EdgeType.AUTHOR_ORG].append((author_node['id'], org_node['id']))
                        # Organization -> Author
                        batch_edges[EdgeType.ORG_AUTHOR].append((org_node['id'], author_node['id']))
                        # Publication -> Organization
                        batch_edges[EdgeType.PUB_ORG].append((values['id'], org_node['id']))
                        # Organization -> Publication
                        batch_edges[EdgeType.ORG_PUB].append((org_node['id'], values['id']))

        # Merge true author data
        logger.debug("Merging true author data into database ...")
        props = []
        with tqdm(total=len(true_author_data.items()), desc="Merging WhoIsWho train_author.json") as pbar:
            for author_id, values in true_author_data.items():
                author_name = values['name']
                for pub_id in values['normal_data']:
                    props.append({'id': pub_id, 'properties': {'true_author': author_name}})
                pbar.update(1)
                if len(props) > batch_size:
                    db.merge_properties_batch(NodeType.PUBLICATION, props)
                    props.clear()

            if props:
                db.merge_properties_batch(NodeType.PUBLICATION, props)

        db.close()
