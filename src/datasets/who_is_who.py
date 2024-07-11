import os
import json
import pandas as pd
import uuid

from .dataset import Dataset
from ..shared import config
from ..shared.rdf_terms import PublicationRDF, AuthorRDF
from ..shared.database_wrapper import DatabaseWrapper

logger = config.get_logger("Dataset")


class WhoIsWhoDataset(Dataset):
    name: str = 'IND-WhoIsWho'

    @staticmethod
    def parse(format: str = 'triples'):
        """ Parse a dataset from raw data. Available formats: triples, dict."""
        if format == 'triples':
            return WhoIsWhoDataset._parse_triples()
        elif format == 'dict':
            return WhoIsWhoDataset._parse_dict()

    @staticmethod
    def _parse_triples():
        logger.info("Parsing IND-WhoIsWho")
        file_path = os.path.join(config.DATASET_DIR, 'IND-WhoIsWho/pid_to_info_all.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as file:
            data = json.load(file)

        triples = []
        # Iterate over each paper
        for paper_id, paper_info in data.items():
            triples.append((paper_id, PublicationRDF.IDENTIFIER, paper_info['id']))
            triples.append((paper_id, PublicationRDF.TITLE, paper_info['title']))
            triples.append((paper_id, PublicationRDF.ABSTRACT, paper_info['abstract']))
            triples.append((paper_id, PublicationRDF.VENUE, paper_info['venue']))
            triples.append((paper_id, PublicationRDF.YEAR_PUBLISHED, paper_info['year']))

            for author in paper_info['authors']:
                triples.append((paper_id, PublicationRDF.AUTHOR, author['name']))
                triples.append((author['name'], AuthorRDF.ORGANIZATION, author['org']))

            for keyword in paper_info['keywords']:
                triples.append((paper_id, PublicationRDF.KEYWORD, keyword))

        df = pd.DataFrame(triples, columns=['h', 'r', 't'])
        return df

    @staticmethod
    def _parse_dict():
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
    def paper_nodes_to_db(db: DatabaseWrapper, max_nodes: int, max_seq_len: int):
        """ Populate the database with the WhoIsWho dataset. """

        data = WhoIsWhoDataset.parse(format='dict')
        data = [value for key, value in data.items()]
        if max_nodes is not None:
            data = data[:max_nodes]

        for pub in data:
            pub_id = pub.pop('id')
            pub_data = {
                'title': pub['title'],
                'abstract': pub['abstract'][0:max_seq_len],
                'venue': pub['venue'],
                'year': pub['year'],
                'keywords': pub['keywords'],
            }
            db.merge_node("Publication", pub_id, pub_data)

    @staticmethod
    def parse_valid():
        """ Parse the WhoIsWho validation dataset. """
        file_path = os.path.join(config.DATASET_DIR, 'IND-WhoIsWho/ind_valid_author.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as file:
            data = json.load(file)

        return data

    @staticmethod
    def parse_train():
        """ Parse the WhoIsWho test dataset. """
        file_path = os.path.join(config.DATASET_DIR, 'IND-WhoIsWho/train_author.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as file:
            data = json.load(file)

        return data

    def parse_graph(self):
        node_data = self.parse(format='dict')
        true_authors = self.parse_train()

        authors = []
        publications = []
        orgs = set()
        edges = set()
        for author_id, values in node_data.items():
            pub_authors = values.pop('authors')

            # Publication node
            publications.append(values)

            for pub_author in pub_authors:
                author_node = {
                    'id': uuid.uuid4(),
                    'name': pub_author['name']
                }
                org_node = pub_author['org']

                # Author node
                authors.append(author_node)
                # Organization node
                orgs.add(org_node)

                # Author -> Organization
                edges.add((author_node['id'], AuthorRDF.ORGANIZATION, org_node))
                # Author -> Publication
                edges.add((author_node['id'], AuthorRDF.PUBLICATION, values['id']))
                # Publication -> Author
                edges.add((values['id'], PublicationRDF.AUTHOR, author_node['id']))

        for author_id, values in true_authors.items():



