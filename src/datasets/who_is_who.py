import os
import json
import pandas as pd
from ..shared import config
from ..shared.rdf_terms import RdfTerms
from .dataset import Dataset

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
        file_path = os.path.join(config.dataset_dir, 'IND-WhoIsWho/pid_to_info_all.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as file:
            data = json.load(file)

        triples = []
        # Iterate over each paper
        for paper_id, paper_info in data.items():
            triples.append((paper_id, RdfTerms.IDENTIFIER, paper_info['id']))
            triples.append((paper_id, RdfTerms.TITLE, paper_info['title']))
            triples.append((paper_id, RdfTerms.ABSTRACT, paper_info['abstract']))
            triples.append((paper_id, RdfTerms.VENUE, paper_info['venue']))
            triples.append((paper_id, RdfTerms.YEAR, paper_info['year']))

            for author in paper_info['authors']:
                triples.append((paper_id, RdfTerms.CREATOR, author['name']))
                triples.append((author['name'], RdfTerms.NAME, author['name']))
                triples.append((author['name'], RdfTerms.ORGANIZATION, author['org']))

            for keyword in paper_info['keywords']:
                triples.append((paper_id, RdfTerms.KEYWORD, keyword))

        df = pd.DataFrame(triples, columns=['h', 'r', 't'])
        return df

    @staticmethod
    def _parse_dict():
        file_path = os.path.join(config.dataset_dir, 'IND-WhoIsWho/pid_to_info_all.json')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, 'r') as file:
            data = json.load(file)

        if not data:
            raise ValueError(f"Unable to parse {os.path.join(config.dataset_dir, 'IND-WhoIsWho/pid_to_info_all.json')}.")

        return data
