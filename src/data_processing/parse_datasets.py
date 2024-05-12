import json
import networkx as nx
from ..shared import config
from .graph_dataset import GraphDataset
from prettytable import PrettyTable
from typing import List
import pandas as pd
from enum import Enum




logger = config.get_logger("Parser")


def paper_author_graph(file_path='data/IND-WhoIsWho/pid_to_info_all.json', include_info=False):
    logger.info("Creating paper author graph")
    g = nx.Graph()

    logger.debug(f"Loading data from {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterate over each paper
    for paper_id, paper_info in data.items():
        # Add the paper node with or without attributes
        if include_info:
            g.add_node(paper_info["title"], type='paper', **paper_info)
        else:
            g.add_node(paper_info["title"], type='paper')

        # Iterate over the authors of the paper
        for author in paper_info['authors']:
            author_id = author['name']

            if not g.has_node(author_id):
                g.add_node(author_id, type='author', org=author['org'])
            # Add an edge between the author and the paper
            g.add_edge(author_id, paper_id)

    logger.debug("Graph created")
    logger.debug(f"Number of nodes: {g.number_of_nodes()}")
    logger.debug(f"Number of edges: {g.number_of_edges()}")

    return g


def parse_who_is_who(file_path='data/IND-WhoIsWho/pid_to_info_all.json'):
    logger.info("Parsing papers")
    logger.debug(f"Loading data from {file_path}")

    # Iterate over each paper
    for paper_id, paper_info in data.items():
        # Add the paper node with or without attributes
        if include_info:
            g.add_node(paper_info["title"], type='paper', **paper_info)
        else:
            g.add_node(paper_info["title"], type='paper')

        # Iterate over the authors of the paper
        for author in paper_info['authors']:
            author_id = author['name']

            if not g.has_node(author_id):
                g.add_node(author_id, type='author', org=author['org'])
            # Add an edge between the author and the paper
            g.add_edge(author_id, paper_id)

    return data


def parse_oc782k_and_eval(file_path='data/OC-782K/and_eval.json'):
    logger.info("Parsing papers")
    logger.debug(f"Loading data from {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data


def parse_oc782k(file_path='data/OC-782K/'):
    logger.info("Parsing dataset OC-782K ...")
    logger.debug(f"Loading data from {file_path}")
    data = {'train': None, 'test': None, 'valid': None}
    files = {'training.txt': 'train', 'testing.txt': 'test', 'validation.txt': 'valid'}
    for file in files.keys():
        data[files[file]] = pd.read_csv(file_path + file, sep='\t', header=None, names=['h', 'r', 't'])

    return GraphDataset('OC-782K', data['train'], data['test'], data['valid'])


def print_ds_stats(datasets: List[GraphDataset]):
    table = PrettyTable()
    table.field_names = ["Dataset", "Train Entities", "Test Entities", "Valid Entities", "Total Distinct Entities", "Total Relations"]
    for dataset in datasets:
        num_train = dataset.count_distinct_entities(split='train')
        num_train = num_train/1000 if num_train is not None else 0
        num_test = dataset.count_distinct_entities(split='test')
        num_test = num_test/1000 if num_test is not None else 0
        num_valid = dataset.count_distinct_entities(split='valid')
        num_valid = num_valid/1000 if num_valid is not None else 0
        num_total = dataset.count_distinct_entities()
        num_total = num_total/1000 if num_total is not None else 0
        num_citations = dataset.count_citations()
        num_citations = num_citations/1000 if num_citations is not None else 0

        # Add row to the table
        table.add_row([
            dataset.name,
            f"{int(num_train)} K",
            f"{int(num_test)} K",
            f"{int(num_valid)} K",
            f"{int(num_total)} K",
            f"{int(num_citations)} K"
        ])
    print(table)


if __name__ == '__main__':
    oc_data = parse_oc782k()
    print_ds_stats([oc_data])
