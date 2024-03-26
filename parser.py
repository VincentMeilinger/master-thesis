import json
import networkx as nx
import config
from graph_db import GraphDB

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


def paper_neo4j(dbm: GraphDB, file_path='data/IND-WhoIsWho/pid_to_info_all.json'):
    logger.info("Parsing papers")
    logger.debug(f"Loading data from {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterate over each paper
    logger.debug("Creating nodes in neo4j graph database")
    for paper_id, paper_info in data.items():
        # Add the paper node with or without attributes
        authors = paper_info.pop('authors')
        dbm.merge_paper(paper_info)
        for author_info in authors:
            dbm.merge_author(author_info)
            dbm.merge_author_paper_relationship(
                author_info['name'],
                author_info['org'],
                paper_info['id']
            )

    logger.debug("Graph created")


if __name__ == '__main__':
    paper_author_graph()
