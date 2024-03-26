import graph
from graph_db import GraphDB
import parser
import networkx as nx
import random
import config

logger = config.get_logger("Main")

# Create a new graph database
graph_db = GraphDB()


if __name__ == '__main__':
    # Create the paper graph
    parser.paper_neo4j(graph_db)

    # Close the database connection
    graph_db.close()
