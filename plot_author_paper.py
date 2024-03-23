import graph
import parser
import networkx as nx
import random
import config

logger = config.get_logger("Plot")

# Load the paper author graph
g = parser.paper_author_graph()

# Sample a subgraph using breadth-first search
subgraph = nx.bfs_tree(g, random.choice(list(g.nodes())), depth_limit=3)
subgraph = subgraph.to_undirected()
subgraph = g.subgraph(subgraph.nodes())

logger.debug(f"Number of nodes in the subgraph: {subgraph.number_of_nodes()}")
graph.visualize_dash(subgraph)
