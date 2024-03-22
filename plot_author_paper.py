import graph
import parser

# Load the paper author graph
g = parser.paper_author_graph()
graph.visualize(g)
