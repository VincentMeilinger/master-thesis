import networkx as nx
from dash import Dash, html
import dash_cytoscape as cyto
from src.shared import config

logger = config.get_logger("Graph")


def nx_to_cytoscape(graph):
    logger.info("Converting NetworkX graph to Cytoscape format")
    cy_graph = nx.readwrite.json_graph.cytoscape_data(graph)
    return cy_graph


def visualize_dash(graph):
    logger.info("Visualizing the graph")
    cy_graph = nx_to_cytoscape(graph)
    app = Dash(__name__)

    app.layout = html.Div([
        html.P("Paper author graph:"),
        cyto.Cytoscape(
            id='cytoscape',
            layout={'name': 'cose'},
            style={'width': '100%', 'height': '100%'},
            elements=cy_graph['elements'],
        )
    ])

    app.run_server(debug=True, port=8051)
    return cy_graph
